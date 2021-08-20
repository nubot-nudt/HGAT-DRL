import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY, ActionAW, ActionDiff
from crowd_sim.envs.utils.state import tensor_to_joint_state
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor_batch
from crowd_nav.policy.graph_model import RGL,GAT_RL
from crowd_nav.policy.value_estimator import DQNNetwork, Noisy_DQNNetwork


class TreeSearchRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TreeSearchRL'
        self.trainable = True
        self.multiagent_training = True
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.action_space = None
        self.rotation_constraint = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.robot_state_dim = 9
        self.human_state_dim = 5
        self.v_pref = 1
        self.share_graph_model = None
        self.value_estimator = None
        self.linear_state_predictor = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.sparse_speed_samples = 2
        self.sparse_rotation_samples = 8
        self.action_group_index = []
        self.traj = None
        self.use_noisy_net = False
        self.count=0

    def configure(self, config, device):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.do_action_clip = config.model_predictive_rl.do_action_clip
        if hasattr(config.model_predictive_rl, 'sparse_search'):
            self.sparse_search = config.model_predictive_rl.sparse_search
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor
        # self.set_device(device)
        self.device = device


        if self.linear_state_predictor:
            self.state_predictor = LinearStatePredictor_batch(config, self.time_step)
            graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
            self.value_estimator = DQNNetwork(config, graph_model)
            self.model = [graph_model, self.value_estimator.value_network]
        else:
            if self.share_graph_model:
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = DQNNetwork(config, graph_model)
                self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
            else:
                graph_model1 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = DQNNetwork(config, graph_model1)
                graph_model2 = GAT_RL(config, self.robot_state_dim, self.human_state_dim)
                self.state_predictor = StatePredictor(config, graph_model2, self.time_step)
                self.model = [graph_model1, graph_model2, self.value_estimator.value_network,
                              self.state_predictor.human_motion_predictor]

        logging.info('Planning depth: {}'.format(self.planning_depth))
        logging.info('Planning width: {}'.format(self.planning_width))
        logging.info('Sparse search: {}'.format(self.sparse_search))

        if self.planning_depth > 1 and not self.do_action_clip:
            logging.warning('Performing d-step planning without action space clipping!')

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_device(self, device):
        self.device = device
        for model in self.model:
            model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_noisy_net(self, use_noisy_net):
        self.use_noisy_net = use_noisy_net

    def set_time_step(self, time_step):
        self.time_step = time_step
        self.state_predictor.time_step = time_step

    def get_normalized_gamma(self):
        return pow(self.gamma, self.time_step * self.v_pref)

    def get_model(self):
        return self.value_estimator

    def get_state_dict(self):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
            else:
                return {
                    'graph_model1': self.value_estimator.graph_model.state_dict(),
                    'graph_model2': self.state_predictor.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict(),
                    'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
                }
        else:
            return {
                    'graph_model': self.value_estimator.graph_model.state_dict(),
                    'value_network': self.value_estimator.value_network.state_dict()
                }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        if self.state_predictor.trainable:
            if self.share_graph_model:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            else:
                self.value_estimator.graph_model.load_state_dict(state_dict['graph_model1'])
                self.state_predictor.graph_model.load_state_dict(state_dict['graph_model2'])

            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])
            self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])
        else:
            self.value_estimator.graph_model.load_state_dict(state_dict['graph_model'])
            self.value_estimator.value_network.load_state_dict(state_dict['value_network'])

    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def build_action_space(self, acc_max):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        # currently, acc_max is 1, and the number of speed samples is 5.
        # currently, rotation constraint is np.pi/3, and rotation_samples is 5
        left_accs = np.linspace(-acc_max, acc_max, self.speed_samples)
        right_accs = np.linspace(-acc_max, acc_max, self.speed_samples)
        action_space = []
        for i, left_acc in enumerate(left_accs):
            for j, right_acc in enumerate(right_accs):
                action_index = i * self.speed_samples + j
                self.action_group_index.append(action_index)
                action_space.append(ActionDiff(left_acc, right_acc))
        self.speeds = left_accs
        self.rotations = right_accs
        self.action_space = action_space

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        self.count=self.count+1
        if self.count == 41:
            print('debug')
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        # if self.reach_destination(state):
        #     return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(1.0)
        max_action = None
        origin_max_value = float('-inf')
        state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon and self.use_noisy_net is False:
            max_action_index = np.random.choice(len(self.action_space))
            max_action = self.action_space[max_action_index]
            self.last_state = self.transform(state)
            return max_action, max_action_index
        else:
            max_value, max_action_index, max_traj = self.V_planning(state_tensor, self.planning_depth, self.planning_width)
            if max_value[0] > origin_max_value:
                max_action = self.action_space[max_action_index[0]]
            if max_action is None:
                raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.last_state = self.transform(state)
            self.traj = max_traj[0]
        return max_action, int(max_action_index[0])

    def V_planning(self, state, depth, width):
        """ Plans n steps into future based on state action value function. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples
        """
        # current_state_value = self.value_estimator(state)
        robot_state_batch = state[0]
        human_state_batch = state[1]
        if depth == 0:
            q_value = torch.Tensor(self.value_estimator(state))
            max_action_value, max_action_indexes = torch.max(q_value, dim=1)
            trajs = []
            for i in range(robot_state_batch.shape[0]):
                cur_state = (robot_state_batch[i, :, :].unsqueeze(0), human_state_batch[i, :, :].unsqueeze(0))
                trajs.append([(cur_state, None, None)])
            return max_action_value, max_action_indexes, trajs
        else:
            q_value = torch.Tensor(self.value_estimator(state))
            max_action_value, max_action_indexes = torch.topk(q_value, width, dim=1)
        # predict next human state
        action_stay = None
        _, pre_next_state = self.state_predictor(state, action_stay)
        next_robot_state_batch = None
        next_human_state_batch = None
        reward_est = torch.zeros(state[0].shape[0], width) * float('inf')

        for i in range(robot_state_batch.shape[0]):
            cur_state = (robot_state_batch[i, :, :].unsqueeze(0), human_state_batch[i, :, :].unsqueeze(0))
            next_human_state = pre_next_state[i, :, :].unsqueeze(0)
            for j in range(width):
                cur_action = self.action_space[max_action_indexes[i][j]]
                next_robot_state = self.compute_next_robot_state(cur_state[0], cur_action)
                if next_robot_state_batch is None:
                    next_robot_state_batch = next_robot_state
                    next_human_state_batch = next_human_state
                else:
                    next_robot_state_batch = torch.cat((next_robot_state_batch, next_robot_state), dim=0)
                    next_human_state_batch = torch.cat((next_human_state_batch, next_human_state), dim=0)
                reward_est[i][j] = self.reward_estimator.estimate_reward_on_predictor(
                    tensor_to_joint_state(cur_state), tensor_to_joint_state((next_robot_state, next_human_state)))
        next_state_batch = (next_robot_state_batch, next_human_state_batch)
        if self.planning_depth - depth >= 2 and self.planning_depth > 2:
            cur_width = 1
        else:
            cur_width = int(self.planning_width/2)
        next_values, next_action_indexes, next_trajs = self.V_planning(next_state_batch, depth-1, cur_width)
        next_values = next_values.view(state[0].shape[0], width)
        returns = (reward_est + self.get_normalized_gamma()*next_values + max_action_value) / 2

        max_action_return, max_action_index = torch.max(returns, dim=1)
        trajs = []
        max_returns = []
        max_actions = []
        for i in range(robot_state_batch.shape[0]):
            cur_state = (robot_state_batch[i, :, :].unsqueeze(0), human_state_batch[i, :, :].unsqueeze(0))
            action_id = max_action_index[i]
            trajs_id = i * width + action_id
            action = max_action_indexes[i][action_id]
            next_traj = next_trajs[trajs_id]
            trajs.append([(cur_state, action, reward_est)] + next_traj)
            max_returns.append(max_action_return[i].data)
            max_actions.append(action)
        max_returns = torch.tensor(max_returns)
        return max_returns, max_actions, trajs

    def compute_next_robot_state(self, robot_state, action):
        if robot_state.shape[0] != 1:
            raise NotImplementedError
        next_state = robot_state.clone().squeeze()
        left_acc = action.al
        right_acc = action.ar
        if self.kinematics == 'holonomic':
            next_state[0] = next_state[0] + action.vx * self.time_step
            next_state[1] = next_state[1] + action.vy * self.time_step
            next_state[2] = action.vx
            next_state[3] = action.vy
        elif self.kinematics == 'differential':
            # px, py, v_left, v_right, radius, gx, gy, vel_max, heading
            next_state[2] = next_state[2] + left_acc * self.time_step
            next_state[3] = next_state[3] + right_acc * self.time_step
            if np.abs(next_state[2]) > next_state[7]:
                next_state[2] = next_state[2] * next_state[7] / np.abs(next_state[2])
            if np.abs(next_state[3]) > next_state[7]:
                next_state[3] = next_state[3] * next_state[7] / np.abs(next_state[3])
            angular_vel = (next_state[2] - next_state[3]) / 2.0 / next_state[4]
            linear_vel = (next_state[2] + next_state[3]) / 2.0
            vx = linear_vel * np.cos(next_state[8])
            vy = linear_vel * np.sin(next_state[8])
            next_state[0] = next_state[0] + vx * self.time_step
            next_state[1] = next_state[1] + vy * self.time_step
            next_state[8] = (next_state[8] + angular_vel * self.time_step) % (2 * np.pi)
        return next_state.unsqueeze(0).unsqueeze(0)
    def get_attention_weights(self):
        return self.value_estimator.graph_model.attention_weights