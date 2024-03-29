import logging
import torch
import numpy as np
from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import tensor_to_joint_state_2types
from crowd_nav.policy.value_estimator import ValueEstimator
from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor_batch
from crowd_nav.policy.graph_model import RGL,GAT_RL


class ModelPredictiveRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ModelPredictiveRL'
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
            self.value_estimator = ValueEstimator(config, graph_model)
            self.model = [graph_model, self.value_estimator.value_network]
        else:
            if self.share_graph_model:
                graph_model = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = ValueEstimator(config, graph_model)
                self.state_predictor = StatePredictor(config, graph_model, self.time_step)
                self.model = [graph_model, self.value_estimator.value_network, self.state_predictor.human_motion_predictor]
            else:
                graph_model1 = RGL(config, self.robot_state_dim, self.human_state_dim)
                self.value_estimator = ValueEstimator(config, graph_model1)
                graph_model2 = RGL(config, self.robot_state_dim, self.human_state_dim)
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

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        # speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        speeds = [(i + 1) / self.speed_samples * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            if self.rotation_constraint == np.pi:
                rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples, endpoint=False)
            else:
                rotations = np.linspace(-self.rotation_constraint, self.rotation_constraint, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        self.action_group_index.append(0)
        for j, speed in enumerate(speeds):
            for i, rotation in enumerate(rotations):
                action_index = j * self.rotation_samples + i + 1
                self.action_group_index.append(action_index)
                if holonomic:
                    action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
                else:
                    action_space.append(ActionRot(speed, rotation))
        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.robot_state.v_pref)

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action_index = np.random.choice(len(self.action_space))
            max_action = self.action_space[max_action_index]
        else:
            max_action = None
            max_value = float('-inf')
            max_traj = None

            if self.do_action_clip:
                state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
                action_space_clipped = self.action_clip(state_tensor, self.action_space, self.planning_width)
            else:
                action_space_clipped = self.action_space
            state_tensor = state.to_tensor(add_batch_size=True, device=self.device)
            state_tensor = self.transform_state(state_tensor)
            actions = []
            if self.kinematics == "holonomic":
                actions.append(ActionXY(0, 0))
            else:
                actions.append(ActionRot(0, 0))
            # actions.append(ActionXY(0, 0))
            pre_next_state = self.state_predictor(state_tensor, actions)
            next_robot_states = None
            next_human_states = None
            next_value = []
            rewards = []
            for action in action_space_clipped:
                next_robot_state = self.compute_next_robot_state(state_tensor[0], action)
                next_human_state = pre_next_state[1]
                if next_robot_states is None and next_human_states is None:
                    next_robot_states = next_robot_state
                    next_human_states = next_human_state
                else:
                    next_robot_states = torch.cat((next_robot_states, next_robot_state), dim=0)
                    next_human_states = torch.cat((next_human_states, next_human_state), dim=0)
                next_state = tensor_to_joint_state_2types((next_robot_state, next_human_state))
                reward_est, _ = self.reward_estimator.estimate_reward_on_predictor(state, next_state)
                # max_next_return, max_next_traj = self.V_planning((next_robot_state, next_human_state), self.planning_depth, self.planning_width)
                # value = reward_est + self.get_normalized_gamma() * max_next_return
                # if value > max_value:
                #     max_value = value
                #     max_action = action
                #     max_traj = [(state_tensor, action, reward_est)] + max_next_traj
                # reward_est = self.estimate_reward(state, action)
                rewards.append(reward_est)
                # next_state = self.state_predictor(state_tensor, action)
            rewards_tensor = torch.tensor(rewards).to(self.device)
            next_state_batch = (next_robot_states, next_human_states)
            next_value = self.value_estimator(next_state_batch).squeeze(1)
            value = rewards_tensor + next_value * self.get_normalized_gamma()
            max_action_index = value.argmax()
            best_value = value[max_action_index]
            if best_value > max_value:
                max_action = action_space_clipped[max_action_index]
                next_state = tensor_to_joint_state_2types((next_robot_states[max_action_index], next_human_states[max_action_index]))
                max_next_traj = [(next_state.to_tensor(), None, None)]
                max_traj = [(state_tensor, max_action, rewards[max_action_index])] + max_next_traj
            if max_action is None:
                raise ValueError('Value network is not well trained.')

        if self.phase == 'train':
            self.last_state = self.transform(state)
        else:
            self.traj = max_traj
        for action_index in range(len(self.action_space)):
            action = self.action_space[action_index]
            if action is max_action:
                max_action_index = action_index
                break
        return max_action, int(max_action_index)

    def action_clip(self, state, action_space, width, depth=1):
        values = []
        actions = []
        if self.kinematics == "holonomic":
            actions.append(ActionXY(0, 0))
        else:
            actions.append(ActionRot(0, 0))
        # actions.append(ActionXY(0, 0))
        next_robot_states = None
        next_human_states = None
        pre_next_state = self.state_predictor(state, actions)
        for action in action_space:
            # actions = []
            # actions.append(action)
            next_robot_state = self.compute_next_robot_state(state[0], action)
            next_human_state = pre_next_state[1]
            if next_robot_states is None and next_human_states is None:
                next_robot_states = next_robot_state
                next_human_states = next_human_state
            else:
                next_robot_states = torch.cat((next_robot_states, next_robot_state), dim=0)
                next_human_states = torch.cat((next_human_states, next_human_state), dim=0)
            next_state_tensor = (next_robot_state, next_human_state)
            next_state = tensor_to_joint_state_2types(next_state_tensor)
            reward_est, _ = self.reward_estimator.estimate_reward_on_predictor(state, next_state)
            values.append(reward_est)
        next_return = self.value_estimator((next_robot_states, next_human_states)).squeeze()
        next_return = np.array(next_return.data.detach())
        values = np.array(values) + self.get_normalized_gamma() * next_return
        values = values.tolist()

        if self.sparse_search:
            # self.sparse_speed_samples = 2
            # search in a sparse grained action space
            added_groups = set()
            max_indices = np.argsort(np.array(values))[::-1]
            clipped_action_space = []
            for index in max_indices:
                if self.action_group_index[index] not in added_groups:
                    clipped_action_space.append(action_space[index])
                    added_groups.add(self.action_group_index[index])
                    if len(clipped_action_space) == width:
                        break
        else:
            max_indexes = np.argpartition(np.array(values), -width)[-width:]
            clipped_action_space = [action_space[i] for i in max_indexes]

        # print(clipped_action_space)
        return clipped_action_space

    def V_planning(self, state, depth, width):
        """ Plans n steps into future. Computes the value for the current state as well as the trajectories
        defined as a list of (state, action, reward) triples

        """

        current_state_value = self.value_estimator(state)
        if depth == 1:
            return current_state_value, [(state, None, None)]

        if self.do_action_clip:
            action_space_clipped = self.action_clip(state, self.action_space, width)
        else:
            action_space_clipped = self.action_space

        returns = []
        trajs = []
        actions =[]
        if self.kinematics == "holonomic":
            actions.append(ActionXY(0, 0))
        else:
            actions.append(ActionRot(0, 0))
        # actions.append(ActionXY(0, 0))
        pre_next_state = self.state_predictor(state, actions)
        for action in action_space_clipped:
            next_robot_staete = self.compute_next_robot_state(state[0], action)
            next_state_est = next_robot_staete, pre_next_state[1]
            # reward_est = self.estimate_reward(state, action)
            reward_est, _ = self.reward_estimator.estimate_reward_on_predictor(tensor_to_joint_state_2types(state),
                                                                               tensor_to_joint_state_2types(next_state_est))
            next_value, next_traj = self.V_planning(next_state_est, depth - 1, self.planning_width)
            return_value = current_state_value / depth + (depth - 1) / depth * (self.get_normalized_gamma() * next_value + reward_est)
            returns.append(return_value)
            trajs.append([(state, action, reward_est)] + next_traj)
        max_index = np.argmax(returns)
        max_return = returns[max_index]
        max_traj = trajs[max_index]
        return max_return, max_traj

    def compute_next_robot_state(self, robot_state, action):
        if robot_state.shape[0] != 1:
            raise NotImplementedError
        next_state = robot_state.clone().squeeze()
        if self.kinematics == 'holonomic':
            next_state[0] = next_state[0] + action.vx * self.time_step
            next_state[1] = next_state[1] + action.vy * self.time_step
            next_state[2] = action.vx
            next_state[3] = action.vy
        else:
            next_state[8] = (next_state[8] + action.r) % (2 * np.pi)
            next_state[0] = next_state[0] + np.cos(next_state[8]) * action.v * self.time_step
            next_state[1] = next_state[1] + np.sin(next_state[8]) * action.v * self.time_step
            next_state[2] = np.cos(next_state[8]) * action.v
            next_state[3] = np.sin(next_state[8]) * action.v
        return next_state.unsqueeze(0).unsqueeze(0)

    def get_attention_weights(self):
        return self.value_estimator.graph_model.attention_weights


    def transform_state(self, state):
        """
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (batch_size, number, state_length)(for example 100*1*9)
        human state tensor is of size (batch_size, number, state_length)(for example 100*5*5)
        """
        # for robot
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        #  0     1      2     3      4        5     6      7         8
        # for human
        #  'px', 'py', 'vx', 'vy', 'radius'
        #  0     1      2     3      4
        # for obstacle
        # 'px', 'py', 'radius'
        #  0     1     2
        # for wall
        # 'sx', 'sy', 'ex', 'ey'
        #  0     1     2     3
        assert len(state[0].shape) == 3
        robot_state = state[0]
        human_state = state[1]

        obstacle_state = state[2]
        if len(obstacle_state.shape) == 3:
            obs_pos = obstacle_state[:, :, 0:2]
            obs_vel = torch.zeros_like(obs_pos)
            obs_radius = obstacle_state[:, :, 2]
            obs_radius = obs_radius.unsqueeze(2)
            obs_human = torch.cat((obs_pos, obs_vel, obs_radius), dim=2)
            human_state = torch.cat((human_state, obs_human), dim=1)
        return (robot_state, human_state)