import logging
import torch
import numpy as np

from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY, ActionDiff

from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor_batch
from crowd_nav.policy.graph_model import RGL, GAT_RL, PG_GAT_RL
from crowd_nav.policy.value_estimator import DQNNetwork, Noisy_DQNNetwork
from crowd_nav.policy.actor import Actor
from crowd_nav.policy.critic import Critic

class TD3RL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'TD3RL'
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
        self.robot_state_dim = 5
        self.human_state_dim = 5
        self.v_pref = 1
        self.share_graph_model = None
        self.value_estimator = None
        self.actor = None
        self.critic = None
        self.linear_state_predictor = None
        self.state_predictor = None
        self.planning_depth = None
        self.planning_width = None
        self.do_action_clip = None
        self.sparse_search = None
        self.action_group_index = []
        self.traj = None
        self.use_noisy_net = False
        self.count = 0
        self.action_dim = 2
        # max_action must be a tensor
        self.max_action = None
        self.min_action = None
        self.expl_noise = 0.2

    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.rotation_constraint = config.action_space.rotation_constraint

    def configure(self, config, device):
        self.set_common_parameters(config)
        self.robot_state_dim = config.gat.robot_state_dim
        self.human_state_dim = config.gat.human_state_dim
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor
        # self.set_device(device)
        self.device = device
        graph_model1 = PG_GAT_RL(config, self.robot_state_dim, self.human_state_dim)
        self.actor = Actor(config, graph_model1, self.action_dim, self.max_action, self.min_action)
        graph_model2 = PG_GAT_RL(config, self.robot_state_dim, self.human_state_dim)
        graph_model3 = PG_GAT_RL(config, self.robot_state_dim, self.human_state_dim)
        self.critic = Critic(config, graph_model2, graph_model3, self.action_dim)
        graph_model4 = PG_GAT_RL(config, self.robot_state_dim, self.human_state_dim)
        self.state_predictor = StatePredictor(config, graph_model4, self.time_step)
        self.model = [graph_model1, graph_model2, graph_model3, graph_model4, self.actor.action_network,
                      self.critic.score_network1, self.critic.score_network2,
                      self.state_predictor.human_motion_predictor]
        logging.info('TD3 action_dim is : {}'.format(self.action_dim))

    def set_action(self, action_dims, max_action, min_action):
        self.action_dim = action_dims
        self.max_action = max_action
        self.min_action = min_action
        self.actor.set_action(action_dims, max_action, min_action)
        self.critic.set_action(action_dims)

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
        return self.actor

    def get_state_dict(self):
        return {
                'graph_model1': self.actor.graph_model.state_dict(),
                'graph_model2': self.critic.graph_model1.state_dict(),
                'graph_model3': self.critic.graph_model2.state_dict(),
                'graph_model4': self.state_predictor.graph_model.state_dict(),
                'action_network': self.actor.action_network.state_dict(),
                'score_network1': self.critic.score_network1.state_dict(),
                'score_network2': self.critic.score_network2.state_dict(),
                'motion_predictor': self.state_predictor.human_motion_predictor.state_dict()
            }

    def get_traj(self):
        return self.traj

    def load_state_dict(self, state_dict):
        self.actor.graph_model.load_state_dict(state_dict['graph_model1'])
        self.critic.graph_model1.load_state_dict(state_dict['graph_model2'])
        self.critic.graph_model2.load_state_dict(state_dict['graph_model3'])
        self.state_predictor.graph_model.load_state_dict(state_dict['graph_model4'])
        self.actor.action_network.load_state_dict(state_dict['action_network'])
        self.critic.score_network1.load_state_dict(state_dict['score_network1'])
        self.critic.score_network2.load_state_dict(state_dict['score_network2'])
        self.state_predictor.human_motion_predictor.load_state_dict(state_dict['motion_predictor'])


    def save_model(self, file):
        torch.save(self.get_state_dict(), file)

    def load_model(self, file):
        checkpoint = torch.load(file)
        self.load_state_dict(checkpoint)

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)
        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        if self.phase == 'train':
            self.last_state = self.transform(state)
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        state_tensor = state.to_tensor(add_batch_size=False, device=self.device)
        state_tensor = self.rotate_state(state_tensor)
        state_tensor = state_tensor.unsqueeze(dim=0)
        if self.phase == 'train':
            action = (
                    self.actor(state_tensor).squeeze().detach().numpy()
                    + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
            Action = None
            if self.kinematics =='holonomic':
                speed = action[0]
                theta = action[1]
                Action = ActionXY(speed*np.cos(theta), speed * np.sin(theta))
            elif self.kinematics =='unicycle':
                speed = action[0]
                theta = action[1]
                Action = ActionRot(speed, theta)
            elif self.kinematics == 'differential':
                Action = ActionDiff(action[0],action[1])
            else:
                print('wrong kinematics')
            return Action, torch.tensor(action).float()
        else:
            with torch.no_grad():
                action = self.actor(state_tensor).squeeze().numpy()
                Action = None
                if self.kinematics == 'holonomic':
                    speed = action[0]
                    theta = action[1]
                    Action = ActionXY(speed * np.cos(theta), speed * np.sin(theta))
                elif self.kinematics == 'unicycle':
                    speed = action[0]
                    theta = action[1]
                    Action = ActionRot(speed, theta)
                elif self.kinematics == 'differential':
                    Action = ActionDiff(action[0], action[1])
                else:
                    print('wrong kinematics')
                return Action, torch.tensor(action).float()

    # def get_attention_weights(self):
    #     return self.actor.graph_model.attention_weights

    def rotate_state(self, state):
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
        assert len(state[0].shape) == 2
        if state[1] is None:
            robot_state = state[0]
            robot_feature_dim = 5
            human_feature_dim = 5
            dx = robot_state[:, 5] - robot_state[:, 0]
            dy = robot_state[:, 6] - robot_state[:, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            radius_r = robot_state[:, 4].unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
            rot = torch.atan2(dy, dx)
            cos_rot = torch.cos(rot)
            sin_rot = torch.sin(rot)
            vx = (robot_state[:, 2].unsqueeze(1) * cos_rot +
                  robot_state[:, 3].unsqueeze(1) * sin_rot).reshape((1, -1))
            vy = (robot_state[:, 3].unsqueeze(1) * cos_rot -
                  robot_state[:, 2].unsqueeze(1) * sin_rot).reshape((1, -1))
            v_pref = robot_state[:, 7].unsqueeze(1)
            theta = robot_state[:, 8].unsqueeze(1)
            px_r = torch.zeros_like(v_pref)
            py_r = torch.zeros_like(v_pref)
            new_robot_state = torch.cat((px_r, py_r, vx, vy, radius_r, dg, rot, v_pref, theta), dim=1)
            new_state = (new_robot_state, None)
            return new_state
        else:
            robot_state = state[0]
            human_state = state[1]
            obstacle_state = state[2]
            wall_state = state[3]
            human_num = human_state.shape[0]
            robot_num = robot_state.shape[0]
            obstacle_num = obstacle_state.shape[0]
            wall_num = wall_state.shape[1]

            dx = robot_state[:, 5] - robot_state[:, 0]
            dy = robot_state[:, 6] - robot_state[:, 1]
            dx = dx.unsqueeze(1)
            dy = dy.unsqueeze(1)
            dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
            rot = torch.atan2(dy, dx)
            cos_rot = torch.cos(rot)
            sin_rot = torch.sin(rot)
            transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=0).reshape(2, 2)
            a = robot_state[:, 2:4]
            robot_velocities = torch.mm(robot_state[:, 2:4], transform_matrix)
            radius_r = robot_state[:, 4].unsqueeze(1)
            v_pref = robot_state[:, 7].unsqueeze(1)
            target_heading = torch.zeros_like(radius_r)
            pos_r = torch.zeros_like(robot_velocities)
            cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
            new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading),
                                        dim=1)

            human_positions = human_state[:, 0:2] - robot_state[:, 0:2]
            human_positions = torch.mm(human_positions, transform_matrix)
            human_velocities = human_state[:, 2:4]
            human_velocities = torch.mm(human_velocities, transform_matrix)
            human_radius = human_state[:, 4].unsqueeze(1) + 0.3
            new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=1)

            wall_start_positions = wall_state[:, 0:2] - robot_state[:, 0:2]
            wall_start_positions = torch.mm(wall_start_positions, transform_matrix)
            wall_end_positions = wall_state[:, 2:4] - robot_state[:, 0:2]
            wall_end_positions = torch.mm(wall_end_positions, transform_matrix)
            wall_radius = torch.zeros((wall_state.shape[0], 1)) + 0.3
            new_wall_states = torch.cat((wall_start_positions, wall_end_positions, wall_radius), dim=1)

            if len(obstacle_state.shape) == 2:
                obstacle_positions = obstacle_state[:, 0:2] - robot_state[:, 0:2]
                obstacle_positions = torch.mm(obstacle_positions, transform_matrix)
                obstacle_radius = obstacle_state[:, 2].unsqueeze(1) + 0.3
                new_obstacle_states = torch.cat((obstacle_positions, obstacle_radius), dim=1)
                robot_feature_dim = new_robot_state.shape[1]
                human_feature_dim = new_human_state.shape[1]
                obstacle_feature_dim = new_obstacle_states.shape[1]
                wall_feature_dim = new_wall_states.shape[1]
                robot_zero_feature = torch.zeros(
                    [robot_num, human_feature_dim + obstacle_feature_dim + wall_feature_dim])
                human_zero_feature1 = torch.zeros([human_num, robot_feature_dim])
                human_zero_feature2 = torch.zeros([human_num, obstacle_feature_dim + wall_feature_dim])
                obstacle_zero_feature1 = torch.zeros([obstacle_num, robot_feature_dim + human_feature_dim])
                obstacle_zero_feature2 = torch.zeros([obstacle_num, wall_feature_dim])
                wall_zero_feature = torch.zeros(
                    [wall_num, robot_feature_dim + human_feature_dim + obstacle_feature_dim])

                new_robot_state = torch.cat((new_robot_state, robot_zero_feature), dim=1)
                new_human_state = torch.cat((human_zero_feature1, new_human_state, human_zero_feature2), dim=1)
                new_obstacle_states = torch.cat((obstacle_zero_feature1, new_obstacle_states, obstacle_zero_feature2),
                                                dim=1)
                new_wall_states = torch.cat((wall_zero_feature, new_wall_states), dim=1)
                new_state = torch.cat((new_robot_state, new_human_state, new_obstacle_states, new_wall_states), dim=0)
            else:
                robot_feature_dim = new_robot_state.shape[1]
                human_feature_dim = new_human_state.shape[1]
                obstacle_feature_dim = 3
                wall_feature_dim = new_wall_states.shape[1]
                robot_zero_feature = torch.zeros(
                    [robot_num, human_feature_dim + obstacle_feature_dim + wall_feature_dim])
                human_zero_feature1 = torch.zeros([human_num, robot_feature_dim])
                human_zero_feature2 = torch.zeros([human_num, obstacle_feature_dim + wall_feature_dim])
                wall_zero_feature = torch.zeros(
                    [wall_num, robot_feature_dim + human_feature_dim + obstacle_feature_dim])
                new_robot_state = torch.cat((new_robot_state, robot_zero_feature), dim=1)
                new_human_state = torch.cat((human_zero_feature1, new_human_state, human_zero_feature2), dim=1)

                new_wall_states = torch.cat((wall_zero_feature, new_wall_states), dim=1)
                new_state = torch.cat((new_robot_state, new_human_state, new_wall_states), dim=0)
            return new_state