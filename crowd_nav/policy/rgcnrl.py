import logging
import torch
import numpy as np

from numpy.linalg import norm
import itertools
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY, ActionDiff

from crowd_nav.policy.state_predictor import StatePredictor, LinearStatePredictor_batch
from crowd_nav.policy.graph_model import DGL_RGCN_RL, PG_GAT_RL0
from crowd_nav.policy.actor import GraphActor, Actor0
from crowd_nav.policy.critic import GraphCritic, Critic0
from crowd_nav.utils.crowdgraph import CrowdNavGraph
from crowd_nav.safelayer.cbf_layer import CascadeCBFLayer

class RGCNRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'RGCNRL'
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
        self.safelayer = CascadeCBFLayer()
    def set_common_parameters(self, config):
        self.gamma = config.rl.gamma
        self.kinematics = config.action_space.kinematics
        self.rotation_constraint = config.action_space.rotation_constraint

    def configure(self, config, device):
        self.set_common_parameters(config)
        self.planning_depth = config.model_predictive_rl.planning_depth
        self.planning_width = config.model_predictive_rl.planning_width
        self.share_graph_model = config.model_predictive_rl.share_graph_model
        self.linear_state_predictor = config.model_predictive_rl.linear_state_predictor
        self.device = device

        graph_model1 = DGL_RGCN_RL(config, self.robot_state_dim, self.human_state_dim)
        self.actor = GraphActor(config, graph_model1, self.action_dim, self.max_action, self.min_action)
        graph_model2 = DGL_RGCN_RL(config, self.robot_state_dim, self.human_state_dim)
        graph_model3 = DGL_RGCN_RL(config, self.robot_state_dim, self.human_state_dim)
        self.critic = GraphCritic(config, graph_model2, graph_model3, self.action_dim)

        graph_model4 = DGL_RGCN_RL(config, self.robot_state_dim, self.human_state_dim)
        self.state_predictor = StatePredictor(config, graph_model4, self.time_step)

        self.model = [graph_model1, graph_model2, graph_model3, graph_model4, self.actor.action_network,
                      self.critic.score_network1, self.critic.score_network2,
                      self.state_predictor.human_motion_predictor]
        logging.info('RGCN action_dim is : {}'.format(self.action_dim))

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
            print("Wrong Arrival!!")
            if self.kinematics == 'holonomic':
                return ActionXY(0.0, 0.0), torch.tensor((0, 0)).float()
            elif self.kinematics == 'unicycle':
                return ActionRot(0.0, 0), torch.tensor((0, 0)).float()
            else:
                return ActionDiff(0.0, 0.0), torch.tensor((0,0)).float()
        state_tensor = state.to_tensor(add_batch_size=False, device=self.device)
        state_graph = CrowdNavGraph(state_tensor).graph
        if self.phase == 'train':
            action = (
                    self.actor(state_graph).squeeze().detach().numpy()
                    + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
            ).clip(-self.max_action, self.max_action)
            Action = None
            # safe_action = self.safelayer.get_safe_action(state_tensor, action)
            if self.kinematics =='holonomic':
                speed = action[0]
                theta = action[1]
                Action = ActionXY(speed*np.cos(theta), speed * np.sin(theta))
            elif self.kinematics =='unicycle':
                speed = action[0]
                theta = action[1]
                Action = ActionRot(speed, theta)
            elif self.kinematics == 'differential':
                Action = ActionDiff(action[0], action[1])
            else:
                print('wrong kinematics')
            return Action, torch.tensor(action).float()
        else:
            with torch.no_grad():
                action = self.actor(state_graph).squeeze().numpy()
                Action = None
                # safe_action = self.safelayer.get_safe_action(state_tensor, action)
                if self.kinematics == 'holonomic':
                    speed = action[0]
                    theta = action[1]
                    Action = ActionXY(speed * np.cos(theta), speed * np.sin(theta))
                elif self.kinematics == 'unicycle':
                    speed = action[0]
                    theta = action[1]
                    Action = ActionRot(speed, theta)
                elif self.kinematics == 'differential':
                    # if (action[0] - safe_action[0]) ** 2 + (action[1] - safe_action[1]) ** 2 > 0.01:
                    #     action[0] = safe_action[0]
                    #     action[1] = safe_action[1]
                    Action = ActionDiff(action[0], action[1])
                    # print("velocity is %f, %f",{state_tensor[0][0][2],state_tensor[0][0][3]})
                    # print("action is %f, %f", {action[0], action[1]})
                else:
                    print('wrong kinematics')
                return Action, torch.tensor(action).float()