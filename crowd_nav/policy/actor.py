import torch.nn as nn
import torch
import torch.nn.functional as F
from crowd_nav.policy.helpers import mlp
import numpy as np
import copy

class Actor(nn.Module):
    def __init__(self, config, graph_model, action_dim, max_action, min_action):
        super(Actor, self).__init__()
        self.graph_model = graph_model
        self.action_network = mlp(config.gcn.X_dim, [256, action_dim])
        self.max_action = None
        self.min_action = None
        self.action_dim = action_dim
        self.action_amplitude = max_action
        self.action_middle = min_action

    def set_action(self, action_dim, max_action, min_action):
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.action_amplitude = (self.max_action - self.min_action) / 2.0
        self.action_amplitude = torch.from_numpy(self.action_amplitude)
        self.action_middle = torch.from_numpy(self.min_action + self.max_action) / 2.0

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        assert len(state.size()) == 3

        # only use the feature of robot node as state representation
        state_embedding = self.graph_model(state)[:, 0, :]
        a = self.action_network(state_embedding)
        action = self.action_middle + self.action_amplitude * torch.tanh(a)
        return action

        # def forward(self, state):
        #     """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        #     """
        #     assert len(state[0].shape) == 3
        #     assert len(state[1].shape) == 3
        #
        #     # only use the feature of robot node as state representation
        #     state_embedding = self.graph_model(state)[:, 0, :]
        #     a = self.action_network(state_embedding)
        #     action = self.action_middle + self.action_amplitude * torch.tanh(a)
        #     return action
        # return self.max_action * torch.tanh(a)

class Actor0(nn.Module):
    def __init__(self, config, graph_model, action_dim, max_action, min_action):
        super(Actor0, self).__init__()
        self.graph_model = graph_model
        self.action_network = mlp(config.gcn.X_dim, [256, action_dim])
        self.max_action = None
        self.min_action = None
        self.action_dim = action_dim
        self.action_amplitude = max_action
        self.action_middle = min_action

    def set_action(self, action_dim, max_action, min_action):
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.action_amplitude = (self.max_action - self.min_action) / 2.0
        self.action_amplitude = torch.from_numpy(self.action_amplitude)
        self.action_middle = torch.from_numpy(self.min_action + self.max_action) / 2.0

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        if state.batch_size == 1:
            cur_state = copy.deepcopy(state)
            robot_state = state.ndata['h'][0, 4:13]
            robot_state = robot_state.unsqueeze(dim=0)
            robot_state = robot_state.unsqueeze(dim=0)
            # state_embedding = self.graph_model(state)[0, :]
            # state_embedding = torch.cat((robot_state, state_embedding), dim=0)

            state_embedding = self.graph_model(robot_state)[:, 0, :]
            a = self.action_network(state_embedding)
            action = self.action_middle + self.action_amplitude * torch.tanh(a)
        # batch training phase
        else:
            cur_state = copy.deepcopy(state)
            cur_features = cur_state.ndata['h']
            actor_state = copy.deepcopy(state)
            # state_embedding = self.graph_model(actor_state)
            num_nodes = actor_state._batch_num_nodes['_N'].numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id = robot_id + num_nodes[i]
            robot_ids = torch.LongTensor(robot_ids)
            # robot_ids = torch.cat((torch.zeros(1), num_nodes[:-1]), dim=0).type(torch.int64)
            cur_robot_feature = torch.index_select(cur_features, 0, robot_ids)
            robot_state = cur_robot_feature[:, 4:9]
            robot_state = robot_state.unsqueeze(dim=1)
            # batch_state_embedding = torch.index_select(state_embedding, 0, robot_ids)
            # batch_state_embedding = torch.cat((cur_robot_feature[:, 4:13], batch_state_embedding), dim=1)
            batch_state_embedding = self.graph_model(robot_state)[:, 0, :]

            a = self.action_network(batch_state_embedding)
            action = self.action_middle + self.action_amplitude * torch.tanh(a)
        return action

class GraphActor(nn.Module):
    def __init__(self, config, graph_model, action_dim, max_action, min_action):
        super(GraphActor, self).__init__()
        self.graph_model = graph_model
        # self.action_network = mlp(config.gcn.X_dim + 9, [64, 128, action_dim])
        self.action_network = mlp(5, [256, action_dim])
        self.encode_r = mlp(5, [64, 32], last_relu=True)
        self.max_action = None
        self.min_action = None
        self.action_dim = action_dim
        self.action_amplitude = max_action
        self.action_middle = min_action

    def set_action(self, action_dim, max_action, min_action):
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.action_amplitude = (self.max_action - self.min_action) / 2.0
        self.action_amplitude = torch.from_numpy(self.action_amplitude)
        self.action_middle = torch.from_numpy(self.min_action + self.max_action) / 2.0

    def forward(self, state):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        # assert len(state.size()) == 3

        # only use the feature of robot node as state representation
        # exploration or test phase
        if state.batch_size == 1:
            cur_state = copy.deepcopy(state)
            robot_state = state.ndata['h'][0, 4:9]
            # state_embedding = self.graph_model(state)[0, :]
            # state_embedding = torch.cat((robot_state, state_embedding), dim=0)

            # state_embedding = self.encode_r(robot_state)
            state_embedding = robot_state
            a = self.action_network(state_embedding)
            action = self.action_middle + self.action_amplitude * torch.tanh(a)
        # batch training phase
        else:
            cur_state = copy.deepcopy(state)
            cur_features = cur_state.ndata['h']
            actor_state = copy.deepcopy(state)
            num_nodes = actor_state._batch_num_nodes['_N'].numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id = robot_id + num_nodes[i]
            robot_ids = torch.LongTensor(robot_ids)
            cur_robot_feature = torch.index_select(cur_features, 0, robot_ids)
            # state_embedding = self.graph_model(actor_state)
            # batch_state_embedding = torch.index_select(state_embedding, 0, robot_ids)
            # batch_state_embedding = torch.cat((cur_robot_feature[:, 4:13], batch_state_embedding), dim=1)\
            # batch_state_embedding = self.encode_r(cur_robot_feature[:, 4:9])
            batch_state_embedding = cur_robot_feature[:, 4:9]
            a = self.action_network(batch_state_embedding)
            action = self.action_middle + self.action_amplitude * torch.tanh(a)
        return action


