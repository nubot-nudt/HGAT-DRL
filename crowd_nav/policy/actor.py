import torch.nn as nn
import torch
import torch.nn.functional as F
from crowd_nav.policy.helpers import mlp
import numpy as np

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