import torch.nn as nn
import torch
import torch.nn.functional as F
from crowd_nav.policy.helpers import mlp
import numpy as np

class Critic(nn.Module):

    def __init__(self, config, graph_model1, graph_model2, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.graph_model1 = graph_model2
        self.score_network1 = mlp(config.gcn.X_dim + action_dim, [256, 256, 1])
        # Q2 architecture
        self.graph_model2 = graph_model2
        self.score_network2 = mlp(config.gcn.X_dim + action_dim, [256, 256, 1])
        self.action_dim = action_dim

    def set_action(self, action_dim):
        self.action_dim = action_dim


    def forward(self, state, action):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """

        assert len(state.size()) == 3
        # only use the feature of robot node as state representation
        state_embedding1 = self.graph_model1(state)[:, 0, :]
        sa1 = torch.cat([state_embedding1, action], 1)
        q1 = self.score_network1(sa1)

        state_embedding2 = self.graph_model2(state)[:, 0, :]
        sa2 = torch.cat([state_embedding2, action], 1)
        q2 = self.score_network2(sa2)
        return q1, q2

    def Q1(self, state, action):
        # only use the feature of robot node as state representation
        state_embedding1 = self.graph_model1(state)[:, 0, :]
        sa1 = torch.cat([state_embedding1, action], 1)
        q1 = self.score_network1(sa1)
        return q1
