import torch.nn as nn
import torch
from crowd_nav.policy.helpers import mlp
import copy

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

class Critic0(nn.Module):

    def __init__(self, config, graph_model1, graph_model2, action_dim):
        super(Critic0, self).__init__()
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
        if state.batch_size == 1:
            cur_state1 = copy.deepcopy(state)
            robot_state1 = cur_state1.ndata['h'][0, 4:9]
            robot_state1 = robot_state1.unsqueeze(dim=0)
            robot_state1 = robot_state1.unsqueeze(dim=0)
            # state_embedding = self.graph_model(state)[0, :]
            # state_embedding = torch.cat((robot_state, state_embedding), dim=0)
            state_embedding1 = self.graph_model1(robot_state1)[:, 0, :]
            sa1 = torch.cat([state_embedding1, action], 1)
            q1 = self.score_network1(sa1)

            cur_state2 = copy.deepcopy(state)
            robot_state2 = cur_state2.ndata['h'][0, 4:9]
            robot_state2 = robot_state2.unsqueeze(dim=0)
            robot_state2 = robot_state2.unsqueeze(dim=0)
            state_embedding2 = self.graph_model2(robot_state2)[:, 0, :]
            sa2 = torch.cat([state_embedding2, action], 1)
            q2 = self.score_network2(sa2)
        # batch training phase
        else:
            num_nodes = state._batch_num_nodes['_N'].numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id = robot_id + num_nodes[i]
            robot_ids = torch.LongTensor(robot_ids)
            
            cur_state1 = copy.deepcopy(state)
            cur_features = cur_state1.ndata['h']
            # state_embedding = self.graph_model(actor_state)
            cur_robot_feature1 = torch.index_select(cur_features, 0, robot_ids)
            robot_state1 = cur_robot_feature1[:, 4:9]
            robot_state1 = robot_state1.unsqueeze(dim=1)
            # batch_state_embedding = torch.index_select(state_embedding, 0, robot_ids)
            # batch_state_embedding = torch.cat((cur_robot_feature[:, 4:13], batch_state_embedding), dim=1)
            state_embedding1 = self.graph_model1(robot_state1)[:, 0, :]
            sa1 = torch.cat([state_embedding1, action], 1)
            q1 = self.score_network1(sa1)
            
            
            cur_state2 = copy.deepcopy(state)
            cur_features = cur_state2.ndata['h']
            # state_embedding = self.graph_model(actor_state)
            cur_robot_feature1 = torch.index_select(cur_features, 0, robot_ids)
            robot_state2 = cur_robot_feature1[:, 4:9]
            robot_state2 = robot_state2.unsqueeze(dim=1)
            # batch_state_embedding = torch.index_select(state_embedding, 0, robot_ids)
            # batch_state_embedding = torch.cat((cur_robot_feature[:, 4:13], batch_state_embedding), dim=1)
            state_embedding2 = self.graph_model1(robot_state2)[:, 0, :]
            sa2 = torch.cat([state_embedding2, action], 1)
            q2 = self.score_network1(sa2)
        return q1, q2

    def Q1(self, state, action):
        if state.batch_size == 1:
            cur_state1 = copy.deepcopy(state)
            robot_state1 = cur_state1.ndata['h'][0, 4:9]
            robot_state1 = robot_state1.unsqueeze(dim=0)
            robot_state1 = robot_state1.unsqueeze(dim=0)
            # state_embedding = self.graph_model(state)[0, :]
            # state_embedding = torch.cat((robot_state, state_embedding), dim=0)
            state_embedding1 = self.graph_model1(robot_state1)[:, 0, :]
            sa1 = torch.cat([state_embedding1, action], 1)
            q1 = self.score_network1(sa1)
        # batch training phase
        else:
            num_nodes = state._batch_num_nodes['_N'].numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id = robot_id + num_nodes[i]
            robot_ids = torch.LongTensor(robot_ids)

            cur_state1 = copy.deepcopy(state)
            cur_features = cur_state1.ndata['h']
            # state_embedding = self.graph_model(actor_state)
            cur_robot_feature1 = torch.index_select(cur_features, 0, robot_ids)
            robot_state1 = cur_robot_feature1[:, 4:9]
            robot_state1 = robot_state1.unsqueeze(dim=1)
            # batch_state_embedding = torch.index_select(state_embedding, 0, robot_ids)
            # batch_state_embedding = torch.cat((cur_robot_feature[:, 4:13], batch_state_embedding), dim=1)
            state_embedding1 = self.graph_model1(robot_state1)[:, 0, :]
            sa1 = torch.cat([state_embedding1, action], 1)
            q1 = self.score_network1(sa1)
        return q1

class GraphCritic(nn.Module):
    def __init__(self, config, graph_model1, graph_model2, action_dim):
        super(GraphCritic, self).__init__()
        # Q1 architecture
        self.graph_model1 = graph_model1
        # self.encode_r1 = mlp(5, [64, 32], last_relu=True)
        self.score_network1 = mlp(config.gcn.X_dim + action_dim + 5, [256, 128, 1])
        # self.score_network1 = mlp(5 + action_dim, [256, 256, 1])
        # Q2 architecture
        self.graph_model2 = graph_model2
        # self.score_network2 = mlp(9 + config.gcn.X_dim + action_dim, [64, 128, 256, 1])
        # self.encode_r2 = mlp(5, [64, 32], last_relu=True)
        # self.score_network2 = mlp(5 + action_dim, [256, 256, 1])
        self.score_network2 = mlp(config.gcn.X_dim + action_dim + 5, [256, 128, 1])
        self.action_dim = action_dim

    def set_action(self, action_dim):
        self.action_dim = action_dim


    def forward(self, state, action):
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        # only use the feature of robot node as state representation
        if state.batch_size == 1:
            cur_state = copy.deepcopy(state)
            robot_state = cur_state.ndata['h'][0, 4:9]
            state_embedding1 = self.graph_model1(cur_state)[0, :]
            state_embedding1 = torch.cat((robot_state, state_embedding1), dim=0)
            # state_embedding1 = self.encode_r1(robot_state)
            # state_embedding1 = robot_state
            sa1 = torch.cat([state_embedding1, action], 0)
            q1 = self.score_network1(sa1)

            state_embedding2 = self.graph_model2(state)[0, :]
            state_embedding2 = torch.cat((robot_state, state_embedding2), dim=0)
            # state_embedding2 = robot_state
            # state_embedding2 = torch.encode_r2(robot_state)
            sa2 = torch.cat([state_embedding2, action], 0)
            q2 = self.score_network1(sa2)
        else:
            cur_state = copy.deepcopy(state)
            cur_features = cur_state.ndata['h']

            num_nodes = cur_state._batch_num_nodes['_N'].numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id = robot_id + num_nodes[i]
            robot_ids = torch.LongTensor(robot_ids)
            cur_robot_feature = torch.index_select(cur_features, 0, robot_ids)
            state1 = copy.deepcopy(state)
            state_embedding1 = self.graph_model1(state1)
            batch_state_embedding1 = torch.index_select(state_embedding1, 0, robot_ids)
            batch_robot_state1 = cur_robot_feature[:, 4:9]
            batch_state_embedding1 = torch.cat((batch_robot_state1, batch_state_embedding1), dim=1)
            # batch_state_embedding1 = self.encode_r1(cur_robot_feature[:,4:9])
            # batch_state_embedding1 = cur_robot_feature[:,4:9]
            sa1 = torch.cat([batch_state_embedding1, action], 1)
            q1 = self.score_network1(sa1)

            state2 = copy.deepcopy(state)
            state_embedding2 = self.graph_model2(state2)
            batch_state_embedding2 = torch.index_select(state_embedding2, 0, robot_ids)
            batch_robot_state2 = cur_robot_feature[:,4:9]
            batch_state_embedding2 = torch.cat((batch_robot_state2, batch_state_embedding2), dim=1)
            # batch_state_embedding2 = self.encode_r2(cur_robot_feature[:,4:9])
            # batch_state_embedding2 = cur_robot_feature[:,4:9]
            sa2 = torch.cat([batch_state_embedding2, action], 1)
            q2 = self.score_network2(sa2)
        return q1, q2

    def Q1(self, state, action):
        # only use the feature of robot node as state representation
        if state.batch_size == 1:
            cur_state = copy.deepcopy(state)
            robot_state = cur_state.ndata['h'][0, 4:9]
            # state_embedding1 = self.encode_r1(robot_state)
            state_embedding1 = self.graph_model1(cur_state)[0, :]
            state_embedding1 = torch.cat((robot_state, state_embedding1), dim=0)

            # state_embedding1 = robot_state
            sa1 = torch.cat([state_embedding1, action], 0)
            q1 = self.score_network1(sa1)
        # batch training phase
        else:
            cur_state1 = copy.deepcopy(state)
            cur_features = cur_state1.ndata['h']
            num_nodes = cur_state1._batch_num_nodes['_N'].numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id = robot_id + num_nodes[i]
            robot_ids = torch.LongTensor(robot_ids)
            cur_robot_feature = torch.index_select(cur_features, 0, robot_ids)

            state_embedding1 = self.graph_model1(cur_state1)
            batch_state_embedding1 = torch.index_select(state_embedding1, 0, robot_ids)
            batch_robot_state = cur_robot_feature[:,4:9]
            # batch_state_embedding1 = self.encode_r1(cur_robot_feature[:,4:9])
            batch_state_embedding1 = torch.cat((batch_robot_state, batch_state_embedding1), dim=1)


            # batch_state_embedding1 =cur_robot_feature[:,4:9]
            sa1 = torch.cat([batch_state_embedding1, action], 1)
            q1 = self.score_network1(sa1)
        return q1
