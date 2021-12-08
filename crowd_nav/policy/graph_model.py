import logging
import itertools
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torch.nn import Parameter
from crowd_nav.policy.helpers import mlp, GAT #, GraphAttentionLayer
# from torch_geometric.nn import GATConv


class RGL(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection

        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection

        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)

        if self.similarity_function == 'embedded_gaussian':
            self.w_a = Parameter(torch.randn(self.X_dim, self.X_dim))
            nn.init.orthogonal_(self.w_a.data)
        elif self.similarity_function == 'concatenation':
            self.w_a = mlp(2 * X_dim, [2 * X_dim, 1], last_relu=True)

        self.w_v = mlp(X_dim, [X_dim], last_relu=True)

        # TODO: try other dim size
        embedding_dim = self.X_dim
        self.Ws = torch.nn.ParameterList()
        for i in range(self.num_layer):
            if i == 0:
                self.Ws.append(Parameter(torch.randn(self.X_dim, embedding_dim)))
            elif i == self.num_layer - 1:
                self.Ws.append(Parameter(torch.randn(embedding_dim, final_state_dim)))
            else:
                self.Ws.append(Parameter(torch.randn(embedding_dim, embedding_dim)))

        # for visualization
        self.attention_weights = None

    def compute_similarity_matrix(self, X):
        if self.similarity_function == 'embedded_gaussian':
            A = torch.matmul(torch.matmul(X, self.w_a), X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'gaussian':
            A = torch.matmul(X, X.permute(0, 2, 1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'cosine':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = torch.div(A, norm_matrix)
        elif self.similarity_function == 'cosine_softmax':
            A = torch.matmul(X, X.permute(0, 2, 1))
            magnitudes = torch.norm(A, dim=2, keepdim=True)
            norm_matrix = torch.matmul(magnitudes, magnitudes.permute(0, 2, 1))
            normalized_A = softmax(torch.div(A, norm_matrix), dim=2)
        elif self.similarity_function == 'concatenation':
            indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
            selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
            pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
            A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
            normalized_A = softmax(A, dim=2)
        elif self.similarity_function == 'squared':
            A = torch.matmul(X, X.permute(0, 2, 1))
            squared_A = A * A
            normalized_A = squared_A / torch.sum(squared_A, dim=2, keepdim=True)
        elif self.similarity_function == 'equal_attention':
            normalized_A = (torch.ones(X.size(1), X.size(1)) / X.size(1)).expand(X.size(0), X.size(1), X.size(1))
        elif self.similarity_function == 'diagonal':
            normalized_A = (torch.eye(X.size(1), X.size(1))).expand(X.size(0), X.size(1), X.size(1))
        else:
            raise NotImplementedError

        return normalized_A

    def forward(self, state):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        robot_state, human_states = state

        # compute feature matrix X
        robot_state_embedings = self.w_r(robot_state)
        human_state_embedings = self.w_h(human_states)
        X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)

        # compute matrix A
        if not self.layerwise_graph:
            normalized_A = self.compute_similarity_matrix(X)
            value_X = self.w_v(X)
            self.attention_weights = normalized_A[0, 0, :].data.cpu().numpy()

        next_H = H = value_X
        for i in range(self.num_layer):
            if self.layerwise_graph:
                A = self.compute_similarity_matrix(H)
                next_H = relu(torch.matmul(torch.matmul(A, H), self.Ws[i]))
            else:
                next_H = relu(torch.matmul(torch.matmul(normalized_A, H), self.Ws[i]))

            if self.skip_connection:
                next_H += H
            H = next_H

        return next_H

class GAT_RL(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        wr_dims = config.gcn.wr_dims
        wh_dims = config.gcn.wh_dims
        final_state_dim = config.gcn.final_state_dim
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection

        # design choice

        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection
        self.gat0 = GraphAttentionLayer(self.X_dim, self.X_dim)
        self.gat1 = GraphAttentionLayer(self.X_dim, self.X_dim)

        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        self.w_r = mlp(robot_state_dim, wr_dims, last_relu=True)
        self.w_h = mlp(human_state_dim, wh_dims, last_relu=True)
        # for visualization
        self.attention_weights = None

    def compute_adjectory_matrix(self, state):
        robot_state = state[0]
        human_state = state[1]
        robot_num = robot_state.size()[1]
        human_num = human_state.size()[1]
        Num = robot_num + human_num
        adj = torch.ones((Num, Num))
        for i in range(robot_num, robot_num+human_num):
            adj[i][0] = 0
        adj = adj.repeat(robot_state.size()[0], 1, 1)
        return adj

    def forward(self, state):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        robot_state, human_states = state
        if human_states is None:
            robot_state_embedings = self.w_r(robot_state)
            adj = torch.ones((1, 1))
            adj = adj.repeat(robot_state.size()[0], 1, 1)
            X = robot_state_embedings
            if robot_state.shape[0]==1:
                H1, self.attention_weights = self.gat0(X, adj)
            else:
                H1, _ = self.gat0(X, adj)
            H2, _ = self.gat1(H1, adj)
            if self.skip_connection:
                output = H1 + H2 + X
            else:
                output = H2
            return output
        else:
            adj = self.compute_adjectory_matrix(state)
            # compute feature matrix X
            robot_state_embedings = self.w_r(robot_state)
            human_state_embedings = self.w_h(human_states)
            X = torch.cat([robot_state_embedings, human_state_embedings], dim=1)
            if robot_state.shape[0]==1:
                H1, self.attention_weights = self.gat0(X, adj)
            else:
                H1, _ = self.gat0(X, adj)
            H2, _ = self.gat1(H1, adj)
            if self.skip_connection:
                output = H1 + H2 + X
            else:
                output = H2
            return output

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.w_a = mlp(2 * self.in_features, [2 * self.in_features, 1], last_relu=False)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.04)

    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        A = self.compute_similarity_matrix(input)
        e = self.leakyrelu(A)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        next_H = torch.matmul(attention, input)
        return next_H, attention[0, 0, :].data.cpu().numpy()

    def compute_similarity_matrix(self, X):
        indices = [pair for pair in itertools.product(list(range(X.size(1))), repeat=2)]
        selected_features = torch.index_select(X, dim=1, index=torch.LongTensor(indices).reshape(-1))
        pairwise_features = selected_features.reshape((-1, X.size(1) * X.size(1), X.size(2) * 2))
        A = self.w_a(pairwise_features).reshape(-1, X.size(1), X.size(1))
        return A

# class PGAT(torch.nn.Module):
#     def __init__(self, num_features, n_classes, num_heads, dropout, num_hidden, num_hidden_layers, activation, concat=True, neg_slope=0.2, bias=True):
#         super(PGAT, self).__init__()
#         # Activation
#         self.activation = activation
#         # input layer
#         self.gat_input = GATConv(num_features, num_hidden, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)
#         # Hidden layers
#         self.layers = nn.ModuleList()
#         for _ in range(num_hidden_layers):
#             if concat:
#                 self.layers.append(GATConv(num_hidden*num_heads, num_hidden, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias))
#             else:
#                 self.layers.append(GATConv(num_hidden, num_hidden, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias))
#         # output layer
#         if concat:
#             self.gat_output = GATConv(num_hidden*num_heads, n_classes, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)
#         else:
#             self.gat_output = GATConv(num_hidden, n_classes, heads=num_heads, concat= concat, negative_slope=neg_slope, dropout = dropout, bias=bias)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.gat_input(x, edge_index)
#         x = self.activation(x)
#         for layer in  self.layers:
#             x = layer(x, edge_index)
#             x = self.activation(x)
#         x = self.gat_output(x, edge_index)
#         x = x.mean(1).unsqueeze(dim=1)
#         return torch.tanh(x)

class PG_GAT_RL(nn.Module):
    def __init__(self, config, robot_state_dim, human_state_dim):
        """ The current code might not be compatible with models trained with previous version
        """
        super().__init__()
        self.multiagent_training = config.gcn.multiagent_training
        num_layer = config.gcn.num_layer
        X_dim = config.gcn.X_dim
        final_state_dim = config.gcn.final_state_dim
        similarity_function = config.gcn.similarity_function
        layerwise_graph = config.gcn.layerwise_graph
        skip_connection = config.gcn.skip_connection

        num_head = 4

        # design choice
        # 'gaussian', 'embedded_gaussian', 'cosine', 'cosine_softmax', 'concatenation'
        self.similarity_function = similarity_function
        self.robot_state_dim = robot_state_dim
        self.human_state_dim = human_state_dim
        self.state_dim = self.robot_state_dim + self.human_state_dim
        self.num_layer = num_layer
        self.X_dim = X_dim
        self.hidden_dim = 32
        self.layerwise_graph = layerwise_graph
        self.skip_connection = skip_connection
        self.encoder = mlp(self.state_dim, [64, self.X_dim], last_relu=True)
        self.gatinput = GATMultihead(self.X_dim, self.hidden_dim, self.X_dim, 4)
        # self.gat0 = GATMultihead(self.X_dim, self.hidden_dim, self.X_dim, 1)
        # self.gat1 = GATMultihead(self.X_dim, self.hidden_dim, self.X_dim, 4)
        # self.gat2 = GraphAttentionLayer2(self.X_dim, self.X_dim)
        logging.info('Similarity_func: {}'.format(self.similarity_function))
        logging.info('Layerwise_graph: {}'.format(self.layerwise_graph))
        logging.info('Skip_connection: {}'.format(self.skip_connection))
        logging.info('Number of layers: {}'.format(self.num_layer))

        # for visualize
        self.attention_weights = None

    def compute_adjectory_matrix(self, state):
        robot_num = 1
        human_num = state.shape[1] - robot_num
        Num = robot_num + human_num
        adj = torch.ones((Num, Num))
        for i in range(robot_num, robot_num+human_num):
            adj[i][0] = 0
        adj = adj.repeat(state.shape[0], 1, 1)
        return adj

    def forward(self, state):
        """
        Embed current state tensor pair (robot_state, human_states) into a latent space
        Each tensor is of shape (batch_size, # of agent, features)
        :param state:
        :return:
        """
        if state.shape[1] == 1:
            adj = torch.ones((1, 1))
            adj = adj.repeat(state.shape[0], 1, 1)
            X = state
            if state.shape[0]==1:
                H1, self.attention_weights = self.gat0(X, adj)
            else:
                H1, _ = self.gat0(X, adj)
            H2, _ = self.gat1(H1, adj)
            if self.skip_connection:
                output = H1 + H2 + X
            else:
                output = H2
            return output
        else:
            adj = self.compute_adjectory_matrix(state)
            H0 = self.encoder(state)
            # compute feature matrix X
            if state.shape[0]==1:
                H1 = self.gatinput(H0, adj)
            else:
                H1 = self.gatinput(H0, adj)
            # H2 = self.gat0(H1, adj)
            # H3 = self.gat1(H2, adj)
            # H4, _ = self.gat2(H3, adj)
            if self.skip_connection:
                output = H0 + H1
            else:
                output = H1
            # output = torch.cat((state, output), dim=2)
            return output

class GATMultihead(nn.Module):
    def __init__(self, nfeat, nhid, noutput, nheads):
        """Dense version of GAT."""
        super(GATMultihead, self).__init__()

        self.attentions = [GraphAttentionLayer2(nfeat, nhid, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer2(nhid * nheads, noutput, concat=False)

    def forward(self, x, adj):
        # x = nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = relu(self.out_att(x, adj))
        return x


class GraphAttentionLayer2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer2, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.orthogonal_(self.W.data, gain=1.414)
        self.a = Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.orthogonal_(self.a.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.orthogonal_(self.bias.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input, adj):

        # shape of input is batch_size, graph_size,feature_dims
        # shape of adj is batch_size, graph_size, graph_size
        assert len(input.shape) == 3
        assert len(adj.shape) == 3
        # map input to h
        h = torch.matmul(input, self.W)
        N = h.size()[1]
        batch_size = h.size()[0]
        a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N * N, -1), h.repeat(1, N, 1)],
                            dim=-1).view(batch_size, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, h)
        h_prime = h_prime + self.bias
        return nn.functional.elu(h_prime)


# class GraphAttentionLayer3(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer3, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)
#
#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = nn.functional.softmax(attention, dim=1)
#         h_prime = torch.matmul(attention, Wh)
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
