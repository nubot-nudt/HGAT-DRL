import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from crowd_nav.policy.helpers import mlp

class RGCN(nn.Module):
    def __init__(self, g, gnn_layers, in_dim, out_dim, hidden_dimensions, num_rels, activation,  final_activation,
                 feat_drop, num_bases=-1):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.encoder_dim = [64,32]
        self.hidden_dimensions = [64]
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activation = torch.nn.ReLU()
        self.final_activation = torch.nn.ReLU()
        self.gnn_layers = gnn_layers
        # create RGCN layers
        self.build_model()

    def set_g(self, g):
        self.g = g

    def build_model(self):
        self.layers = nn.ModuleList()
        self.encoder = mlp(self.in_dim, self.encoder_dim, last_relu=True)
        # input to hidden
        # i2h = self.build_input_layer()
        # self.layers.append(i2h)
        # hidden to hidden
        # for i in range(len(self.hidden_dimensions) - 1):
        #     h2h = self.build_hidden_layer(i)
        #     self.layers.append(h2h)
        # hidden to output
        # h2o = self.build_output_layer()
        # self.layers.append(h2o)
        i2o = self.build_i2o_layer()
        self.layers.append(i2o)

    def build_input_layer(self):
        print('Building an INPUT  layer of {}x{}'.format(self.in_dim, self.hidden_dimensions[0]))
        return RelGraphConv(self.in_dim, self.hidden_dimensions[0], self.num_rels,
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)


    def build_hidden_layer(self, i):
        print('Building an HIDDEN  layer of {}x{}'.format(self.hidden_dimensions[i], self.hidden_dimensions[i+1]))
        return RelGraphConv(self.hidden_dimensions[i], self.hidden_dimensions[i+1],  self.num_rels,
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=F.leaky_relu)

    def build_output_layer(self):
        print('Building an OUTPUT  layer of {}x{}'.format(self.hidden_dimensions[-1], self.out_dim))
        return RelGraphConv(self.hidden_dimensions[-1], self.out_dim, self.num_rels,
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.final_activation)

    def build_i2o_layer(self):
        print('Building an I2O  layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
        return RelGraphConv(self.encoder_dim[-1], self.out_dim, self.num_rels,
                            dropout=self.feat_drop, num_bases=self.num_bases, activation=self.final_activation)

    def forward(self, state_graph, node_features, edgetypes):
        h = node_features
        h = self.encoder(h)
        norm = state_graph.edata['norm']
        for layer in self.layers:
            h = layer(state_graph, h, edgetypes)
        return h



