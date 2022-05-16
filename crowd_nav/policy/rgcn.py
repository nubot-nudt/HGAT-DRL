import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from dgl.nn.pytorch.conv.gatconv import GATConv
from dgl.nn.pytorch.conv.gcn2conv import GCN2Conv
from dgl.nn.pytorch.conv.gatv2conv import GATv2Conv
from dgl.nn.pytorch.conv.graphconv import GraphConv
from dgl.nn.pytorch.conv.gatconv import GATConv
from crowd_nav.policy.helpers import mlp
import dgl
class RGCN(nn.Module):
    def __init__(self, g, gnn_layers, in_dim, out_dim, hidden_dimensions, num_rels, activation,  final_activation,
                 feat_drop, num_bases=-1):
        super(RGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.encoder_dim = [64, out_dim]
        self.hidden_dimensions = [32]
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.feat_drop = feat_drop
        self.num_bases = num_bases
        self.activation = torch.nn.ReLU()
        self.final_activation = torch.nn.ReLU()
        self.gnn_layers = gnn_layers
        self.use_rgcn = False
        self.use_gat = False
        self.use_gcn = True
        self.use_rgat = True
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
        if self.use_rgcn is True:
            print('Building an I2O  layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return RelGraphConv(self.encoder_dim[-1], self.out_dim, self.num_rels,
                                dropout=self.feat_drop, num_bases=self.num_bases, activation=self.final_activation)
        elif self.use_gcn is True:
            print('Building an I2O  layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return GraphConv(self.encoder_dim[-1], self.out_dim, activation=self.final_activation)
        elif self.use_gat is True:
            print('Building an I2O  layer of {}x{}'.format(self.encoder_dim[-1], self.out_dim))
            return GATConv(self.encoder_dim[-1], self.out_dim, num_heads=1, activation=self.final_activation)



    def forward(self, state_graph, node_features, edgetypes):
        h = node_features
        h0 = self.encoder(h)
        norm = state_graph.edata['norm']
        output = h0
        for layer in self.layers:
            if self.use_rgcn:
                h1 = layer(state_graph, output, edgetypes)
                output = output + h1
            elif self.use_gat or self.use_gcn:
                state_graph = dgl.add_self_loop(state_graph)
                h1 = layer(state_graph, output)
                h1 = h1.reshape(-1, self.out_dim)
                output = output + h1
        ## skip connection???
        return output



