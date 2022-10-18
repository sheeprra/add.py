import torch.nn.functional as F
from torch import nn

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, features, classes, n_layers, dropout, hidden):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_layer = GraphConvolution(features, hidden)
        self.layers = [GraphConvolution(hidden, hidden) for _ in range(n_layers)]
        for index, layer in enumerate(self.layers):
            self.add_module('layer_{}'.format(index), layer)
        self.out_layer = GraphConvolution(hidden, classes)

    def forward(self, h, adj):
        x = self.in_layer(h, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.layers:
            x = layer(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_layer(x, adj)
        return F.log_softmax(x, dim=1)
