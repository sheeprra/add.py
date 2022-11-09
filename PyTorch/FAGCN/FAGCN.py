import torch.nn.functional as F
from torch import nn

from layers import FALayer


class FAGCN(nn.Module):
    def __init__(self, features, hidden, classes, layer_num, dropout, eps):
        super(FAGCN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layers = [FALayer(hidden) for _ in range(layer_num)]
        for index, layer in enumerate(self.layers):
            self.add_module('layer_{}'.format(index), layer)
        self.last_layer = nn.Linear(hidden, classes)
        self.eps = eps

    def forward(self, h, adj, deg):
        h = self.MLP(h)
        h0 = h
        for layer in self.layers:
            h = layer(h, adj,deg)
            h = self.eps * h0 + h
        output = self.last_layer(h)
        return F.log_softmax(output, dim=1)
