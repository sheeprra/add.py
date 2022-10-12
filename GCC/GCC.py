import torch.nn.functional as F
from torch import nn

from layers import GCCLayer


class GCC(nn.Module):
    def __init__(self, features, classes, k, dropout, yt, kt):
        super(GCC, self).__init__()
        self.k = k
        self.dropout = dropout
        self.layers = [GCCLayer(features, features, yt, kt) for _ in range(k)]
        for index, layer in enumerate(self.layers):
            self.add_module('layer_{}'.format(index), layer)
        self.linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(features, classes),
        )

    def forward(self, X, adj):
        H1 = X
        H = X
        for index, layer in enumerate(self.layers):
            if index == 0:
                H = layer(X, H1, H, adj)
            if index == 1:
                H1 = H
                H = layer(X, X, H, adj)
            else:
                temp = H
                H = layer(X, H1, H, adj)
                H1 = temp
        H = self.linear(H)
        return F.log_softmax(H, dim=1)
