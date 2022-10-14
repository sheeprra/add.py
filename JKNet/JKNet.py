import torch.nn.functional as F
from torch import nn

from layers import GraphConvolution, JumpingKnowledge


class JKNet(nn.Module):
    def __init__(self, features, classes, n_layers, dropout, hidden, mode):
        super(JKNet, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.layer_0 = GraphConvolution(features, hidden)
        for index in range(1, n_layers):
            setattr(self, 'layer_{}'.format(index), GraphConvolution(hidden, hidden))
        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.linear = nn.Linear(hidden, classes)
        elif mode == 'cat':
            self.linear = nn.Linear(hidden * n_layers, classes)

    def forward(self, h, adj):
        layer_output = []
        for index in range(self.n_layers):
            layer = getattr(self, f'layer_{index}')
            h = self.dropout(F.relu(layer(h, adj)))
            layer_output.append(h)
        h = self.jk(layer_output)
        h = self.linear(h)
        return F.log_softmax(h, dim=1)
