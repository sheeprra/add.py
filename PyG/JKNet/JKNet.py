import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, JumpingKnowledge


class JKNet(nn.Module):
    def __init__(self, features, classes, mode, n_layers, hidden, dropout):
        super(JKNet, self).__init__()
        self.n_layers = n_layers
        self.mode = mode
        self.conv_0 = GCNConv(features, hidden)
        self.dropout = nn.Dropout(dropout)

        for i in range(1, self.n_layers):
            setattr(self, 'conv_{}'.format(i), GCNConv(hidden, hidden))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden, classes)
        elif mode == 'cat':
            self.fc = nn.Linear(n_layers * hidden, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        layer_out = []
        for i in range(self.n_layers):
            conv = getattr(self, 'conv_{}'.format(i))
            x = self.dropout(F.relu(conv(x, edge_index)))
            layer_out.append(x)

        h = self.jk(layer_out)
        h = self.fc(h)
        return F.log_softmax(h, dim=1)
