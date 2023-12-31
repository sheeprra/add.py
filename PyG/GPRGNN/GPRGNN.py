import torch.nn.functional as F
from torch import nn

from layers import GPRLayer


class GPRGNN(nn.Module):
    def __init__(self, features, classes, K, dropout, hidden, dprate, alpha):
        super(GPRGNN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )
        self.dprate = dprate
        self.dropout = dropout
        self.conv = GPRLayer(K, alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.MLP(x)
        if self.dprate == 0.0:
            x = self.conv(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.conv(x, edge_index)
            return F.log_softmax(x, dim=1)
