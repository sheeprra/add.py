import torch.nn.functional as F
from torch import nn

from layers import DAGNNConv


class DAGNN(nn.Module):
    def __init__(self, features, classes, k, dropout, hidden):
        super(DAGNN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )
        self.conv = DAGNNConv(classes, k)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.MLP(x)
        output = self.conv(h, edge_index)
        return F.log_softmax(output, dim=1)
