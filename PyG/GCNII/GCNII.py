import torch.nn.functional as F
from torch import nn

from layers import GCNIIConv


class DAGNN(nn.Module):
    def __init__(self, features, classes, k, dropout, hidden, alpha, lamda):
        super(DAGNN, self).__init__()
        self.linear_1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, hidden),
            nn.ReLU()
        )
        self.conv = GCNIIConv(hidden, hidden, k, alpha, lamda)
        self.linear_2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.linear_1(x)
        h = self.conv(h, edge_index)
        output = self.linear_2(h)
        return F.log_softmax(output, dim=1)
