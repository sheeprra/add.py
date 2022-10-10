import torch.nn.functional as F
from torch import nn

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, features, hidden, classes, dropout):
        super(GCN, self).__init__()
        self.layer_1 = GraphConvolution(features, hidden)
        self.layer_2 = GraphConvolution(hidden, classes)
        self.dropout = dropout

    def forward(self, h, adj):
        output = self.layer_1(h, adj)
        output = F.relu(output)
        if self.dropout is not None:
            output = F.dropout(output, self.dropout, training=self.training)
        output = self.layer_2(output, adj)
        return F.log_softmax(output, dim=1)
