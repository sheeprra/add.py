import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SGConv


class SGC(nn.Module):
    def __init__(self, features, k, classes, dropout):
        super(SGC, self).__init__()
        self.conv = SGConv(features, classes, k)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        output = self.conv(x, edge_index)
        return F.log_softmax(output, dim=1)
