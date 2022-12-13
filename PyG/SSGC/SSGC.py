import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SSGConv


class SSGC(nn.Module):
    def __init__(self, features, K, classes, alpha):
        super(SSGC, self).__init__()
        self.conv = SSGConv(features, classes, alpha, K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        output = self.conv(x, edge_index)
        return F.log_softmax(output, dim=1)
