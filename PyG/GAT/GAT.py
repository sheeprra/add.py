import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, features, heads, classes, dropout, hidden):
        super(GAT, self).__init__()
        self.conv = GATConv(features, hidden, heads, dropout=dropout)
        self.last_conv = GATConv(hidden * heads, classes, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        output = self.conv(x, edge_index)
        output = F.relu(output)
        output = self.last_conv(output, edge_index)
        return F.log_softmax(output, dim=1)
