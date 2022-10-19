import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import APPNP


class Net(nn.Module):
    def __init__(self, k, alpha, dropout, features, hidden, classes):
        super(Net, self).__init__()
        self.lin1 = Linear(features, hidden)
        self.lin2 = Linear(hidden, classes)
        self.dropout = dropout
        self.conv = APPNP(K=k, alpha=alpha)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        output = self.conv(x, edge_index)
        return F.log_softmax(output, dim=1)
