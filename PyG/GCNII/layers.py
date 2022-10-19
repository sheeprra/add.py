import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GCNIIConv(MessagePassing):
    def __init__(self, features, classes, k, alpha, lamda, add_self_loops=True, bias=True):
        super(GCNIIConv, self).__init__()
        self.K = k
        self.features = features
        self.classes = classes
        self.alpha = alpha
        self.lamda = lamda
        self.I = torch.eye(features, classes).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.add_self_loops = add_self_loops
        self.bias = bias

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), False, self.add_self_loops,
                                    self.flow,
                                    dtype=x.dtype)
        for index in range(self.K):
            weight = Parameter(torch.FloatTensor(self.features, self.classes).to('cuda'))
            nn.init.xavier_uniform_(weight, gain=1.414)
            beta = math.log(self.lamda / (index + 1) + 1)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, norm=norm)
            res = (1 - self.alpha) * x + self.alpha * h
            mapping = (1 - beta) * self.I + beta * weight
            x = F.relu(torch.matmul(res, mapping))
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
