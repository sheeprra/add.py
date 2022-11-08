import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GRPLayer(MessagePassing):
    def __init__(self, K, alpha):
        super(GRPLayer, self).__init__()
        self.K = K
        self.alpha = alpha
        # 权重
        self.weight = Parameter(torch.FloatTensor(self.K + 1, 1))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=1.2)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        output = x * self.weight[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            output = output + self.weight[k + 1] * x
        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
