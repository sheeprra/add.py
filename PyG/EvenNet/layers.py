import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian


class EvenNetLayer(MessagePassing):
    def __init__(self, K, alpha):
        super(EvenNetLayer, self).__init__()
        self.K = int(K // 2)
        self.alpha = alpha
        # 权重
        self.weight = Parameter(torch.FloatTensor(self.K + 1, 1))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=1.2)

    def forward(self, x, edge_index, edge_weight=None):
        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        # I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=1., num_nodes=x.size(self.node_dim))

        output = x * self.weight[0]
        for k in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2)
            x = self.propagate(edge_index2, x=x, norm=norm2)
            output = output + self.weight[k + 1] * x
        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
