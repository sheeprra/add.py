import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian


class EvenNetLayer(MessagePassing):
    def __init__(self, K, alpha):
        super(EvenNetLayer, self).__init__()
        self.K = int(K // 2)
        self.alpha = alpha
        TEMP = alpha * (1 - alpha) ** (2 * np.arange(K // 2 + 1))
        self.temp = Parameter(torch.tensor(TEMP))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** (2 * k)

    def forward(self, x, edge_index, edge_weight=None):
        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        # I-L
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=1., num_nodes=x.size(self.node_dim))

        output = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2)
            x = self.propagate(edge_index2, x=x, norm=norm2)
            weight = self.temp[k + 1]
            output = output + weight * x
        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
