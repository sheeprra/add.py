import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class DAGNNConv(MessagePassing):
    def __init__(self, classes, k, add_self_loops=True, bias=True):
        super(DAGNNConv, self).__init__()
        self.K = k
        self.add_self_loops = add_self_loops
        self.bias = bias
        self.s = Parameter(torch.FloatTensor(classes, 1))
        nn.init.xavier_uniform_(self.s, gain=1.414)

    def forward(self, x, edge_index, edge_weight=None):
        matrix_list = [x]
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), False, self.add_self_loops,
                                    self.flow,
                                    dtype=x.dtype)
        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, norm=norm)
            matrix_list.append(x)
        H = torch.stack(matrix_list, dim=1)
        S = torch.sigmoid(torch.matmul(H, self.s))
        S = S.transpose(1, 2)
        return torch.squeeze(torch.matmul(S, H))

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
