import torch
import torch_geometric.utils as utils
from torch_geometric.nn import MessagePassing


class GNNHFConv(MessagePassing):
    def __init__(self, alpha, beta):
        super(GNNHFConv, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, edge_index):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.shape[0])
        adj = utils.to_scipy_sparse_matrix(edge_index).todense()
        adj = torch.Tensor(adj).to(device)
        row, col = edge_index[0], edge_index[1]
        deg_inv_sqrt = norm(col, x.size(0), x.dtype)
        adj = torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
        I = torch.eye(x.shape[0]).to(device)
        pre = (self.beta + 1 / self.alpha) * I + (1 - self.beta - 1 / self.alpha) * adj
        res = torch.mm((I + self.beta * adj), x)
        return torch.matmul(torch.linalg.inv(pre), res)


class GNNLFConv(MessagePassing):
    def __init__(self, alpha, mu):
        super(GNNLFConv, self).__init__()
        self.alpha = alpha
        self.mu = mu

    def forward(self, x, edge_index):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.shape[0])
        adj = utils.to_scipy_sparse_matrix(edge_index).todense()
        adj = torch.Tensor(adj).to(device)
        row, col = edge_index[0], edge_index[1]
        deg_inv_sqrt = norm(col, x.size(0), x.dtype)
        adj = torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
        I = torch.eye(x.shape[0]).to(device)
        pre = (self.mu + 1 / self.alpha - 1) * I + (2 - self.mu - 1 / self.alpha) * adj
        res = torch.mm(self.mu * I + (1 - self.mu) * adj, x)
        return torch.matmul(torch.linalg.inv(pre), res)


def norm(col, size, type):
    deg = utils.degree(col, size, dtype=type)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    return deg_inv_sqrt
