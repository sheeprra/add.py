import torch
import torch_geometric.utils as utils
from torch_geometric.datasets import Planetoid


def load_dataset(name):
    return Planetoid(root='../data', name=name)


def norm_Adj(x, edge_index, device):
    edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.shape[0])
    adj = utils.to_scipy_sparse_matrix(edge_index).todense()
    adj = torch.Tensor(adj).to(device)
    row, col = edge_index[0], edge_index[1]
    deg = utils.degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    return torch.mm(torch.mm(deg_inv_sqrt, adj), deg_inv_sqrt)
