import torch
from torch.nn import Module

from util import norm_Adj


class GNNHFConv(Module):
    def __init__(self, alpha, beta):
        super(GNNHFConv, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, edge_index):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj = norm_Adj(x, edge_index, device)
        I = torch.eye(x.shape[0]).to(device)
        pre = (self.beta + 1 / self.alpha) * I + (1 - self.beta - 1 / self.alpha) * adj
        res = torch.mm((I + self.beta * adj), x)
        return torch.matmul(torch.linalg.inv(pre), res)


class GNNLFConv(Module):
    def __init__(self, alpha, mu):
        super(GNNLFConv, self).__init__()
        self.alpha = alpha
        self.mu = mu

    def forward(self, x, edge_index):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj = norm_Adj(x, edge_index, device)
        I = torch.eye(x.shape[0]).to(device)
        pre = (self.mu + 1 / self.alpha - 1) * I + (2 - self.mu - 1 / self.alpha) * adj
        res = torch.mm(self.mu * I + (1 - self.mu) * adj, x)
        return torch.matmul(torch.linalg.inv(pre), res)
