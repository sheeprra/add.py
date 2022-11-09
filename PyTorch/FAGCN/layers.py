import torch
from torch import nn
from torch.nn.parameter import Parameter


class FALayer(nn.Module):
    def __init__(self, hidden):
        super(FALayer, self).__init__()
        self.hidden = hidden
        # 权重
        self.g = Parameter(torch.FloatTensor(2 * hidden, 1))
        # 初始化权重
        nn.init.xavier_uniform_(self.g, gain=1.414)

    def forward(self, h, adj, deg):
        ag = self.get_ag(h)
        matrix = torch.zeros_like(ag)
        ag = torch.where(adj > 0, ag, matrix)
        output = torch.mm(deg, ag)
        output = torch.mm(output, deg)
        return torch.matmul(output, h)

    def get_ag(self, h):
        h1 = torch.matmul(h, self.g[:self.hidden, :])
        h2 = torch.matmul(h, self.g[self.hidden:, :])
        # e.shape (N, N)
        ag = h1 + h2.T  # 构建自身的邻接矩阵
        return torch.tanh(ag)
