import torch
import torch.nn.functional as F
from torch import nn


class AKLayer(nn.Module):
    def __init__(self):
        super(AKLayer, self).__init__()
        # lambda
        self.lambda_ = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)

    def forward(self, h, adj, I):
        lambda_max = 1 + F.relu(self.lambda_)
        alpha = ((2 * lambda_max - 2) / lambda_max) * I
        beta = (2 / lambda_max) * adj
        Ak = alpha + beta
        A = self.normalize(Ak)
        return torch.mm(A, h)

    def normalize(self, matrix):
        """对称归一化"""
        row_sum = torch.asarray(matrix.sum(1))
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)  # 构建成对角矩阵
        return torch.matmul(torch.matmul(d_mat_inv_sqrt, matrix), d_mat_inv_sqrt)
