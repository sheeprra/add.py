import torch
import torch.nn.functional as F
from torch import nn


class GNNLF(nn.Module):
    def __init__(self, features, classes, dropout, hidden, alpha, mu):
        super(GNNLF, self).__init__()
        self.mu = mu
        self.dropout = dropout
        self.alpha = alpha
        self.hidden = hidden
        self.features = features
        self.linear_1 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(features, hidden),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, x, adj):
        H = self.linear_1(x)
        I = torch.eye(adj.shape[0]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        pre = (self.mu + 1 / self.alpha - 1) * I + (2 - self.mu - 1 / self.alpha) * adj
        res = torch.mm(self.mu * I + (1 - self.mu) * adj, H)
        Z = torch.matmul(torch.linalg.inv(pre), res)
        Z = self.linear_2(Z)
        return F.log_softmax(Z, dim=1)
