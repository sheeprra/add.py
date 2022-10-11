import torch
import torch.nn.functional as F
from torch import nn


class APPNP(nn.Module):
    def __init__(self, features, classes, alpha, k, dropout, hidden):
        super(APPNP, self).__init__()
        self.alpha = alpha
        self.k = k
        self.dropout = dropout
        self.layer_1 = nn.Linear(features, hidden)
        self.layer_2 = nn.Linear(hidden, classes)

    def forward(self, h, adj):
        Z0 = self.layer_1(h)
        Z0 = self.layer_2(Z0)
        if self.dropout is not None:
            Zk = F.dropout(Z0, training=self.training)
        else:
            Zk = Z0
        for _ in range(self.k):
            Zk = (1 - self.alpha) * torch.mm(adj, Zk) + self.alpha * Z0
        return F.log_softmax(Zk, dim=1)
