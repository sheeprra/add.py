import math

import torch
import torch.nn.functional as F
from torch import nn

from layers import GCNIILayer


class GCNII(nn.Module):
    def __init__(self, features, classes, k, dropout, hidden, alpha, lamda):
        super(GCNII, self).__init__()
        self.k = k
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.hidden = hidden
        self.linear_1 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(features, hidden),
        )
        self.layers = [GCNIILayer(hidden, hidden) for _ in range(k)]
        for index, layer in enumerate(self.layers):
            self.add_module('layer_{}'.format(index), layer)
        self.linear_2 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, h, adj):
        H0 = self.linear_1(h)
        Hk = H0
        I = torch.eye(self.hidden, self.hidden).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for index, layer in enumerate(self.layers):
            beta = math.log(self.lamda / (index + 1) + 1)
            Hk = layer(Hk, H0, adj, beta, self.alpha, I)
        Hk = self.linear_2(Hk)
        return F.log_softmax(Hk, dim=1)
