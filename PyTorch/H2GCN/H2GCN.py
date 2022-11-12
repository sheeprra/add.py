import torch
import torch.nn.functional as F
from torch import nn


class H2GCN(nn.Module):
    def __init__(self, features, hidden, classes, dropout, K):
        super(H2GCN, self).__init__()
        self.embed = nn.Sequential(nn.Linear(features, hidden),
                                   nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 ** (K + 1) - 1), classes)
        self.K = K

    def forward(self, x, adj):
        hidden_reps = []
        adj_2 = torch.pow(adj, 2)
        x = self.embed(x)
        hidden_reps.append(x)
        for _ in range(self.K):
            r1 = adj.matmul(x)
            r2 = adj_2.matmul(x)
            x = torch.cat([r1, r2], dim=-1)
            hidden_reps.append(x)
        hf = self.dropout(torch.cat(hidden_reps, dim=-1))
        return F.log_softmax(self.fc(hf), dim=1)
