import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class DAGNN(nn.Module):
    def __init__(self, features, classes, k, dropout, hidden):
        super(DAGNN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )
        self.k = k
        self.dropout = dropout
        self.s = Parameter(torch.FloatTensor(classes, 1))
        nn.init.xavier_uniform_(self.s)

    def forward(self, h, adj):
        H = self.MLP(h)
        if self.dropout is not None:
            H = F.dropout(H, training=self.training)
        matrix_list = [H]
        for _ in range(self.k):
            H = torch.mm(adj, H)
            matrix_list.append(H)
        H = torch.stack(matrix_list, dim=1)
        S = F.relu(torch.matmul(H, self.s))
        S = S.transpose(1, 2)
        X_out = torch.squeeze(torch.matmul(S, H))
        return F.log_softmax(X_out, dim=1)
