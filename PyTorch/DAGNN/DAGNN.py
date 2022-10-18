import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class DAGNN(nn.Module):
    def __init__(self, features, classes, k, dropout, hidden):
        super(DAGNN, self).__init__()
        self.k = k
        self.dropout = dropout
        self.MLP = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, classes),
        )

        self.s = Parameter(torch.FloatTensor(classes, 1))
        nn.init.xavier_uniform_(self.s, gain=1.414)

    def forward(self, h, adj):
        H = self.MLP(h)
        matrix_list = [H]
        for _ in range(self.k):
            H = torch.mm(adj, H)
            matrix_list.append(H)
        H = torch.stack(matrix_list, dim=1)
        S = F.sigmoid(torch.matmul(H, self.s))
        S = S.transpose(1, 2)
        X_out = torch.squeeze(torch.matmul(S, H))
        return F.log_softmax(X_out, dim=1)
