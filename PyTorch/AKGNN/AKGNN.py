import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from layers import AKLayer


class AKGNN(nn.Module):
    def __init__(self, features, hidden, classes, dropout, n_layer):
        super(AKGNN, self).__init__()
        # layers
        self.layers = torch.nn.ModuleList([AKLayer() for _ in range(n_layer)])
        # W
        self.W = Parameter(torch.FloatTensor(features, hidden))
        nn.init.xavier_uniform_(self.W, gain=1)
        self.mlp = nn.Linear(hidden * n_layer, classes)
        self.dropout = dropout

    def forward(self, h, adj):
        h = F.leaky_relu(torch.mm(h, self.W))
        h = F.dropout(h, self.dropout, training=self.training)
        res = []
        I = torch.eye(adj.shape[0], adj.shape[0]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for layer in self.layers:
            h = layer(h, adj, I)
            h = F.dropout(h, self.dropout, training=self.training)
            res.append(h)
        output = torch.cat(res, dim=1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.mlp(output)
        return F.log_softmax(output, dim=1)
