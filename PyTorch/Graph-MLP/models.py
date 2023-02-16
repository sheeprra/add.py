import torch
import torch.nn.functional as F
from torch import nn


def get_feature_dis(x):
    """
    论文中的Sim函数
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


class GMLP(nn.Module):
    def __init__(self, input_feature, classes, hidden, dropout):
        super(GMLP, self).__init__()
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(input_feature, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )
        self.last_layer = nn.Linear(hidden, classes)

    def forward(self, x):
        x = self.mlp(x)
        if self.training:
            x_dis = get_feature_dis(x)
            return F.log_softmax(self.last_layer(x), dim=1), x_dis
        else:
            return F.log_softmax(self.last_layer(x), dim=1)
