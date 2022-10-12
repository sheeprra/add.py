import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class GCNIILayer(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(GCNIILayer, self).__init__()
        # 输入特征
        self.input_feature = input_feature
        # 输出特征
        self.output_feature = output_feature
        # 权重
        self.weight = Parameter(torch.FloatTensor(input_feature, output_feature))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=1.2)

    def forward(self, x, H0, adj, beta, alpha, I):
        res = (1 - alpha) * torch.mm(adj, x) + alpha * H0
        mapping = (1 - beta) * I + beta * self.weight
        return F.relu(torch.matmul(res, mapping))
