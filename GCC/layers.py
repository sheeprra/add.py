import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class GCCLayer(nn.Module):
    def __init__(self, input_feature, output_feature, yt, kt):
        super(GCCLayer, self).__init__()
        self.yt = yt
        self.kt = kt
        # 输入特征
        self.input_feature = input_feature
        # 输出特征
        self.output_feature = output_feature
        # 权重
        self.weight = Parameter(torch.FloatTensor(input_feature, output_feature))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=1.4)

    def forward(self, X, H1, H2, adj):
        AH = torch.mm(adj, X)
        HAW = torch.mm(AH, self.weight)
        H = HAW + self.yt * X + self.kt * (H2 - H1)
        return F.relu(H)
