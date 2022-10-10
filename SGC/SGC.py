import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class SGC(nn.Module):
    def __init__(self, input_feature, output_feature, k):
        super(SGC, self).__init__()
        # 输入特征
        self.input_feature = input_feature
        # 输出特征
        self.output_feature = output_feature
        # 权重
        self.weight=Parameter(torch.FloatTensor(input_feature, output_feature))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=1)
        # k times
        self.k = k

    def forward(self, h, adj):
        output = adj
        for i in range(self.k):
            output = torch.mm(output, adj)
        output = torch.mm(output, h)
        output = torch.mm(output, self.weight)
        return F.log_softmax(output, dim=1)
