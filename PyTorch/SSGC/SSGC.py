import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class SSGC(nn.Module):
    def __init__(self, input_feature, output_feature, K, alpha, dropout):
        super(SSGC, self).__init__()
        # 输入特征
        self.input_feature = input_feature
        # 输出特征
        self.output_feature = output_feature
        # 权重
        self.weight = Parameter(torch.FloatTensor(input_feature, output_feature))
        # 初始化权重
        nn.init.xavier_uniform_(self.weight, gain=1)
        # k times
        self.K = K
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, h, adj):
        res = []
        for i in range(self.K):
            T = adj if i == 0 else torch.mm(T, adj)
            temp = (1 - self.alpha) * torch.mm(T, h) + self.alpha * h
            res.append(temp)
        output = torch.stack(res, dim=1)
        output = torch.mean(output, dim=1)
        output = torch.mm(output, self.weight)
        return F.log_softmax(output, dim=1)
