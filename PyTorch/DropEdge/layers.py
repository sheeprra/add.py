import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(GraphConvolution, self).__init__()
        # 输入特征
        self.input_feature = input_feature
        # 输出特征
        self.output_feature = output_feature
        # 权重
        self.weight = Parameter(torch.FloatTensor(input_feature, output_feature))
        # 初始化权重
        nn.init.xavier_normal_(self.weight, gain=0.8)

    def forward(self, h, adj):
        output = torch.mm(h, self.weight)
        output = torch.mm(adj, output)
        return F.relu(output)
