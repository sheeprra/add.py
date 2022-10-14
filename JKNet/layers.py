import torch
from torch import nn
from torch.nn.parameter import Parameter


class JumpingKnowledge(nn.Module):
    def __init__(self, mode):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode

    def forward(self, layer_output):
        if self.mode == 'cat':
            return torch.cat(layer_output, dim=1)
        elif self.mode == 'max':
            return torch.stack(layer_output, dim=0).max(dim=0)[0]


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
        nn.init.xavier_uniform_(self.weight, gain=1.)

    def forward(self, h, adj):
        output = torch.mm(h, self.weight)
        output = torch.mm(adj, output)
        return output
