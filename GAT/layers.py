import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class GATLayer(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(GATLayer, self).__init__()
        # 输入特征
        self.input_feature = input_feature
        # 输出特征
        self.output_feature = output_feature
        # 权重
        self.w = Parameter(torch.FloatTensor(input_feature, output_feature))
        self.a = Parameter(torch.FloatTensor(2 * output_feature, 1))
        # 初始化权重
        nn.init.xavier_uniform_(self.w, gain=1)
        nn.init.xavier_uniform_(self.a, gain=0.5)

    def forward(self, h, adj):
        # Wh.shape (N, out_feature)
        Wh = torch.mm(h, self.w)
        e = self.get_e(Wh)
        matrix = -9e15 * torch.ones_like(e)
        alpha = torch.where(adj > 0, e, matrix)
        alpha = F.softmax(alpha, dim=1)
        output = F.relu(torch.mm(alpha, Wh))
        return output

    def get_e(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])
        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])
        # e.shape (N, N)
        e = Wh1 + Wh2.T  # 构建自身的邻接矩阵
        return F.leaky_relu(e)
