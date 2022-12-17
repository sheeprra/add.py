import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SuperGATConv


class Net(nn.Module):
    def __init__(self, features, heads, classes, dropout, attention_type):
        super(Net, self).__init__()
        self.dropout = dropout
        self.conv1 = SuperGATConv(features, classes, heads=heads,
                                  dropout=dropout, attention_type=attention_type,
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.conv2 = SuperGATConv(classes * heads, classes, heads=heads,
                                  concat=False, dropout=dropout,
                                  attention_type=attention_type, edge_sample_ratio=0.8,
                                  is_undirected=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        return F.log_softmax(x, dim=-1), att_loss
