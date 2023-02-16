import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import GMLP
from util import load_data, accuracy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 参数配置
lr = 0.01
weight_decay = 5e-4
hidden = 256
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
dropout = 0.6
# α is the weighting coefficient to balance the those two losses
alpha = 2.0
# temperature parameter
tau = 1.0
# 数据集为cora或citeseer
dataset = 'cora'

adj_matrix, features, labels = load_data(dataset)

model = GMLP(input_feature=features.shape[1], classes=labels.max().item() + 1, hidden=hidden, dropout=dropout)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if torch.cuda.is_available():
    model = model.cuda()
    features = features.cuda()
    adj_matrix = adj_matrix.cuda()
    labels = labels.cuda()


def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def train(epoch):
    model.train()
    optimizer.zero_grad()
    output, x_dis = model(features)
    loss_train_class = F.nll_loss(output[idx_train], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, adj_matrix, tau=tau)
    loss_train = loss_train_class + loss_Ncontrast * alpha
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()))


def test():
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


for epoch in range(400):
    train(epoch)

test()
