import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from JKNet import JKNet
from util import load_data, accuracy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 参数配置
lr = 0.005
weight_decay = 5e-4
hidden = 16
idx_train = range(1200)
idx_val = range(1200, 1600)
idx_test = range(1600, 2000)
dropout = 0.5
n_layers = 6
# The way to aggregate outputs of layers  可选值 'max', 'cat'
mode = 'max'
# 数据集为cora或citeseer
dataset = 'cora'
adj_matrix, features, labels = load_data(dataset)

model = JKNet(features=features.shape[1], hidden=hidden, classes=labels.max().item() + 1, dropout=dropout,
              n_layers=n_layers, mode=mode)
print(model)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

if torch.cuda.is_available():
    model = model.cuda()
    features = features.cuda()
    adj_matrix = adj_matrix.cuda()
    labels = labels.cuda()


def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_matrix)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
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
    output = model(features, adj_matrix)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


for epoch in range(100):
    train(epoch)

test()
