import torch
import torch.nn.functional as F

from JKNet import JKNet
from util import load_dataset

name = 'Cora'
lr = 0.005
dropout = 0.5
weight_decay = 5e-4
n_layers = 6
mode = 'max'
hidden = 16

dataset = load_dataset(name)

model = JKNet(dataset.num_node_features, dataset.num_classes, mode, n_layers, hidden, dropout)
data = dataset[0]
data.train_mask[:] = False
data.train_mask[:1624] = True
data.test_mask[:] = False
data.test_mask[2176:2708] = True

if torch.cuda.is_available():
    model = model.cuda()
    data = data.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()  # 评估
_, pred = model(data).max(dim=1)  # 最终输出的最大值，即属于哪一类
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()  # 计算测试集上的效果
acc = int(correct) / int(data.test_mask.sum())  # 准确率
print(f"Test acc:{acc:.4f}")
