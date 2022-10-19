import torch
import torch.nn.functional as F

from GNNHF import GNNHF
from util import load_dataset

name = 'Cora'
lr = 0.005
weight_decay = 5e-4
dropout = 0.6
alpha = 0.1
beta = 0.9
hidden = 64

dataset = load_dataset(name)
model = GNNHF(dataset.num_node_features, dataset.num_classes, dropout, hidden, alpha, beta)
data = dataset[0]

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
print(f"Test acc:{acc}")
