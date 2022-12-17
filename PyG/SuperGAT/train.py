import torch
import torch.nn.functional as F
from Net import Net
from util import load_dataset

name = 'Cora'
lr = 0.01
dropout = 0.6
weight_decay = 5e-4
heads = 8
attention_type = 'MX'

dataset = load_dataset(name)

model = Net(dataset.num_node_features, heads, dataset.num_classes, dropout, attention_type)
data = dataset[0]

if torch.cuda.is_available():
    model = model.cuda()
    data = data.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(data):
    model.train()
    optimizer.zero_grad()
    out, att_loss = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss += 4.0 * att_loss
    loss.backward()
    optimizer.step()


def test(data):
    model.eval()
    logits, accs = model(data)[0], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 501):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
