import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges

from GAE import GCNEncoder
from util import load_dataset

name = 'Cora'
out_channels = 2
lr = 0.01
weight_decay = 5e-4

dataset = load_dataset(name)
dataset.transform = T.NormalizeFeatures()

model = GAE(GCNEncoder(dataset.num_node_features, out_channels))
data = dataset[0]
data = train_test_split_edges(data)
train_pos_edge_index = data.train_pos_edge_index
if torch.cuda.is_available():
    model = model.cuda()
    data = data.cuda()
    train_pos_edge_index = train_pos_edge_index.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    z = model.encoder(data.x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()

model.eval()
z = model.encode(data.x, train_pos_edge_index)
auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
print(f"Test auc:{auc}  ap:{ap}")
