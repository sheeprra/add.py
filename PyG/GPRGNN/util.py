from torch_geometric.datasets import Planetoid


def load_dataset(name):
    return Planetoid(root='../data', name=name)
