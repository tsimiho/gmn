import random

import torch
from torch.optim import Adam
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

from models import GraphMatchingNetwork

dataset = TUDataset(
    root="data", name="PROTEINS", use_node_attr=True, transform=NormalizeFeatures()
)


class Args:
    def __init__(self):
        self.dim = 64
        self.feat_dim = dataset.num_features
        self.num_layers = 7
        self.margin = 0.1
        self.lr = 0.001
        self.n_classes = dataset.num_features
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = Args()


def create_graph_pairs(dataset, num_pairs=100000):
    pairs = []
    labels = []
    for _ in range(num_pairs):
        idx1, idx2 = random.sample(range(len(dataset)), 2)
        graph1, graph2 = dataset[idx1], dataset[idx2]

        label = 1 if graph1.y == graph2.y else -1
        pairs.append((graph1, graph2))
        labels.append(label)
    return pairs, labels


def collate_graph_pairs(batch):
    graph1_list, graph2_list, labels = [], [], []
    for (graph1, graph2), label in batch:
        graph1_list.append(graph1)
        graph2_list.append(graph2)
        labels.append(label)

    batched_graph1 = Batch.from_data_list(graph1_list)
    batched_graph2 = Batch.from_data_list(graph2_list)
    labels = torch.tensor(labels)
    return batched_graph1, batched_graph2, labels


pairs, labels = create_graph_pairs(dataset)
pair_dataset = [(pair, label) for pair, label in zip(pairs, labels)]
train_loader = DataLoader(
    pair_dataset, batch_size=1, shuffle=True, collate_fn=collate_graph_pairs
)

model = GraphMatchingNetwork(args).to(args.device)
optimizer = Adam(model.parameters(), lr=args.lr)


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    epochs = 0

    for graph_pair, label in train_loader:
        optimizer.zero_grad()

        graph1, graph2, label = (
            graph_pair[0].to(device),
            graph_pair[1].to(device),
            label.to(device),
        )

        feats_1, adjs_1 = graph1.x, graph1.edge_index
        feats_2, adjs_2 = graph2.x, graph2.edge_index
        sizes_1 = torch.bincount(graph1.batch)
        sizes_2 = torch.bincount(graph2.batch)

        emb_1, emb_2 = model(feats_1, adjs_1, feats_2, adjs_2, sizes_1, sizes_2)

        metrics = model.compute_metrics(emb_1, emb_2, label)
        loss = metrics["loss"]
        acc = metrics["acc"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc

        epochs += 1

        if epochs % 1000 == 0:
            print(f"Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")


train(model, train_loader, optimizer, args.device)
