import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import Node2Vec
from torch_geometric.utils import to_networkx


def create_cycle_graph(n_nodes):
    edge_index = [[i, (i + 1) % n_nodes] for i in range(n_nodes)] + [
        [(i + 1) % n_nodes, i] for i in range(n_nodes)
    ]
    return Data(
        edge_index=torch.tensor(edge_index).t().contiguous(),
        y=torch.tensor([0]),
        num_nodes=n_nodes,
    )


def create_complete_graph(n_nodes):
    edge_index = [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j]
    return Data(
        edge_index=torch.tensor(edge_index).t().contiguous(),
        y=torch.tensor([1]),
        num_nodes=n_nodes,
    )


def create_line_graph(n_nodes):
    edge_index = [[i, i + 1] for i in range(n_nodes - 1)] + [
        [i + 1, i] for i in range(n_nodes - 1)
    ]
    return Data(
        edge_index=torch.tensor(edge_index).t().contiguous(),
        y=torch.tensor([2]),
        num_nodes=n_nodes,
    )


def create_star_graph(n_leaves):
    edge_index = [[0, i] for i in range(1, n_leaves + 1)] + [
        [i, 0] for i in range(1, n_leaves + 1)
    ]
    return Data(
        edge_index=torch.tensor(edge_index).t().contiguous(),
        y=torch.tensor([3]),
        num_nodes=n_leaves + 1,
    )


def create_wheel_graph(n_nodes):
    edges = []
    for i in range(1, n_nodes):
        edges.append([0, i])
        edges.append([i, i % (n_nodes - 1) + 1])
    edges += [[j, i] for i, j in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index, y=torch.tensor([4]), num_nodes=n_nodes)


def plot_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(5, 5))
    nx.draw(
        G,
        with_labels=False,
        node_color="skyblue",
        node_size=500,
        edge_color="k",
        linewidths=2,
        font_size=15,
        pos=nx.spring_layout(G),
    )
    plt.show()


def add_noise_to_graph(graph, num_new_nodes, avg_new_edges_per_node=2):
    assert avg_new_edges_per_node > 0, "Each new node should have at least one edge."

    num_nodes = graph.num_nodes
    num_edges = graph.edge_index.size(1)

    new_edges_list = []
    for new_node_id in range(num_nodes, num_nodes + num_new_nodes):
        connections = np.random.choice(num_nodes, avg_new_edges_per_node, replace=False)
        for conn in connections:
            new_edges_list.append([new_node_id, conn])
            new_edges_list.append([conn, new_node_id])

    new_edges = torch.tensor(new_edges_list, dtype=torch.long).t()
    new_edge_index = torch.cat([graph.edge_index, new_edges], dim=1)

    x = torch.zeros(graph.num_nodes + num_new_nodes, 4)

    new_graph = Data(
        x=x,
        edge_index=new_edge_index,
        num_nodes=graph.num_nodes + num_new_nodes,
        y=graph.y,
    )

    return new_graph


funcs = [
    create_complete_graph,
    create_cycle_graph,
    create_line_graph,
    create_star_graph,
    create_wheel_graph,
]

nodes = [2, 3, 4]
edges = [4, 5, 6]


class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.name = "Synthetic"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def create_dataset():
    graphs = []
    for func in funcs:
        graph = func(8)
        for _ in range(4):
            for n in nodes:
                for e in edges:
                    noisy_graph = add_noise_to_graph(graph, n, e)
                    graphs.append(noisy_graph)
    return GraphDataset(graphs)
