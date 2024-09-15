import argparse
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch_geometric.utils import to_networkx

import configs
import gengraph
import models
import utils.featgen as featgen
import utils.graph_utils as graph_utils
import utils.io_utils as io_utils
import utils.math_utils as math_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
from dataset import GraphDataset, create_dataset

# print(G[0])
# print((labels))

pyg_dataset = pyg_dataset = GraphDataset(
    torch.load("my_data/cycle_line_star_complete_1.pt", weights_only=False)
)


def remove_duplicate_edges(edge_index):
    unique_edges = set()

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if (u, v) not in unique_edges and (v, u) not in unique_edges:
            unique_edges.add((u, v))

    unique_edges = list(unique_edges)
    unique_edges = torch.tensor(unique_edges, dtype=torch.long).t()

    return unique_edges


pyg_dataset[0].edge_index = remove_duplicate_edges(pyg_dataset[0].edge_index)

print(pyg_dataset[0])


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


plot_graph(pyg_dataset[0])
