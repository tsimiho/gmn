import numpy as np
import torch
from sklearn.decomposition import PCA

from configure import *
from evaluation import auc, compute_similarity
from gmn_utils import *
from loss import pairwise_loss, triplet_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_graph(graph):
    from_idx = graph.edge_index[0]
    to_idx = graph.edge_index[1]

    num_edges = graph.edge_index.shape[1]
    graph_idx = np.zeros(graph.x.shape[0], dtype=int)

    node_features = graph.x
    edge_features = torch.ones((num_edges, 4), dtype=torch.float32).requires_grad_(True)

    return node_features, edge_features, from_idx, to_idx, graph_idx


def reduce_dimensions(features, new_dim=64):
    pca = PCA(n_components=new_dim)
    reduced_features = pca.fit_transform(features)
    return torch.tensor(reduced_features, dtype=torch.float)


def similarity(gmn, config, graph1, graph2):
    (
        g1_node_features,
        g1_edge_features,
        g1_from_idx,
        g1_to_idx,
        g1_graph_idx,
    ) = convert_graph(graph1)
    (
        g2_node_features,
        g2_edge_features,
        g2_from_idx,
        g2_to_idx,
        g2_graph_idx,
    ) = convert_graph(graph2)

    stacked_node_features = torch.cat([g1_node_features, g2_node_features], dim=0)
    stacked_edge_features = torch.cat([g1_edge_features, g2_edge_features], dim=0)

    offset = max(g1_from_idx.max(), g1_to_idx.max()) + 1
    adjusted_g2_from_idx = g2_from_idx + offset
    adjusted_g2_to_idx = g2_to_idx + offset

    stacked_from_idx = torch.cat([g1_from_idx, adjusted_g2_from_idx], dim=0)
    stacked_to_idx = torch.cat([g1_to_idx, adjusted_g2_to_idx], dim=0)

    graph_idx_first = torch.zeros(g1_node_features.size(0), dtype=int)
    graph_idx_second = torch.ones(g2_node_features.size(0), dtype=int)
    stacked_graph_idx = torch.cat([graph_idx_first, graph_idx_second], dim=0)

    graph_vectors = gmn(
        stacked_node_features.to(device),
        stacked_edge_features.to(device),
        stacked_from_idx.to(device),
        stacked_to_idx.to(device),
        stacked_graph_idx.to(device),
        2,
    )

    x, y = reshape_and_split_tensor(graph_vectors, 2)
    similarity = compute_similarity(config, x, y)

    return similarity[0]
