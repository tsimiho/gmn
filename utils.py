import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx, to_undirected
from torch_scatter import scatter_mean


def acc_f1(output, labels, average="binary", logging=None, verbose=True):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds
        labels = labels
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


def dynamic_partition(data, partitions, num_partitions):
    res = []
    for i in range(num_partitions):
        res.append(data[torch.where(partitions == i)])
    return res


def adj_matrix_to_edge_index(adj_matrix, device=None):
    edge_index = [[], []]
    for i, row in enumerate(adj_matrix.detach().numpy().tolist()):
        for j, cell_value in enumerate(row[i + 1 :]):
            if cell_value == 1:
                edge_index[0].append(i)
                edge_index[1].append(j)
        edge_index[0].append(i)
        edge_index[1].append(i)
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    if device:
        edge_index = edge_index.to(device)
    return edge_index


def trim_feats(feats, sizes):
    stacked_num_nodes = sum(sizes)
    stacked_tree_feats = torch.zeros(
        (stacked_num_nodes, feats.shape[-1]), dtype=torch.float64
    )
    start_index = 0
    for i, size in enumerate(sizes):
        end_index = start_index + size
        stacked_tree_feats[start_index:end_index, :] = feats[i, :size, :]
        start_index = end_index
    return stacked_tree_feats


def create_batch(sizes):
    sizes = sizes.tolist()
    sizes = list(map(int, sizes))
    batch = []
    for i, size in enumerate(sizes):
        batch.extend([i] * size)
    batch = torch.tensor(batch, dtype=torch.int64)
    return batch


def pairwise_cosine_similarity(a, b):
    a_norm = a / torch.norm(a, dim=1).unsqueeze(-1)
    b_norm = b / torch.norm(b, dim=1).unsqueeze(-1)
    res = torch.matmul(a_norm, b_norm.transpose(-2, -1))
    return res


def compute_cross_attention(x_i, x_j):
    a = pairwise_cosine_similarity(x_i, x_j)
    a_i = F.softmax(a, dim=1)
    a_j = F.softmax(a, dim=0)
    att_i = torch.matmul(a_i, x_j)
    att_j = torch.matmul(a_j.T, x_i)
    return att_i, att_j, a_i, a_j


def batch_block_pair_attention(data, batch, n_graphs):
    all_attention_x = []
    all_attention_y = []
    all_a_i = []
    all_a_j = []

    for i in range(0, n_graphs, 2):
        x_i = data[batch == i]
        x_j = data[batch == i + 1]

        attention_x, attention_y, a_i, a_j = compute_cross_attention(x_i, x_j)

        all_attention_x.append(attention_x)
        all_attention_y.append(attention_y)
        all_a_i.append(a_i)
        all_a_j.append(a_j)

    result_x = torch.cat(all_attention_x, dim=0)
    result_y = torch.cat(all_attention_y, dim=0)
    # result_a_i = torch.cat(all_a_i, dim=0) if len(all_a_i) > 0 else torch.Tensor()
    # result_a_j = torch.cat(all_a_j, dim=0) if len(all_a_j) > 0 else torch.Tensor()

    result = torch.cat([result_x, result_y], dim=0)

    return result, all_a_i, all_a_j


def spectral_clustering(edge_index, num_nodes, num_clusters=10):
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    adjacency_matrix_np = adjacency_matrix.numpy()
    clustering = SpectralClustering(
        n_clusters=num_clusters, affinity="precomputed", assign_labels="discretize"
    ).fit(adjacency_matrix_np)
    return clustering.labels_


def cdm(node_embeddings, epsilon=1e-6, tau=0.5):
    exp_embeddings = torch.exp(node_embeddings)
    sum_exp_embeddings = torch.sum(exp_embeddings, dim=1, keepdim=True)
    q_i_tilde = exp_embeddings / sum_exp_embeddings
    max_q_i_tilde = torch.max(q_i_tilde, dim=1, keepdim=True).values
    q_i = q_i_tilde / (max_q_i_tilde + epsilon)
    r_i = (q_i >= tau).int()
    class_assignments = r_i.tolist()
    node_classes = [
        assignment.index(1) if 1 in assignment else -1
        for assignment in class_assignments
    ]

    return torch.tensor(node_classes)


def pyg_cluster(features, edge_index, node_clusters):
    unique_clusters = node_clusters.unique()
    supernode_features = []
    node_to_supernode = torch.empty(node_clusters.size(0), dtype=torch.long)
    for i, cluster in enumerate(unique_clusters):
        cluster_mask = node_clusters == cluster
        cluster_features = features[cluster_mask]
        supernode_feature = cluster_features.mean(dim=0)
        supernode_features.append(supernode_feature)
        node_to_supernode[cluster_mask] = i
    supernode_features = torch.stack(supernode_features, dim=0)

    supernode_edges = []
    for edge in edge_index.t():
        supernode_u = node_to_supernode[edge[0]]
        supernode_v = node_to_supernode[edge[1]]
        if supernode_u != supernode_v:
            supernode_edges.append([supernode_u, supernode_v])
    supernode_edges = torch.tensor(supernode_edges, dtype=torch.long).t()

    supernode_edges = to_undirected(supernode_edges, num_nodes=len(unique_clusters))

    clustered_data = Data(x=supernode_features, edge_index=supernode_edges)

    return clustered_data


def analyze_dataset(dataset):
    num_graphs = len(dataset)
    num_classes = dataset.num_classes if hasattr(dataset, "num_classes") else None
    num_node_features = (
        dataset.num_node_features if hasattr(dataset, "num_node_features") else None
    )
    labels = [data.y.item() for data in dataset]
    num_nodes = [data.num_nodes for data in dataset]
    class_distribution = Counter(labels)
    avg_num_nodes = sum(num_nodes) / len(num_nodes)
    min_nodes = min(num_nodes)
    max_nodes = max(num_nodes)

    small_threshold = 30
    large_threshold = 60

    small_graphs = []
    medium_graphs = []
    large_graphs = []

    classes = {
        "class_0": [],
        "class_1": [],
        "class_2": [],
        "class_3": [],
        "class_4": [],
        "class_5": [],
    }

    for i in dataset:
        c = i.y
        class_name = "class_" + str(c.item())
        classes[class_name].append(i)

    print(f"Total number of graphs: {num_graphs}")
    if num_classes:
        print(f"Number of classes: {num_classes}")
    if num_node_features:
        print(f"Number of node features: {num_node_features}")
    print(f"Average number of nodes per graph: {int(avg_num_nodes)}")
    print(f"Max number of nodes in a graph: {max_nodes}")
    print(f"Min number of nodes in a graph: {min_nodes}")

    print("Class distribution:")
    for cls, count in class_distribution.items():
        print(f" - Class {cls}: {count} graphs ({100 * count / num_graphs:.2f}%)")

    plt.figure(figsize=(10, 6))
    plt.hist(num_nodes, bins=30, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Number of Nodes in {dataset.name} Dataset")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    for graph in dataset:
        if graph.num_nodes < small_threshold:
            small_graphs.append(graph)
        elif graph.num_nodes > large_threshold:
            large_graphs.append(graph)
        else:
            medium_graphs.append(graph)

    print(f"Small graphs: {len(small_graphs)}")
    print(f"Medium graphs: {len(medium_graphs)}")
    print(f"Large graphs: {len(large_graphs)}")

    return small_graphs, medium_graphs, large_graphs, classes


def create_graph_pairs(dataset, num_pairs):
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


def create_clustered_graph(node_features, clusters, original_edge_index):
    num_clusters = torch.unique(clusters)
    supernode_features = scatter_mean(
        node_features, clusters, dim=0, dim_size=num_clusters
    )
    node_to_cluster = clusters[original_edge_index]
    supernode_edge_index = torch.unique(node_to_cluster, dim=1)
    clustered_graph = Data(x=supernode_features, edge_index=supernode_edge_index)

    return clustered_graph


def entropy(labels):
    _, label_counts = np.unique(labels, return_counts=True)
    probabilities = label_counts / label_counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy


def nmi(y_true, y_pred):
    expanded_true, expanded_pred = [], []
    for true_classes, pred in zip(y_true, y_pred):
        if pred in true_classes:
            expanded_true.append(true_classes[0])
        else:
            expanded_true.append(-1)
        expanded_pred.append(pred)

    entropy_true = entropy(expanded_true)
    entropy_pred = entropy(expanded_pred)
    mutual_info = mutual_info_score(expanded_true, expanded_pred)

    nmi = mutual_info / np.mean([entropy_true, entropy_pred])
    return nmi


def plot_graph_pair(data1, data2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    G1 = to_networkx(data1, to_undirected=True)
    nx.draw(
        G1,
        ax=axes[0],
        with_labels=False,
        node_color="skyblue",
        node_size=500,
        edge_color="k",
        linewidths=2,
        font_size=15,
        pos=nx.spring_layout(G1),
    )
    axes[0].set_title("Graph 1")

    G2 = to_networkx(data2, to_undirected=True)
    nx.draw(
        G2,
        ax=axes[1],
        with_labels=False,
        node_color="skyblue",
        edge_color="k",
        linewidths=2,
        font_size=15,
        pos=nx.spring_layout(G2),
    )
    axes[1].set_title("Graph 2")

    plt.tight_layout()
    plt.show()


def plot_graphs(
    edge_index, layer_clusters, graph1, graph2, sim, title="", trainable=False
):
    num_rows = len(layer_clusters)
    fig, axes = plt.subplots(num_rows, 4, figsize=(32, num_rows * 8))

    if num_rows > 1:
        x_pos = (
            axes[0][1].get_position().x1
            + (axes[0][2].get_position().x0 - axes[0][1].get_position().x1) / 2
        )
        fig.patches.extend(
            [
                plt.Rectangle(
                    (x_pos, 0), 0.001, 0.95, transform=fig.transFigure, color="black"
                )
            ]
        )
        fig.text(
            0.25, 0.95, f"Graph1 ({graph1})", ha="center", va="center", fontsize=17
        )
        fig.text(
            0.75, 0.95, f"Graph2 ({graph2})", ha="center", va="center", fontsize=17
        )
        fig.text(0.5, 0.97, f"Sim: {sim}", ha="center", va="center", fontsize=17)
    else:
        x_pos = (
            axes[1].get_position().x1
            + (axes[2].get_position().x0 - axes[1].get_position().x1) / 2
        )
        fig.patches.extend(
            [
                plt.Rectangle(
                    (x_pos, 0), 0.001, 0.9, transform=fig.transFigure, color="black"
                )
            ]
        )
        fig.text(0.25, 0.9, f"Graph1 ({graph1})", ha="center", va="center", fontsize=17)
        fig.text(0.75, 0.9, f"Graph2 ({graph2})", ha="center", va="center", fontsize=17)
        fig.text(0.5, 0.95, f"Sim: {sim}", ha="center", va="center", fontsize=17)

    fig.suptitle(title, fontsize=20, va="bottom", ha="center")

    axes = axes.flatten()

    def draw_graph(ax, G, title, node_colors=None):
        pos = nx.spring_layout(G)
        if node_colors:
            nx.draw(
                G,
                pos,
                ax=ax,
                with_labels=True,
                node_color=node_colors,
                cmap=plt.get_cmap("viridis"),
            )
        else:
            nx.draw(G, pos, ax=ax, with_labels=True)
        ax.set_title(title)

    for i, clusters in enumerate(layer_clusters):
        ax_index = 4 * i
        G_clustered = nx.Graph()
        for source, target in edge_index[0].T:
            G_clustered.add_edge(int(source), int(target))
        axes[ax_index].set_axis_off()
        draw_graph(axes[ax_index], G_clustered, f"Layer {i+1}")

        ax_index = 4 * i + 1
        if trainable:
            supergraph = to_networkx(clusters[0], to_undirected=True)
        else:
            G = nx.Graph()
            G.add_edges_from(edge_index[0].t().tolist())
            supernode_mapping = {node: int(cls) for node, cls in enumerate(clusters[0])}
            supergraph = nx.Graph()
            for cls in torch.unique(clusters[0]):
                supergraph.add_node(cls.item())
            for n1, n2 in G.edges():
                sn1, sn2 = supernode_mapping[n1], supernode_mapping[n2]
                if sn1 != sn2:
                    supergraph.add_edge(sn1, sn2)
            axes[ax_index].set_axis_off()
        draw_graph(axes[ax_index], supergraph, f"Layer {i+1} Clustered", None)

        ax_index = 4 * i + 2
        G_clustered = nx.Graph()
        for source, target in edge_index[1].T:
            G_clustered.add_edge(int(source), int(target))
        axes[ax_index].set_axis_off()
        draw_graph(axes[ax_index], G_clustered, f"Layer {i+1}")

        ax_index = 4 * i + 3
        if trainable:
            supergraph = to_networkx(clusters[1], to_undirected=True)
        else:
            G = nx.Graph()
            G.add_edges_from(edge_index[1].t().tolist())
            supernode_mapping = {node: int(cls) for node, cls in enumerate(clusters[1])}
            supergraph = nx.Graph()
            for cls in torch.unique(clusters[1]):
                supergraph.add_node(cls.item())
            for n1, n2 in G.edges():
                sn1, sn2 = supernode_mapping[n1], supernode_mapping[n2]
                if sn1 != sn2:
                    supergraph.add_edge(sn1, sn2)
        axes[ax_index].set_axis_off()
        draw_graph(axes[ax_index], supergraph, f"Layer {i+1} Clustered", None)

    plt.tight_layout()
    if num_rows > 1:
        fig.subplots_adjust(top=0.9)
    else:
        fig.subplots_adjust(top=0.8)
    plt.show()


def normalize_attention(a_x_s):
    a_min, a_max = a_x_s.min(), a_x_s.max()
    if a_max > a_min:
        return (a_x_s - a_min) / (a_max - a_min)
    return a_x_s


def visualize_graphs_with_attention(
    graph1, graph2, a_x_s, a_y_s, threshold=0.9, topk=None
):
    G1 = to_networkx(graph1, to_undirected=True)
    G2 = to_networkx(graph2, to_undirected=True)

    pos1 = nx.kamada_kawai_layout(G1)
    pos2 = nx.kamada_kawai_layout(G2)

    pos2_shifted = {k: [v[0] + 3, v[1]] for k, v in pos2.items()}

    plt.figure(figsize=(24, 14))
    nx.draw(
        G1, pos1, with_labels=True, node_color="skyblue", edge_color="k", node_size=700
    )
    nx.draw(
        G2,
        pos2_shifted,
        with_labels=True,
        node_color="lightcoral",
        edge_color="k",
        node_size=700,
    )

    combined_pos = {**pos1, **{k + len(G1): v for k, v in pos2_shifted.items()}}

    a_x_s = normalize_attention(a_x_s)
    a_y_s = normalize_attention(a_y_s).T

    def keep_topk(tensor, k):
        topk_values, topk_indices = torch.topk(tensor, k=k, dim=1)
        zero_tensor = torch.zeros_like(tensor)
        zero_tensor.scatter_(1, topk_indices, topk_values)

        return zero_tensor

    if topk:
        a_x_s = keep_topk(a_x_s, topk)
        a_y_s = keep_topk(a_y_s, topk)

    for i, j in torch.nonzero(a_x_s > threshold):
        src = i.item()
        target = j.item() + len(G1)
        weight = a_x_s[i, j].item()
        plt.plot(
            [combined_pos[src][0], combined_pos[target][0]],
            [combined_pos[src][1], combined_pos[target][1]],
            color="blue",
            alpha=min(weight * 5, 1.0),
            lw=weight * 2,
        )

    for i, j in torch.nonzero(a_y_s > threshold):
        src = i.item() + len(G1)
        target = j.item()
        weight = a_y_s[i, j].item()
        plt.plot(
            [combined_pos[src][0], combined_pos[target][0]],
            [combined_pos[src][1], combined_pos[target][1]],
            color="red",
            alpha=min(weight * 5, 1.0),
            lw=weight * 2,
        )

    plt.axis("off")
    plt.show()