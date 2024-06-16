import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree, to_networkx, to_undirected
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


def analyze_dataset(dataset, min_num_nodes=None, max_num_nodes=None):
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

    small_threshold = 20
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

    small_classes = {
        "class_0": [],
        "class_1": [],
        "class_2": [],
        "class_3": [],
        "class_4": [],
        "class_5": [],
    }

    medium_classes = {
        "class_0": [],
        "class_1": [],
        "class_2": [],
        "class_3": [],
        "class_4": [],
        "class_5": [],
    }

    large_classes = {
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
        if min_num_nodes and max_num_nodes:
            if min_num_nodes <= i.num_nodes <= max_num_nodes:
                classes[class_name].append(i)
        else:
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
        c = graph.y
        class_name = "class_" + str(c.item())
        if graph.num_nodes < small_threshold:
            small_graphs.append(graph)
            small_classes[class_name].append(graph)
        elif graph.num_nodes > large_threshold:
            large_graphs.append(graph)
            large_classes[class_name].append(graph)
        else:
            medium_graphs.append(graph)
            medium_classes[class_name].append(graph)

    print(f"Small graphs: {len(small_graphs)}")
    print(f"Medium graphs: {len(medium_graphs)}")
    print(f"Large graphs: {len(large_graphs)}")

    return (
        small_graphs,
        medium_graphs,
        large_graphs,
        classes,
        small_classes,
        medium_classes,
        large_classes,
    )


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


def plot_graph_pair(data1, data2, title1="", title2="", title=""):
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
    axes[0].set_title(title1)

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
    axes[1].set_title(title2)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_graphs(edge_index, layer_clusters, graph1, graph2, title="", trainable=False):
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
        # fig.text(0.5, 0.97, f"Sim: {sim}", ha="center", va="center", fontsize=17)
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
        # fig.text(0.5, 0.95, f"Sim: {sim}", ha="center", va="center", fontsize=17)

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


def normalize_attention(a_x_s):
    a_min, a_max = a_x_s.min(), a_x_s.max()
    if a_max > a_min:
        return (a_x_s - a_min) / (a_max - a_min)
    return a_x_s


def visualize_graphs_with_attention(
    graph1, graph2, a_x_s, a_y_s, threshold=0.9, topk=None, title=None
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

    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def plot_graph(G, title=None):
    nx.draw(
        to_networkx(G, to_undirected=True),
        with_labels=False,
        node_color="skyblue",
        node_size=500,
        edge_color="k",
        linewidths=2,
        font_size=15,
    )
    if title:
        plt.title(title)
    plt.show()


def plot_all_classes(graphs, accs, title="Title", layers=3, classes=4):
    n = len(graphs)
    # layers = 3
    cols = 2 * classes
    rows = layers

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    fig.suptitle(title, fontsize=20, va="bottom", ha="center")

    def draw_graph(ax, G, title, node_colors=None):
        pos = nx.spring_layout(G)
        if node_colors:
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color=node_colors,
                cmap=plt.get_cmap("viridis"),
            )
        else:
            nx.draw(G, pos, ax=ax, with_labels=True)
        ax.set_title(title)

    for idx, graph in enumerate(graphs):
        col = idx // rows
        row = idx % rows
        subplot_index = col + row * cols
        plt.sca(axes[subplot_index])
        supergraph = to_networkx(graph, to_undirected=True)
        draw_graph(
            axes[subplot_index], supergraph, f"Layer {row} (acc: {accs[idx]})", None
        )

    for col in range(1, cols):
        x_position = col / cols
        fig.patches.extend(
            [
                plt.Rectangle(
                    (x_position - 0.001 / 2, 0),
                    0.001,
                    0.95 if col % 2 == 0 else 0.9,
                    transform=fig.transFigure,
                    color="black" if col % 2 == 0 else "grey",
                )
            ]
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
    # plt.savefig(f"results/{title}.png", bbox_inches="tight")
    # plt.close()


def is_cycle(graph):
    node_degrees = degree(graph.edge_index[0])
    return torch.all(node_degrees == 2).item()


def is_line(graph):
    node_degrees = degree(graph.edge_index[0])
    degree_one_count = (node_degrees == 1).sum().item()
    return (
        degree_one_count == 2
        and torch.all((node_degrees == 1) | (node_degrees == 2)).item()
    )


def is_wheel(graph):
    node_degrees = degree(graph.edge_index[0])
    unique_degrees = torch.unique(node_degrees)
    if len(unique_degrees) == 2:
        high = torch.max(unique_degrees)
        low = torch.min(unique_degrees)
        return (
            (node_degrees == low).sum().item() == 1
            and low.item() == 3
            and high.item() == (node_degrees.size(0) - 1)
        )
    return False


def is_complete(graph):
    node_degrees = degree(graph.edge_index[0])
    expected_degree = graph.num_nodes - 1
    return torch.all(node_degrees == expected_degree).item()


def is_star(graph):
    if graph.edge_index.size(1) == 0:
        return False

    node_degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
    max_degree = torch.max(node_degrees).item()
    degree_zero_count = (node_degrees == 0).sum().item()

    return (max_degree == graph.num_nodes - 1 - degree_zero_count) and (
        (node_degrees == 1).sum().item() == max_degree
    )


def number_of_cycles(N, E, edges):
    graph = [[] for i in range(N)]
    for i in range(E):
        graph[edges[i][0]].append(edges[i][1])
        graph[edges[i][1]].append(edges[i][0])
    return (E - N) + 1


def norm_barplot(model, i=0):
    norms_first_graph = [
        norms_tuple[i].detach().numpy() for norms_tuple in model.norms_per_layer
    ]

    num_layers = len(norms_first_graph)
    max_nodes = max(len(norms) for norms in norms_first_graph)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.8 / max_nodes

    for node_idx in range(max_nodes):
        node_norms = [
            layer[node_idx] if node_idx < len(layer) else 0
            for layer in norms_first_graph
        ]

        x_positions = np.arange(num_layers) + bar_width * node_idx

        ax.bar(
            x_positions,
            node_norms,
            bar_width,
            label=f"Node {node_idx + 1}" if node_idx == 0 else "",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Norms of Node Features")
    ax.set_title(f"Norms of Node Features for Each Node by Layer - Graph {i+1}")
    ax.set_xticks(np.arange(num_layers) + bar_width * max_nodes / 2)
    ax.set_xticklabels([f"Layer {layer_idx + 1}" for layer_idx in range(num_layers)])

    plt.tight_layout()
    plt.show()


def cross_barplot(model, i=0):
    norms_first_graph = [
        norms_tuple[i].detach().numpy()
        for norms_tuple in model.attention_sums_per_layer
    ]

    num_layers = len(norms_first_graph)
    max_nodes = max(len(norms) for norms in norms_first_graph)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.8 / max_nodes

    for node_idx in range(max_nodes):
        node_norms = [
            layer[node_idx] if node_idx < len(layer) else 0
            for layer in norms_first_graph
        ]

        x_positions = np.arange(num_layers) + bar_width * node_idx

        ax.bar(
            x_positions,
            node_norms,
            bar_width,
            label=f"Node {node_idx + 1}" if node_idx == 0 else "",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Norms of Node Features")
    ax.set_title(f"Norms of Node Features for Each Node by Layer - Graph {i+1}")
    ax.set_xticks(np.arange(num_layers) + bar_width * max_nodes / 2)
    ax.set_xticklabels([f"Layer {layer_idx + 1}" for layer_idx in range(num_layers)])

    plt.tight_layout()
    plt.show()


def plot_layer_barplot(class0, class1, class2, class3, n):
    counts0 = [class0.count(i) for i in range(1, n + 1)]
    counts1 = [class1.count(i) for i in range(1, n + 1)]
    counts2 = [class2.count(i) for i in range(1, n + 1)]
    counts3 = [class3.count(i) for i in range(1, n + 1)]

    fig, ax = plt.subplots(figsize=(15, 8))

    bar_width = 0.15

    indices = np.arange(1, n + 1)

    bars0 = ax.bar(indices - 1.5 * bar_width, counts0, width=bar_width, label="Class 0")
    bars1 = ax.bar(indices - 0.5 * bar_width, counts1, width=bar_width, label="Class 1")
    bars2 = ax.bar(indices + 0.5 * bar_width, counts2, width=bar_width, label="Class 2")
    bars3 = ax.bar(indices + 1.5 * bar_width, counts3, width=bar_width, label="Class 3")

    ax.set_xlabel("Number", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_title("Frequency of Numbers by Class", fontsize=16)

    ax.set_xticks(indices)
    ax.set_xticklabels([str(i) for i in indices])

    ax.legend()

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    add_value_labels(bars0)
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()
    plt.show()


def best_k(combined_scores, threshold=0.8):
    def calculate_cumulative_scores(scores):
        normalized_scores = F.softmax(scores, dim=0)
        cumulative_scores = torch.cumsum(normalized_scores, dim=0)
        return cumulative_scores

    def calculate_confidence(cumulative_scores, k):
        gradients = np.gradient(cumulative_scores.detach().numpy())
        confidence = gradients[k - 1]
        return confidence

    def find_best_k_and_confidence(cumulative_scores, total_percentage):
        k = ((cumulative_scores / cumulative_scores[-1]) >= total_percentage).nonzero()[
            0
        ].item() + 1
        confidence = calculate_confidence(cumulative_scores, k)
        return k, confidence

    layer_scores_1 = []
    layer_scores_2 = []
    for i in range(len(combined_scores)):
        score1, score2 = combined_scores[i]
        layer_scores_1.append(score1)
        layer_scores_2.append(score2)

    print(layer_scores_1)
    print(layer_scores_2)

    k_values_graph1 = []
    confidences_graph1 = []
    for i, scores in enumerate(layer_scores_1):
        cumulative_scores = calculate_cumulative_scores(scores)
        best_k, confidence = find_best_k_and_confidence(cumulative_scores, threshold)
        k_values_graph1.append(best_k)
        confidences_graph1.append(confidence)

    k_values_graph2 = []
    confidences_graph2 = []
    for i, scores in enumerate(layer_scores_2):
        cumulative_scores = calculate_cumulative_scores(scores)
        best_k, confidence = find_best_k_and_confidence(cumulative_scores, threshold)
        k_values_graph2.append(best_k)
        confidences_graph2.append(confidence)

    def calculate_weighted_average(ks, confidences):
        normalized_confidences = [float(i) / sum(confidences) for i in confidences]
        weighted_ks = sum(k * w for k, w in zip(ks, normalized_confidences))
        return int(round(weighted_ks))

    combined_ks = k_values_graph1 + k_values_graph2
    combined_confidences = confidences_graph1 + confidences_graph2

    print(combined_ks)

    k = calculate_weighted_average(combined_ks, combined_confidences)

    return k


def calculate_entropy(scores):
    probabilities = F.softmax(scores, dim=0)
    log_probabilities = torch.log(probabilities + 1e-10)
    entropy = -torch.sum(probabilities * log_probabilities)
    return entropy.item()


def to_nx(x, edge_idx):
    return to_networkx(Data(x=x, edge_index=edge_idx), to_undirected=True)


def plot_mutag(
    graph1, graph2=None, original_x1=None, perm1=None, original_x2=None, perm2=None
):
    import matplotlib
    import matplotlib.patches as mpatches

    colormap = matplotlib.colormaps.get_cmap("Pastel1")

    color_map = {
        0: colormap(0),
        1: colormap(1),
        2: colormap(2),
        3: colormap(3),
        4: colormap(4),
        5: colormap(5),
        6: colormap(6),
        "other": colormap(7),
    }

    def plot_single_graph(graph, ax, original_x=None, perm=None):
        G = to_networkx(graph, to_undirected=True)

        node_colors = []
        node_labels = {}

        if original_x is not None and perm is not None:
            mapped_x = original_x[perm[: graph.num_nodes]]
            for node in range(graph.num_nodes):
                one_hot = mapped_x[node].tolist()
                try:
                    node_type = one_hot.index(1)
                except ValueError:
                    node_type = "other"
                node_colors.append(color_map[node_type])
                node_labels[node] = perm[node].item()
        else:
            for node in range(graph.num_nodes):
                one_hot = graph.x[node].tolist()
                try:
                    node_type = one_hot.index(1)
                except ValueError:
                    node_type = "other"
                node_colors.append(color_map[node_type])
                node_labels[node] = perm[node].item() if (perm is not None) else node

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            with_labels=True,
            labels=node_labels,
            node_size=500,
            font_weight="bold",
            ax=ax,
        )

    legend_handles = [
        mpatches.Patch(color=colormap(0), label="C"),
        mpatches.Patch(color=colormap(1), label="N"),
        mpatches.Patch(color=colormap(2), label="O"),
        mpatches.Patch(color=colormap(3), label="F"),
        mpatches.Patch(color=colormap(4), label="I"),
        mpatches.Patch(color=colormap(5), label="Cl"),
        mpatches.Patch(color=colormap(6), label="Br"),
        mpatches.Patch(color=colormap(7), label="-"),
    ]

    if graph2 is None:
        fig, ax = plt.subplots()
        plot_single_graph(graph1, ax, original_x1, perm1)
        fig.legend(handles=legend_handles, loc="lower left", title="Node Types")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_single_graph(graph1, axes[0], original_x1, perm1)
        plot_single_graph(graph2, axes[1], original_x2, perm2)
        axes[0].set_title("Graph 1")
        axes[1].set_title("Graph 2")
        fig.legend(handles=legend_handles, loc="lower left", title="Node Types")

    plt.show()


import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def plot_mutag_summary(
    graph1, graph2=None, original_x1=None, perm1=None, original_x2=None, perm2=None
):
    colormap = matplotlib.colormaps.get_cmap("Pastel1")

    color_map = {
        0: colormap(0),
        1: colormap(1),
        2: colormap(2),
        3: colormap(3),
        4: colormap(4),
        5: colormap(5),
        6: colormap(6),
        "other": colormap(7),
    }

    def plot_single_graph(graph, ax, original_x=None, perm=None, summary_ids=None):
        G = to_networkx(graph, to_undirected=True)

        node_colors = []
        node_labels = {}

        if original_x is not None and perm is not None:
            mapped_x = original_x[perm[: graph.num_nodes]]
            for node in range(graph.num_nodes):
                one_hot = mapped_x[node].tolist()
                try:
                    node_type = one_hot.index(1)
                except ValueError:
                    node_type = "other"
                node_colors.append(color_map[node_type])
                node_labels[node] = perm[node].item()
        else:
            for node in range(graph.num_nodes):
                one_hot = graph.x[node].tolist()
                try:
                    node_type = one_hot.index(1)
                except ValueError:
                    node_type = "other"
                node_colors.append(color_map[node_type])
                if summary_ids and node < len(summary_ids):
                    node_labels[node] = summary_ids[node]
                else:
                    node_labels[node] = (
                        perm[node].item() if (perm is not None) else node
                    )

        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            with_labels=True,
            labels=node_labels,
            node_size=500,
            font_weight="bold",
            ax=ax,
        )

    legend_handles = [
        mpatches.Patch(color=colormap(0), label="C"),
        mpatches.Patch(color=colormap(1), label="N"),
        mpatches.Patch(color=colormap(2), label="O"),
        mpatches.Patch(color=colormap(3), label="F"),
        mpatches.Patch(color=colormap(4), label="I"),
        mpatches.Patch(color=colormap(5), label="Cl"),
        mpatches.Patch(color=colormap(6), label="Br"),
        mpatches.Patch(color=colormap(7), label="-"),
    ]

    if graph2 is None:
        fig, ax = plt.subplots()
        plot_single_graph(graph1, ax, original_x1, perm1)
        fig.legend(handles=legend_handles, loc="lower left", title="Node Types")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_single_graph(graph1, axes[0], original_x1, perm1)
        plot_single_graph(
            graph2, axes[1], original_x2, perm2, summary_ids=graph2.summary_ids
        )
        axes[0].set_title("Graph 1")
        axes[1].set_title("Graph 2")
        fig.legend(handles=legend_handles, loc="lower left", title="Node Types")

    plt.show()
