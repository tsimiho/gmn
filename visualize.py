import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch_geometric.utils


def normalize_attention_values(values):
    min_val = values.min()
    max_val = values.max()
    return (values - min_val) / (max_val - min_val)


def visualize_attention(
    graph, top_indices_x, top_values_x, top_indices_y, top_values_y, normalize=True
):
    G = torch_geometric.utils.to_networkx(graph, to_undirected=True)

    fig, ax = plt.subplots(figsize=(24, 12))

    pos_g1 = nx.spring_layout(G, k=0.15, iterations=50)
    pos_g2 = {node: (x + 2.0, y) for node, (x, y) in pos_g1.items()}

    nx.draw(
        G,
        pos_g1,
        ax=ax,
        with_labels=True,
        node_color="lightblue",
        edge_color="grey",
        width=1,
        node_size=300,
    )

    nx.draw(
        G,
        pos_g2,
        ax=ax,
        with_labels=True,
        node_color="lightgreen",
        edge_color="grey",
        width=1,
        node_size=300,
    )

    if normalize:
        top_values_x = normalize_attention_values(top_values_x)
        top_values_y = normalize_attention_values(top_values_y)

    for i in range(top_indices_x.shape[0]):
        for j in range(top_indices_x.shape[1]):
            node_from = i
            node_to = top_indices_x[i, j].item()
            value = top_values_x[i, j].item()
            alpha = min(max(value, 0), 1)
            ax.annotate(
                "",
                xy=pos_g2[node_to],
                xycoords="data",
                xytext=pos_g1[node_from],
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="red", alpha=alpha),
            )

    for i in range(top_indices_y.shape[0]):
        for j in range(top_indices_y.shape[1]):
            node_from = i
            node_to = top_indices_y[i, j].item()
            value = top_values_y[i, j].item()
            alpha = min(max(value, 0), 1)
            ax.annotate(
                "",
                xy=pos_g1[node_to],
                xycoords="data",
                xytext=pos_g2[node_from],
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color="blue", alpha=alpha),
            )

    plt.axis("off")
    plt.show()
