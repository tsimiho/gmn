import time

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, from_networkx, to_networkx

from custom.utils import *


def create_subgraphs(pattern, graph1, graph2):
    g1_nodes = set()
    g2_nodes = set()
    for g1_node, g2_node in pattern.items():
        g1_nodes.add(g1_node)
        g2_nodes.add(g2_node)

    g1_node_map = {node: idx for idx, node in enumerate(g1_nodes)}
    g2_node_map = {node: idx for idx, node in enumerate(g2_nodes)}

    g1_edge_index = []
    for edge in graph1.edge_index.t():
        if edge[0].item() in g1_nodes and edge[1].item() in g1_nodes:
            g1_edge_index.append(
                [g1_node_map[edge[0].item()], g1_node_map[edge[1].item()]]
            )

    g2_edge_index = []
    for edge in graph2.edge_index.t():
        if edge[0].item() in g2_nodes and edge[1].item() in g2_nodes:
            g2_edge_index.append(
                [g2_node_map[edge[0].item()], g2_node_map[edge[1].item()]]
            )

    g1_subgraph = Data(
        x=graph1.x[list(g1_nodes)],
        edge_index=torch.tensor(g1_edge_index, dtype=torch.long).t().contiguous(),
        original_node_ids=torch.tensor(list(g1_nodes), dtype=torch.long),
    )
    g2_subgraph = Data(
        x=graph2.x[list(g2_nodes)],
        edge_index=torch.tensor(g2_edge_index, dtype=torch.long).t().contiguous(),
        original_node_ids=torch.tensor(list(g2_nodes), dtype=torch.long),
    )

    return g1_subgraph, g2_subgraph


def print_patterns(patterns, graph1, graph2):
    for i in patterns:
        g1_subgraph, g2_subgraph = create_subgraphs(i, graph1, graph2)

        if nx.is_isomorphic(
            to_networkx(g1_subgraph, to_undirected=True),
            to_networkx(g2_subgraph, to_undirected=True),
        ):

            plot_mutag(
                g1_subgraph,
                g2_subgraph,
                perm1=g1_subgraph.original_node_ids,
                perm2=g2_subgraph.original_node_ids,
            )


def mutual_pairs(attention_nodes, i=0):
    outer_layer = attention_nodes[i]
    g1_attention, g2_attention = outer_layer

    mutual_pairs = []

    for g1_node, g1_attends in enumerate(g1_attention):
        for g2_node in g1_attends:
            if g1_node in g2_attention[g2_node]:
                pair = (g1_node, g2_node)
                if pair not in mutual_pairs:
                    mutual_pairs.append(pair)

    random.shuffle(mutual_pairs)
    return mutual_pairs


def vf2(G1, G2, mp):
    max_size = 0
    all_mappings = []
    unique_mappings = set()
    start_time = time.time()

    G1_degrees = degree(G1.edge_index[0], G1.num_nodes)
    G2_degrees = degree(G2.edge_index[0], G2.num_nodes)

    nodes1 = list(range(G1.num_nodes))
    nodes2 = list(range(G2.num_nodes))

    def feasible(n1, n2, M):
        if not torch.equal(G1.x[n1], G2.x[n2]):
            return False
        if (n1, n2) not in mp:
            return False

        count1 = 0
        count2 = 0

        for neighbor in G1.edge_index[1][G1.edge_index[0] == n1]:
            if neighbor.item() in M:
                count1 += 1

        for neighbor in G2.edge_index[1][G2.edge_index[0] == n2]:
            if neighbor.item() in M.values():
                count2 += 1

        if count1 != count2:
            return False

        for neighbor in G1.edge_index[1][G1.edge_index[0] == n1]:
            if (
                neighbor.item() in M
                and not (
                    G2.edge_index[1][G2.edge_index[0] == n2] == M[neighbor.item()]
                ).any()
            ):
                return False

        return True

    def canonical_form(M):
        G1_set = set(M.keys())
        G2_set = set(M.values())
        return frozenset(G1_set), frozenset(G2_set)

    def match(M, neighbors1, neighbors2):
        nonlocal max_size, all_mappings, unique_mappings

        if len(M) > max_size:
            max_size = len(M)
            all_mappings = [M.copy()]
            unique_mappings = {canonical_form(M)}
        elif len(M) == max_size:
            canonical = canonical_form(M)
            if canonical not in unique_mappings:
                all_mappings.append(M.copy())
                unique_mappings.add(canonical)

        candidates1 = sorted(neighbors1, key=lambda n: -G1_degrees[n].item())
        candidates2 = sorted(neighbors2, key=lambda n: -G2_degrees[n].item())

        for n1 in candidates1:
            if n1 not in M:
                for n2 in candidates2:
                    if n2 not in M.values() and feasible(n1, n2, M):
                        M[n1] = n2
                        new_neighbors1 = set(
                            G1.edge_index[1][G1.edge_index[0] == n1].tolist()
                        )
                        new_neighbors2 = set(
                            G2.edge_index[1][G2.edge_index[0] == n2].tolist()
                        )
                        neighbors1.update(new_neighbors1 - set(M.keys()))
                        neighbors2.update(new_neighbors2 - set(M.values()))
                        match(M, neighbors1, neighbors2)
                        del M[n1]
                        neighbors1.difference_update(new_neighbors1)
                        neighbors2.difference_update(new_neighbors2)

    for n1 in nodes1:
        for n2 in nodes2:
            if (n1, n2) in mp:
                M = {n1: n2}
                neighbors1 = set(G1.edge_index[1][G1.edge_index[0] == n1].tolist())
                neighbors2 = set(G2.edge_index[1][G2.edge_index[0] == n2].tolist())
                match(M, neighbors1, neighbors2)

    return all_mappings


def mcsplit(data1, data2, mp=None):
    def pyg_to_nx(data):
        return to_networkx(data, to_undirected=True)

    def nx_to_pyg(G):
        return from_networkx(G)

    def find_mcs(G1, G2, mp):
        incumbent = []
        max_size = 0

        def search(mapping, future, incumbent):
            nonlocal max_size
            if len(mapping) > max_size:
                max_size = len(mapping)
                incumbent = mapping[:]
            bound = len(mapping) + sum(min(len(g), len(h)) for g, h in future)
            if bound <= max_size:
                return incumbent

            G, H = min(future, key=lambda x: max(len(x[0]), len(x[1])))
            v = max(G, key=lambda x: G1.degree[x])
            for w in H:
                if (mp != None) and ((v, w) not in mp):
                    continue
                future_prime = []
                for G_prime, H_prime in future:
                    G_prime_adj = [
                        u for u in G_prime if u != v and u in G1.neighbors(v)
                    ]
                    H_prime_adj = [
                        u for u in H_prime if u != w and u in G2.neighbors(w)
                    ]
                    if G_prime_adj and H_prime_adj:
                        future_prime.append((G_prime_adj, H_prime_adj))
                    G_prime_non_adj = [
                        u for u in G_prime if u != v and u not in G1.neighbors(v)
                    ]
                    H_prime_non_adj = [
                        u for u in H_prime if u != w and u not in G2.neighbors(w)
                    ]
                    if G_prime_non_adj and H_prime_non_adj:
                        future_prime.append((G_prime_non_adj, H_prime_non_adj))
                incumbent = search(mapping + [(v, w)], future_prime, incumbent)
            G_prime = [u for u in G if u != v]
            future.remove((G, H))
            if G_prime:
                future.append((G_prime, H))
            incumbent = search(mapping, future, incumbent)
            return incumbent

        future = [(list(G1.nodes), list(G2.nodes))]
        incumbent = search([], future, incumbent)

        return incumbent

    G1 = pyg_to_nx(data1)
    G2 = pyg_to_nx(data2)
    mapping = find_mcs(G1, G2, mp)
    common_nodes = [v for v, _ in mapping]
    subgraph = G1.subgraph(common_nodes)
    return nx_to_pyg(subgraph)
