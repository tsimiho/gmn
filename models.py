import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from torch_geometric.data import Data

from layers import GraphAggregator, GraphConvolution, GraphMatchingConvolution
from utils import acc_f1, adj_matrix_to_edge_index, create_batch, trim_feats, calculate_accuracy


class GraphMatchingNetwork(torch.nn.Module):
    def __init__(self, args):
        super(GraphMatchingNetwork, self).__init__()
        self.args = args
        self.margin = self.args.margin
        if args.n_classes > 2:
            self.f1_average = "micro"
        else:
            self.f1_average = "binary"
        self.layers = torch.nn.ModuleList()
        self.layers.append(
            GraphMatchingConvolution(self.args.feat_dim, self.args.dim, args)
        )
        for _ in range(self.args.num_layers - 1):
            self.layers.append(
                GraphMatchingConvolution(self.args.dim, self.args.dim, args)
            )
        self.aggregator = GraphAggregator(self.args.dim, self.args.dim, self.args)
        self.cls = torch.nn.Linear(self.args.dim * 2, self.args.n_classes)

    def compute_emb(self, feats, edge_index, batch):
        for i in range(self.args.num_layers):
            feats, edge_index, batch = self.layers[i](feats, edge_index, batch)
        feats = self.aggregator(feats, edge_index, batch)
        return feats, edge_index, batch

    def combine_pair_embedding(
        self, feats_1, edge_index_1, feats_2, edge_index_2, sizes_1, sizes_2
    ):
        feats = torch.cat([feats_1, feats_2], dim=0)
        max_node_idx_1 = sizes_1.sum()
        edge_index_2_offset = edge_index_2 + max_node_idx_1
        edge_index = torch.cat([edge_index_1, edge_index_2_offset], dim=1)
        batch = create_batch(torch.cat([sizes_1, sizes_2], dim=0))
        feats, edge_index, batch = (
            feats.to(self.args.device),
            edge_index.to(self.args.device),
            batch.to(self.args.device),
        )
        return feats, edge_index, batch

    def forward(self, feats_1, edge_index_1, feats_2, edge_index_2, sizes_1, sizes_2):
        feats, edge_index, batch = self.combine_pair_embedding(
            feats_1, edge_index_1, feats_2, edge_index_2, sizes_1, sizes_2
        )
        emb, _, _ = self.compute_emb(feats, edge_index, batch)
        emb_1 = emb[: emb.shape[0] // 2, :]
        emb_2 = emb[emb.shape[0] // 2 :, :]
        return emb_1, emb_2

    def compute_metrics(self, emb_1, emb_2, labels):
        distances = torch.norm(emb_1 - emb_2, p=2, dim=1)
        loss = F.relu(self.margin - labels * (1 - distances))
        predicted_similar = torch.where(
            distances < self.margin, torch.ones_like(labels), -torch.ones_like(labels)
        )
        acc = (predicted_similar == labels).float().item()
        # acc = calculate_accuracy(emb_1, emb_2, labels, self.margin)
        f1 = f1_score(
            labels.cpu().numpy(),
            predicted_similar.cpu().numpy(),
            average="binary",
            zero_division=0,
        )
        metrics = {"loss": loss, "acc": acc, "f1": f1}
        return metrics

    def init_metric_dict(self):
        return {"acc": -1, "f1": -1}

    def has_improved(self, m1, m2):
        return m1["acc"] < m2["acc"]
