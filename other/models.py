import os
import os.path as osp
import sys
from math import ceil

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import (
    DMoNPooling,
    GCNConv,
    GraphConv,
    Sequential,
    dense_mincut_pool,
)
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class DMoN(torch.nn.Module):
    def __init__(
        self,
        mp_units,
        mp_act,
        in_channels,
        n_clusters,
        mlp_units=[],
        mlp_act="Identity",
    ):
        super().__init__()
        self.name = "DMoN"

        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        mp = [
            (
                GCNConv(in_channels, mp_units[0], normalize=False, cached=False),
                "x, edge_index -> x",
            ),
            mp_act,
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GCNConv(
                        mp_units[i], mp_units[i + 1], normalize=False, cached=False
                    ),
                    "x, edge_index -> x",
                )
            )
            mp.append(mp_act)
        self.mp = Sequential("x, edge_index", mp)
        out_chan = mp_units[-1]

        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

        self.dmon_pooling = DMoNPooling(mp_units[-1], n_clusters)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.mp(x, edge_index)
        s = self.mlp(x)
        adj = utils.to_dense_adj(edge_index)
        s, x_pooled, adj_pooled, spectral_loss, ortho_loss, cluster_loss = (
            self.dmon_pooling(x, adj)
        )

        loss = spectral_loss + ortho_loss + cluster_loss

        return torch.softmax(s[0], dim=-1), loss


def just_balance_pool(x, adj, s, mask=None, normalize=True):
    EPS = 1e-15
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    ss = torch.matmul(s.transpose(1, 2), s)
    ss_sqrt = torch.sqrt(ss + EPS)
    loss = torch.mean(-torch.einsum("ijj->i", ss_sqrt))
    if normalize:
        loss = loss / torch.sqrt(torch.tensor(num_nodes * k))

    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, loss


class JustBalance(torch.nn.Module):
    def __init__(
        self,
        mp_units,
        mp_act,
        in_channels,
        n_clusters,
        mlp_units=[],
        mlp_act="Identity",
    ):
        super().__init__()
        self.name = "JustBalance"

        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        mp = [
            (
                GCNConv(in_channels, mp_units[0], normalize=False, cached=False),
                "x, edge_index -> x",
            ),
            mp_act,
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GCNConv(
                        mp_units[i], mp_units[i + 1], normalize=False, cached=False
                    ),
                    "x, edge_index -> x",
                )
            )
            mp.append(mp_act)
        self.mp = Sequential("x, edge_index", mp)
        out_chan = mp_units[-1]

        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.mp(x, edge_index)
        s = self.mlp(x)
        adj = utils.to_dense_adj(edge_index)
        _, _, b_loss = just_balance_pool(x, adj, s)
        return torch.softmax(s, dim=-1), b_loss


class MinCut(torch.nn.Module):
    def __init__(
        self,
        mp_units,
        mp_act,
        in_channels,
        n_clusters,
        mlp_units=[],
        mlp_act="Identity",
    ):
        super().__init__()
        self.name = "MinCut"

        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        mp = [
            (
                GCNConv(in_channels, mp_units[0], normalize=False, cached=False),
                "x, edge_index -> x",
            ),
            mp_act,
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GCNConv(
                        mp_units[i], mp_units[i + 1], normalize=False, cached=False
                    ),
                    "x, edge_index -> x",
                )
            )
            mp.append(mp_act)
        self.mp = Sequential("x, edge_index", mp)
        out_chan = mp_units[-1]

        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

        self.dmon_pooling = DMoNPooling(mp_units[-1], n_clusters)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.mp(x, edge_index)
        s = self.mlp(x)
        adj = utils.to_dense_adj(edge_index)
        x_pooled, adj_pooled, mc_loss, o_loss = dense_mincut_pool(x, adj, s)

        loss = mc_loss + o_loss

        return torch.softmax(s, dim=-1), loss
