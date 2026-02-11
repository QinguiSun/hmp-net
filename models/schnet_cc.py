# models/schnet_cc.py
from typing import Optional

import torch
from torch_geometric.nn import SchNet
from torch_scatter import scatter_add
from torch_cluster import radius_graph


class SchNetEnergyCharge(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_embeddings: int = 2,
        num_filters: int = 128,
        num_layers: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
    ):
        super().__init__()

        # 1) 自己保存一份（版本无关，永远可用）
        self.max_num_neighbors = int(max_num_neighbors)

        self.backbone = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout="add",
        )

        # 2) 同时挂到 backbone 上（让你现有 self.backbone.max_num_neighbors 写法成立）
        self.backbone.max_num_neighbors = self.max_num_neighbors

        self.backbone.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=hidden_channels,
        )

        self.energy_lin = torch.nn.Linear(hidden_channels // 2, 1)
        self.charge_lin = torch.nn.Linear(hidden_channels // 2, 1)

    def forward(self, batch):
        atoms = getattr(batch, "node_atom", None)
        if atoms is None:
            atoms = getattr(batch, "z", None)
        if atoms is None:
            raise AttributeError("Batch must have `node_atom` (H=0,C=1) or `z`.")
        atoms = atoms.long()

        pos = batch.pos
        b = batch.batch

        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            # 兼容：如果未来 backbone 又有了该属性，也能用；没有就回退到 self.max_num_neighbors
            mnn = getattr(self.backbone, "max_num_neighbors", self.max_num_neighbors)
            edge_index = radius_graph(
                pos,
                r=self.backbone.cutoff,
                batch=b,
                loop=False,
                max_num_neighbors=mnn,
            )

        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.backbone.distance_expansion(edge_weight)

        h = self.backbone.embedding(atoms)
        for interaction in self.backbone.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.backbone.lin1(h)
        h = self.backbone.act(h)

        e_atom = self.energy_lin(h).view(-1)
        q_atom = self.charge_lin(h).view(-1)

        E_total = scatter_add(e_atom, b, dim=0)
        Q_total = scatter_add(q_atom, b, dim=0)

        return {
            "e_atom": e_atom,
            "q_atom": q_atom,
            "E_total": E_total,
            "Q_total": Q_total,
        }
