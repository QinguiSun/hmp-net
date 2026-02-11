# models/schnet_cc.py
from typing import Optional

import torch
from torch_geometric.nn import SchNet
from torch_scatter import scatter_add
from torch_cluster import radius_graph

from torch import Tensor
from torch.nn import Linear, Sequential
from torch_geometric.nn import MessagePassing
from math import pi as PI
import torch.nn.functional as F


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

        self.backbone.interaction = InteractionBlock(
            hidden_channels = hidden_channels,
            num_gaussians = num_gaussians,
            num_filters = num_filters,
            cutoff = cutoff
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

        # ------add virtual node-------
        num = atoms.length()
        edge_index_vn = torch.zeros([2, num*num])
        for i in num*(num-1):
            edge_index_vn[0,i] = i // num
            edge_index_vn[1,i] = i % num
        # remove the self-loop of edge_index_vn
        

        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.backbone.distance_expansion(edge_weight)

        # ------add virtual node-------
        num = atoms.length()
        edge_index_vn = torch.zeros([2, num*num])
        for i in num*(num-1):
            edge_index_vn[0,i] = i // num
            edge_index_vn[1,i] = i % num
        # remove the self-loop of edge_index_vn
        
        row_vn, col_vn = edge_index_vn
        edge_weight_vn = (pos[row_vn] - pos[col_vn]).norm(dim=-1)
        edge_attr_vn = self.backbone.distance_expansion(edge_weight_vn)

        h = self.backbone.embedding(atoms)
        h_vn = torch.zeros(h)
        for interaction in self.backbone.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            h_vn = h_vn + interaction(h, edge_index_vn, edge_weight_vn, edge_attr_vn)

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
    
class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int,
                 num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor,
                edge_attr: Tensor) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift