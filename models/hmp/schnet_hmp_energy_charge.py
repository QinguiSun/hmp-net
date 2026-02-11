# models/hmp/schnet_hmp_energy_charge.py
import torch
from torch import nn
from math import pi as PI

from torch_geometric.nn.models.schnet import InteractionBlock
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_cluster import radius_graph
from torch_scatter import scatter_add, scatter_softmax

from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SchNetInteraction(nn.Module):
    """
    InteractionBlock wrapper supporting virtual edge reweighting via attention (softmax on virtual edges per source node).
    """
    def __init__(self, interaction_block: InteractionBlock, hidden_channels: int, num_gaussians: int):
        super().__init__()
        self.interaction_block = interaction_block
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + num_gaussians, hidden_channels),
            #nn.ReLU(),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, h, pos, edge_index, virtual_edge_mask=None, size=None):
        row, col = edge_index
        #edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        diff = pos[row] - pos[col]
        edge_weight = torch.sqrt((diff * diff).sum(dim=-1) + 1e-9)
        
        with torch.no_grad():
            if edge_weight.numel() > 0:
                md = float(edge_weight.min().item())
                if md < 1e-5:
                    print(f"[WARN] very small edge distance: {md:.3e} (may break 2nd derivatives)")


        edge_attr = self.interaction_block.distance_expansion(edge_weight)

        full_decay_weights = torch.ones_like(edge_weight)

        if virtual_edge_mask is not None and virtual_edge_mask.any():
            h_i = h[row]
            h_j = h[col]
            score = self.attn_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1)).squeeze(-1)

            score_v = score[virtual_edge_mask]
            row_v = row[virtual_edge_mask]
            decay_v = scatter_softmax(score_v, row_v)  # normalize over outgoing virtual edges per source
            full_decay_weights[virtual_edge_mask] = decay_v

        edge_attr = edge_attr * full_decay_weights.unsqueeze(-1)
        h_update = self.interaction_block(h, edge_index, edge_weight, edge_attr)
        return h_update, pos


class HMPLayer(nn.Module):
    def __init__(
        self,
        backbone_layer: SchNetInteraction,
        h_dim: int,
        s_dim: int,
        master_selection_hidden_dim: int,
        lambda_attn: float,
        master_rate: float,
    ):
        super().__init__()
        self.backbone_layer = backbone_layer
        self.s_dim = s_dim
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim, ratio=master_rate)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)

    def forward(self, h, pos, edge_index, batch):
        num_nodes = h.size(0)
        h_local = h
        pos_local = pos

        h_scalar = h_local[:, : self.s_dim]
        m, _ = self.master_selection(h_scalar)
        master_nodes_mask = m > 0.5
        num_master_nodes = int(master_nodes_mask.sum().item())

        if num_master_nodes <= 1:
            return h_local, pos_local, None

        master_indices = torch.where(master_nodes_mask)[0]
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)

        if edge_index_induced.numel() > 0:
            adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        else:
            adj_induced = torch.zeros((num_master_nodes, num_master_nodes), device=h.device)

        h_master = h_local[master_nodes_mask]
        pos_master = pos_local[master_nodes_mask]
        h_master_scalar = h_master[:, : self.s_dim]

        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)

        edge_reindex_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_virtual, _ = remove_self_loops(edge_reindex_virtual)
        edge_index_virtual = to_undirected(edge_index_virtual, num_nodes=h_master.size(0))

        # merge induced + virtual edges
        num_induced_edges = edge_index_induced.shape[1]
        num_virtual_edges = edge_index_virtual.shape[1]
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)

        virtual_edge_mask = torch.cat(
            [
                torch.zeros(num_induced_edges, dtype=torch.bool, device=h.device),
                torch.ones(num_virtual_edges, dtype=torch.bool, device=h.device),
            ],
            dim=0,
        )

        h_master_update, _ = self.backbone_layer(h_master, pos_master, edge_index_master, virtual_edge_mask=virtual_edge_mask)
        h_hierarchical = h_master + h_master_update

        h_hierarchical_expanded = torch.zeros_like(h_local)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical

        m_expanded = m.unsqueeze(1)
        h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded

        # one more local update on original graph
        h_update, _ = self.backbone_layer(h_final, pos, edge_index)
        h_out = h_final + h_update

        return h_out, pos_local, edge_reindex_virtual


class HMP_SchNetEnergyCharge(nn.Module):
    """
    HMP-SchNet producing per-atom energy/charge and graph totals:
      e_atom: [N]
      q_atom: [N]
      E_total: [G] = sum_i e_atom_i
      Q_total: [G] = sum_i q_atom_i
    Force is computed in training script via autograd: F = -dE/dR
    """
    def __init__(
        self,
        num_layers: int = 4,
        hidden_channels: int = 128,
        num_embeddings: int = 2,  # H/C
        num_filters: int = 128,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        s_dim: int = 32,
        master_selection_hidden_dim: int = 64,
        lambda_attn: float = 0.1,
        master_rate: float = 0.25,
    ):
        super().__init__()
        self.master_rate = master_rate
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_gaussians = num_gaussians

        self.emb_in = nn.Embedding(num_embeddings, hidden_channels)

        # SchNet distance expansion
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.hmp_layers = nn.ModuleList()
        for _ in range(num_layers):
            interaction = InteractionBlock(
                hidden_channels=hidden_channels,
                num_gaussians=num_gaussians,
                num_filters=num_filters,
                cutoff=cutoff,
            )
            interaction.distance_expansion = self.distance_expansion
            schnet_layer = SchNetInteraction(interaction, hidden_channels=hidden_channels, num_gaussians=num_gaussians)

            self.hmp_layers.append(
                HMPLayer(
                    backbone_layer=schnet_layer,
                    h_dim=hidden_channels,
                    s_dim=s_dim,
                    master_selection_hidden_dim=master_selection_hidden_dim,
                    lambda_attn=lambda_attn,
                    master_rate=master_rate,
                )
            )

        # per-atom heads (match SchNet style: Linear -> SiLU -> Linear)
        self.atom_trunk = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
        )
        self.energy_lin = nn.Linear(hidden_channels // 2, 1)
        self.charge_lin = nn.Linear(hidden_channels // 2, 1)

    def update_tau(self, epoch, n_epochs):
        initial_tau = 1.0
        final_tau = 0.1
        anneal_epochs = n_epochs // 2
        if epoch <= anneal_epochs:
            tau = initial_tau - (initial_tau - final_tau) * (epoch / anneal_epochs)
        else:
            tau = final_tau
        for layer in self.hmp_layers:
            layer.master_selection.tau.fill_(tau)

    def forward(self, batch):
        # atom types: CarbonChain 推荐你在 Dataset 内提供 node_atom (H=0,C=1)
        atoms = getattr(batch, "node_atom", None)
        if atoms is None:
            atoms = getattr(batch, "z", None)
        if atoms is None:
            raise AttributeError("Batch must have `node_atom` (H=0,C=1) or `z`.")
        atoms = atoms.long()

        pos = batch.pos
        b = batch.batch

        # ensure edge_index
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            edge_index = radius_graph(
                batch.pos.detach(),
                r=self.cutoff,
                batch=batch.batch,
                loop=False,
                max_num_neighbors=self.max_num_neighbors,
            )


        h = self.emb_in(atoms)

        for layer in self.hmp_layers:
            h, pos, _ = layer(h, pos, edge_index, b)

        h2 = self.atom_trunk(h)
        e_atom = self.energy_lin(h2).view(-1)   # [N]
        q_atom = self.charge_lin(h2).view(-1)   # [N]

        E_total = scatter_add(e_atom, b, dim=0)  # [G]
        Q_total = scatter_add(q_atom, b, dim=0)  # [G]

        return {
            "e_atom": e_atom,
            "q_atom": q_atom,
            "E_total": E_total,
            "Q_total": Q_total,
        }
