import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
import e3nn
from typing import Optional


from models.mace_modules.blocks import RadialEmbeddingBlock
from models.layers.tfn_layer import TensorProductConvLayer
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration


class HMPLayer(nn.Module):
    """Hierarchical Message Passing layer wrapping a TFN convolution."""

    def __init__(
        self,
        conv: TensorProductConvLayer,
        s_dim: int,
        master_selection_hidden_dim: int,
        lambda_attn: float,
        master_rate: float,
        spherical_harmonics: e3nn.o3.SphericalHarmonics,
        radial_embedding: RadialEmbeddingBlock,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.s_dim = s_dim
        self.spherical_harmonics = spherical_harmonics
        self.radial_embedding = radial_embedding
        self.master_selection = MasterSelection(
            in_dim=s_dim, hidden_dim=master_selection_hidden_dim, ratio=master_rate
        )
        self.virtual_generation = VirtualGeneration(
            in_dim=s_dim, lambda_attn=lambda_attn
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        # Local propagation
        vectors = pos[edge_index[0]] - pos[edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        h_update = self.conv(h, edge_index, edge_sh, edge_feats)
        h_local = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))

        # Invariant topology learning
        h_scalar = h_local[:, :self.s_dim]
        m, _ = self.master_selection(h_scalar)
        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()

        if num_master_nodes <= 1:
            return h_local, pos, torch.zeros((0, 0), device=h.device), m

        master_indices = torch.where(master_nodes_mask)[0]
        edge_index_induced, _ = subgraph(
            master_indices, edge_index, relabel_nodes=True, num_nodes=h.size(0)
        )
        adj_induced = to_dense_adj(
            edge_index_induced, max_num_nodes=num_master_nodes
        ).squeeze(0)

        h_master = h_local[master_nodes_mask]
        pos_master = pos[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]

        # Virtual edge generation
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)
        edge_index_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)

        vectors_master = pos_master[edge_index_master[0]] - pos_master[edge_index_master[1]]
        lengths_master = torch.linalg.norm(vectors_master, dim=-1, keepdim=True)
        edge_sh_master = self.spherical_harmonics(vectors_master)
        edge_feats_master = self.radial_embedding(lengths_master)

        # Hierarchical update
        if self.conv.in_irreps == self.conv.out_irreps:
            h_master_update = self.conv(h_master, edge_index_master, edge_sh_master, edge_feats_master)
            h_hierarchical = h_master_update + F.pad(h_master, (0, h_master_update.shape[-1] - h_master.shape[-1]))
        else:
            # First layer: skip hierarchical update
            h_hierarchical = h_master

        h_hierarchical_expanded = torch.zeros_like(h_local)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical

        m_expanded = m.unsqueeze(1)
        h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded

        return h_final, pos, A_virtual, m


class HMP_TFNModel(nn.Module):
    """Tensor Field Network model with Hierarchical Message Passing."""

    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        num_layers: int = 5,
        emb_dim: int = 128,
        #hidden_irreps: e3nn.o3.Irreps | None = None,
        hidden_irreps: Optional[e3nn.o3.Irreps] = None,
        mlp_dim: int = 256,
        in_dim: int = 1,
        out_dim: int = 1,
        aggr: str = "sum",
        gate: bool = True,
        batch_norm: bool = False,
        s_dim: int = 16,
        master_selection_hidden_dim: int = 32,
        lambda_attn: float = 0.1,
        master_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim

        # Edge embedding modules
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        if hidden_irreps is None:
            hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
        self.hidden_irreps = hidden_irreps

        # Convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(
            TensorProductConvLayer(
                in_irreps=e3nn.o3.Irreps(f"{emb_dim}x0e"),
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=gate,
            )
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                TensorProductConvLayer(
                    in_irreps=hidden_irreps,
                    out_irreps=hidden_irreps,
                    sh_irreps=sh_irreps,
                    edge_feats_dim=self.radial_embedding.out_dim,
                    mlp_dim=mlp_dim,
                    aggr=aggr,
                    batch_norm=batch_norm,
                    gate=gate,
                )
            )

        # Wrap convolutions with HMP layers
        self.hmp_layers = nn.ModuleList()
        for conv in self.convs:
            layer = HMPLayer(
                conv=conv,
                s_dim=s_dim,
                master_selection_hidden_dim=master_selection_hidden_dim,
                lambda_attn=lambda_attn,
                master_rate=master_rate,
                spherical_harmonics=self.spherical_harmonics,
                radial_embedding=self.radial_embedding,
            )
            self.hmp_layers.append(layer)

        self.emb_in = nn.Embedding(in_dim, emb_dim)
        self.pool = global_add_pool
        self.pred = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim),
        )

    def update_tau(self, epoch: int, n_epochs: int) -> None:
        """Update the Gumbel-Softmax temperature for all HMP layers."""
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
        h = self.emb_in(batch.atoms)
        pos = batch.pos

        virtual_adjs = []
        masks = []

        for layer in self.hmp_layers:
            h, pos, A_virtual, m = layer(h, pos, batch.edge_index)
            virtual_adjs.append(A_virtual)
            masks.append(m)

        pooled_h = self.pool(h[:, :self.emb_dim], batch.batch)
        prediction = self.pred(pooled_h)
        return prediction

