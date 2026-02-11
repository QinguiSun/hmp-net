# models/schnet.py
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SchNet
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import radius_graph


class SchNetModel(SchNet):
    """
    SchNet (short-range) + latent-charge Coulomb head (long-range).
    Trained with Energy + Forces only (no partial-charge supervision).
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_embeddings: int = 100,
        out_dim: int = 1,
        num_filters: int = 128,
        num_layers: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        pool: str = "sum",
        use_long_range: bool = True,
        learn_screening: bool = True,
        init_softcore_a: float = 0.2,
        init_kappa: float = 0.0,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=pool,
        )
        self._cutoff = float(cutoff)
        self._max_num_neighbors = int(max_num_neighbors)


        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_channels)
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        self.lin2 = nn.Linear(hidden_channels // 2, out_dim)

        self.use_long_range = use_long_range
        if use_long_range:
            self.q_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1),
            )
            self.lr_gate = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
            self.a_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
            self.register_buffer("_a_bias", torch.tensor(float(init_softcore_a)))

            self.learn_screening = learn_screening
            if learn_screening:
                self.kappa_head = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_channels // 2, 1),
                )
                self.register_buffer("_kappa_bias", torch.tensor(float(init_kappa)))

            self.coulomb_scale = nn.Parameter(torch.tensor(1.0))

    def _get_atoms(self, batch) -> torch.Tensor:
        atoms = getattr(batch, "atoms", None)
        if atoms is None:
            atoms = getattr(batch, "z", None)
        if atoms is None:
            atoms = getattr(batch, "atomic_numbers", None)
        if atoms is None:
            raise AttributeError("Expected atomic numbers in batch.atoms / batch.z / batch.atomic_numbers")
        return atoms.long()

    @staticmethod
    def _neutralize_per_graph(q: torch.Tensor, batch_idx: torch.Tensor, q_tot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enforce sum_i q_i = Q_tot per graph.
        q: (N,)
        batch_idx: (N,)
        q_tot: (B,) or (B,1); default zeros
        """
        B = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
        if q_tot is None:
            q_tot = q.new_zeros(B)
        q_tot = q_tot.view(-1)

        q_sum = global_add_pool(q.view(-1, 1), batch_idx).view(-1)          # (B,)
        ones = torch.ones_like(q).view(-1, 1)
        n_atoms = global_add_pool(ones, batch_idx).view(-1).clamp_min(1.0)  # (B,)

        q_centered = q - (q_sum / n_atoms)[batch_idx]
        q_projected = q_centered + (q_tot / n_atoms)[batch_idx]
        return q_projected

    def _coulomb_energy_dense(
        self,
        pos: torch.Tensor,
        q: torch.Tensor,
        batch_idx: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          U_coul: (B,1)
          lambda_lr: (B,1)
          q_reg: (B,1)
          a: (B,1)
          kappa: (B,1)
        """
        lambda_lr = torch.sigmoid(self.lr_gate(g))  # (B,1)
        a = F.softplus(self.a_head(g) + self._a_bias) + 1e-6  # (B,1)

        if self.learn_screening:
            kappa = F.softplus(self.kappa_head(g) + self._kappa_bias)  # (B,1)
        else:
            kappa = g.new_zeros((g.size(0), 1))

        pos_d, mask = to_dense_batch(pos, batch_idx)
        q_d, _ = to_dense_batch(q, batch_idx)

        B, Nmax, _ = pos_d.shape
        r = torch.cdist(pos_d, pos_d, p=2)

        a_ = a.view(B, 1, 1)
        k_ = kappa.view(B, 1, 1)
        kernel = torch.exp(-k_ * r) * torch.rsqrt(r * r + a_ * a_)

        m = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(Nmax, device=pos.device, dtype=torch.bool).unsqueeze(0)
        m = m & (~eye)

        qq = q_d.unsqueeze(2) * q_d.unsqueeze(1)
        pair = qq * kernel
        U_coul = 0.5 * (pair * m).sum(dim=(1, 2), keepdim=True)

        q_reg = global_mean_pool((q * q).view(-1, 1), batch_idx)
        return self.coulomb_scale * U_coul, lambda_lr, q_reg, a, kappa

    def forward(self, batch, q_tot: Optional[torch.Tensor] = None, return_aux: bool = False):
        atoms = self._get_atoms(batch)
        h = self.embedding(atoms)

        # Build edge_index on the fly if missing
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            edge_index = radius_graph(
                batch.pos,
                r=self._cutoff,
                batch=batch.batch,
                max_num_neighbors=self._max_num_neighbors,
                loop=False,
            )


        # SchNet interactions (short-range)
        row, col = edge_index
        edge_weight = (batch.pos[row] - batch.pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # short-range energy
        g = self.pool(h, batch.batch)
        U_sr = self.lin1(g)
        U_sr = self.act(U_sr)
        U_sr = self.lin2(U_sr)  # (B,1)

        if not self.use_long_range:
            if return_aux:
                aux = {"n_edges": torch.tensor(edge_index.size(1), device=U_sr.device)}
                return U_sr, aux
            return U_sr

        # latent charges (no supervision)
        q_tilde = self.q_head(h).view(-1)
        q = self._neutralize_per_graph(q_tilde, batch.batch, q_tot=q_tot)

        # long-range Coulomb energy
        U_coul, lambda_lr, q_reg, a, kappa = self._coulomb_energy_dense(batch.pos, q, batch.batch, g)
        U = U_sr + lambda_lr * U_coul

        if return_aux:
            # debug: check charge conservation per graph
            q_sum = global_add_pool(q.view(-1, 1), batch.batch).view(-1)  # (B,)
            aux = {
                "U_sr": U_sr.detach(),
                "U_coul": U_coul.detach(),
                "lambda_lr": lambda_lr.detach(),
                "q_reg": q_reg.detach(),
                "a": a.detach(),
                "kappa": kappa.detach(),
                "n_edges": torch.tensor(edge_index.size(1), device=U_sr.device),
                "q_sum_abs_max": q_sum.abs().max().detach(),
                "lambda_lr_mean": lambda_lr.mean().detach(),
            }
            return U, aux

        return U
