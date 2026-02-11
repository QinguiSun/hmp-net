# models/schnet_hmp_mlp.py
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SchNet
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.utils import remove_self_loops, to_undirected

from torch_geometric.nn.models.schnet import InteractionBlock
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        # 计算高斯函数的中心点 (mu)
        offset = torch.linspace(start, stop, num_gaussians)
        # 计算高斯函数的宽�? (beta)，与中心点的间距有关
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        # �? offset 注册为模型的缓冲�? (buffer)，这样它会被移动到正确的设备 (cpu/cuda)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # 输入 dist: [num_edges]
        # 1. 调整维度以进行广�?: dist.view(-1, 1) -> [num_edges, 1]
        # 2. 从每个距离中减去所有高斯中心点: [num_edges, 1] - [num_gaussians] -> [num_edges, num_gaussians]
        # 3. 计算高斯函数: exp(coeff * (dist - offset)^2)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class SchNetInteraction(nn.Module):
        # 在构造函数中添加 distance_decay_d0
    def __init__(self, interaction_block, hidden_channels, num_gaussians):
        super().__init__()
        self.interaction_block = interaction_block
    # --- MODIFICATION END ---

        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + num_gaussians, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, h, pos, edge_index, virtual_edge_mask=None, size=None):
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.interaction_block.distance_expansion(edge_weight)

        # 默认情况下，所有边的权重都�?1.0
        # 注意：这里的 full_decay_weights 是在局部坐标系中创建的
        full_decay_weights = torch.ones_like(edge_weight)

        # 仅当处理主节点子图时 (�? virtual_edge_mask 不是 None �?) 才计算注意力
        if virtual_edge_mask is not None:
            # 确保 virtual_edge_mask 中至少有一个 True，否则后续索引会出错
            if virtual_edge_mask.any():
                h_i = h[row]
                h_j = h[col]
                d_feat = edge_attr

                # 计算所有边的注意力分数
                score = self.attn_mlp(torch.cat([h_i, h_j, d_feat], dim=-1)).squeeze(-1)

                # 提取只属于虚拟边的分数和源节点索引
                score_virtual = score[virtual_edge_mask]
                row_virtual = row[virtual_edge_mask]

                # 在虚拟边子集上进行 scatter_softmax
                decay_virtual = scatter_softmax(score_virtual, row_virtual)
                #decay_virtual = score_virtual

                # --- 关键修正 ---
                # 将计算出的虚拟边权重，直接赋值给 full_decay_weights 中对应虚拟边的位置
                # full_decay_weights 和 virtual_edge_mask 都在同一个局部上下文中，尺寸匹配
                full_decay_weights[virtual_edge_mask] = decay_virtual

                # --- Debug 打印 ---
                #print("--- Decay Weights Debug Info (Master Subgraph) ---")
                #print(f"Total edges in subgraph: {edge_index.shape[1]}")
                #print(f"Virtual edges in subgraph: {virtual_edge_mask.sum().item()}")
                #print(f"full_decay_weights.shape: {full_decay_weights.shape}")
                #print(f"decay_virtual.shape: {decay_virtual.shape}")

        # 用最终的权重（真实边�?1，虚拟边为softmax结果）来调整边属�?
        edge_attr = edge_attr * full_decay_weights.unsqueeze(-1)

        h_update = self.interaction_block(h, edge_index, edge_weight, edge_attr)
        return h_update, pos

class HMPLayer(nn.Module):
    # --- MODIFICATION START ---
    # 在构造函数中接收 distance_decay_d0
    def __init__(self, backbone_layer, h_dim, s_dim, master_selection_hidden_dim, lambda_attn, master_rate, distance_decay_d0=None):
    # --- MODIFICATION END ---
        super().__init__()
        self.backbone_layer = backbone_layer
        self.s_dim = s_dim
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim, ratio=master_rate)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)

        # --- MODIFICATION START ---
        # �? d0 存储起来（虽然在此类中不直接使用，但在构�? backbone_layer 时已传入�?
        self.distance_decay_d0 = distance_decay_d0
        # --- MODIFICATION END ---
        self.attention = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim)
        )

    def forward(self, h, h_global, pos, edge_index, batch):
        num_nodes = h.size(0)
        h_update, _ = self.backbone_layer(h, pos, edge_index)
        #h_local = h + h_update
        h_local = h
        h_global = h_global  # 未使用，但保留以防将来扩展
        pos_local = pos


        h_scalar = h_global[:, :self.s_dim]
        m, _ = self.master_selection(h_scalar)
        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()

        if num_master_nodes <= 1:
            print(f'num_master_nodes: {num_master_nodes.item()}')
            print("Not enough master nodes selected; skipping hierarchical update.")
            edge_index_virtual_global = None
            return h_local, h_global, pos_local, torch.zeros((0, 0), device=h.device), m, edge_index_virtual_global

        master_indices = torch.where(master_nodes_mask)[0]
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)

        if edge_index_induced.numel() > 0:
            adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        else:
            adj_induced = torch.zeros((num_master_nodes, num_master_nodes), device=h.device)

        h_master = h_global[master_nodes_mask]
        pos_master = pos_local[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]

        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)

        edge_reindex_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_virtual, _ = remove_self_loops(edge_reindex_virtual)
        edge_index_virtual = to_undirected(edge_index_virtual, num_nodes=h_master.size(0))

        # --- MODIFICATION START: 创建虚拟边掩码并将其传递给 backbone_layer ---
        # 记录 induced edges �? virtual edges 的数�?
        num_induced_edges = edge_index_induced.shape[1]
        num_virtual_edges = edge_index_virtual.shape[1]

        master_indices = master_indices.to(device=edge_reindex_virtual.device, dtype=torch.long)
        edge_index_virtual_global = master_indices[edge_reindex_virtual]

        # �? induced edges �? virtual edges 合并
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)

        # 创建一个布尔掩码，用于标识哪些是虚拟边
        # �? num_induced_edges 个是 False (真实�?)，后 num_virtual_edges 个是 True (虚拟�?)
        virtual_edge_mask = torch.cat([
            torch.zeros(num_induced_edges, dtype=torch.bool, device=h.device),
            torch.ones(num_virtual_edges, dtype=torch.bool, device=h.device)
        ])

        # 在调�? backbone_layer 时，传入新的 virtual_edge_mask 参数
        h_master_update, _ = self.backbone_layer(h_master, pos_master, edge_index_master, virtual_edge_mask=virtual_edge_mask)
        # --- MODIFICATION END ---

        h_hierarchical = h_master + h_master_update

        h_hierarchical_expanded = torch.zeros_like(h_global)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical

        #m_expanded = m.unsqueeze(1)
        #h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded
        h_global = h_global + self.attention(torch.cat((h_global, h_hierarchical_expanded), dim=-1)) * h_hierarchical_expanded
        #h_update, _ = self.backbone_layer(h_final, pos, edge_index)
        h_local = h + h_update

        return h_local, h_global, pos_local, A_virtual, m, edge_index_virtual_global


class HMP_SchNetModel(torch.nn.Module):
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

        s_dim: int = 16,
        master_selection_hidden_dim: int = 32,
        lambda_attn: float = 0.1,
        master_rate: float = 0.2,
    ):
        super().__init__()
        self._cutoff = float(cutoff)
        self._max_num_neighbors = int(max_num_neighbors)
        self._master_rate = master_rate
        self._num_gaussians = num_gaussians


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

        self.distance_expansion = GaussianSmearing(0.0, self._cutoff, self._num_gaussians)

        self.hmp_layers = nn.ModuleList()
        for _ in range(num_layers):
            # --- MODIFICATION START: InteractionBlock �? num_gaussians 参数需要正确设�? ---
            # InteractionBlock 接收的是高斯展开后的特征维度，即 num_gaussians
            interaction = InteractionBlock(hidden_channels=hidden_channels, num_gaussians=self._num_gaussians,
                                           num_filters=num_filters, cutoff=self._cutoff)
            # --- MODIFICATION END ---

            # 这行代码的逻辑保持不变，但现在它传递的是一个正确的模块
            interaction.distance_expansion = self.distance_expansion

            schnet_layer = SchNetInteraction(interaction, hidden_channels=hidden_channels, num_gaussians=self._num_gaussians)

            hmp_layer = HMPLayer(
                backbone_layer=schnet_layer, h_dim=hidden_channels, s_dim=s_dim,
                master_selection_hidden_dim=master_selection_hidden_dim,
                lambda_attn=lambda_attn, master_rate=self._master_rate)
            self.hmp_layers.append(hmp_layer)

        self.pool = global_add_pool
        # --- MODIFICATION START: SchNet 的输出层通常有特殊结�? ---
        # 原始�? SchNet 输出层结�?
        self.pred1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(), # SchNet 使用 SiLU (�? swish) 激活函�?
            nn.Linear(hidden_channels // 2, out_dim)
        )
        self.pred2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_dim)
        )

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
        #r = torch.cdist(pos_d, pos_d, p=2)

        # robust differentiable pairwise distance (supports higher-order derivatives)
        x = pos_d  # (B, Nmax, 3)
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)         # (B, Nmax, 1)
        r2 = x_norm + x_norm.transpose(1, 2) - 2.0 * (x @ x.transpose(1, 2))  # (B, Nmax, Nmax)
        r2 = torch.clamp(r2, min=1e-12)
        r = torch.sqrt(r2)  # (B, Nmax, Nmax)

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
        h_global = h.clone()

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

        pos = batch.pos

        last_layer_virtual_edges = None


        for i, layer in enumerate(self.hmp_layers):
            h, h_global, pos, A_virtual, m, edge_index_virtual_global = layer(h, h_global, pos, edge_index, batch.batch)

            if i == len(self.hmp_layers) - 1:
                last_layer_virtual_edges = edge_index_virtual_global

        pooled_h = self.pool(h, batch.batch)
        pooled_h_global = self.pool(h_global, batch.batch)
        prediction1 = self.pred1(pooled_h)
        prediction2 = self.pred2(pooled_h_global)
        U_sr = prediction1 + prediction2   # (B,1)

        if not self.use_long_range:
            if return_aux:
                aux = {"n_edges": torch.tensor(edge_index.size(1), device=U_sr.device)}
                return U_sr, aux
            return U_sr

        # latent charges (no supervision)
        q_tilde = self.q_head(h).view(-1)
        q = self._neutralize_per_graph(q_tilde, batch.batch, q_tot=q_tot)

        # long-range Coulomb energy
        U_coul, lambda_lr, q_reg, a, kappa = self._coulomb_energy_dense(batch.pos, q, batch.batch, pooled_h_global)
        U = U_sr + lambda_lr * U_coul.squeeze(-1)

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
