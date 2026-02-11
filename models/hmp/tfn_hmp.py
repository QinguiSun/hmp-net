import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_scatter import scatter_softmax # 必须引入
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
        radial_dim: int = None, # 新增参数：径向嵌入维度
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
        
        # --- MODIFICATION START: 定义 Attention MLP ---
        # 如果未显式传入 radial_dim，尝试从 radial_embedding 获取
        if radial_dim is None:
            radial_dim = radial_embedding.out_dim
            
        self.attn_mlp = nn.Sequential(
            # 输入: Source_Scalar + Target_Scalar + Edge_Radial_Embedding
            nn.Linear(2 * s_dim + radial_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, 1)
        )
        # --- MODIFICATION END ---

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        # 1. Local propagation
        vectors = pos[edge_index[0]] - pos[edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        h_update = self.conv(h, edge_index, edge_sh, edge_feats)
        h_local = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))

        # 2. Invariant topology learning
        h_scalar = h_local[:, :self.s_dim]
        m, _ = self.master_selection(h_scalar)
        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()
        
        device = h.device # 缓存 device

        if num_master_nodes <= 1:
            return h_local, pos, torch.zeros((0, 0), device=device), m

        master_indices = torch.where(master_nodes_mask)[0]
        
        # Induced Subgraph (真实边)
        edge_index_induced, _ = subgraph(
            master_indices, edge_index, relabel_nodes=True, num_nodes=h.size(0)
        )

        # --- 开始修改 ---
        # 检查子图中是否存在边
        if edge_index_induced.numel() > 0:
            # 如果有边，正常转换为邻接矩阵
            adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        else:
            # 如果没有边，创建一个全零的邻接矩阵
            # 确保它在正确的设备上
            adj_induced = torch.zeros((num_master_nodes, num_master_nodes), device=h.device)
        # --- 结束修改 ---
        
        

        h_master = h_local[master_nodes_mask]
        pos_master = pos[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]

        # 3. Virtual edge generation
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)
        edge_reindex_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_virtual, _ = remove_self_loops(edge_reindex_virtual)
        edge_index_virtual = to_undirected(edge_index_virtual, num_nodes=h_master.size(0))
        
        # 记录真实边和虚拟边的数量，用于后续切片
        num_induced_edges = edge_index_induced.size(1)
        num_virtual_edges = edge_index_virtual.size(1)
        
        # 合并边索引：[Induced, Virtual]
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        
        # 安全性检查
        M = h_master.size(0)
        E = edge_index_master.size(1) if edge_index_master.numel() > 0 else 0
        
        # 简单的连通性断言
        if M > 1 and E == 0:
             # 如果没有边，直接返回（避免后续计算报错）
             # 实际上这种情况很少见，除非 virtual generation 阈值极高
             return h_local, pos, A_virtual, m

        # 计算 Master 层级的几何特征
        vectors_master = pos_master[edge_index_master[0]] - pos_master[edge_index_master[1]]
        lengths_master = torch.linalg.norm(vectors_master, dim=-1, keepdim=True)
        edge_sh_master = self.spherical_harmonics(vectors_master)
        edge_feats_master = self.radial_embedding(lengths_master)

        # --- MODIFICATION START: Distance Decay Weights 计算 ---
        
        # 初始化权重为 1.0 (针对真实边)
        full_weights = torch.ones(E, dtype=torch.float, device=device)
        
        if num_virtual_edges > 0:
            # 利用切片，只提取虚拟边的数据进行计算
            # 虚拟边位于 edge_feats_master 的后半部分 [num_induced_edges:]
            
            # 1. 获取虚拟边的 Source 和 Target 节点索引
            # edge_index_master: [2, E], 其中 [0] 是 src, [1] 是 dst
            src_idx_virt = edge_index_master[0, num_induced_edges:]
            dst_idx_virt = edge_index_master[1, num_induced_edges:]
            
            # 2. 获取节点标量特征
            h_src = h_master_scalar[src_idx_virt]
            h_dst = h_master_scalar[dst_idx_virt]
            
            # 3. 获取边的径向特征 (Radial Embedding)
            d_feat_virt = edge_feats_master[num_induced_edges:]
            
            # 4. MLP 计算 Score
            # Input: [Num_Virt, s_dim*2 + radial_dim]
            attn_input = torch.cat([h_src, h_dst, d_feat_virt], dim=-1)
            scores = self.attn_mlp(attn_input).squeeze(-1) # [Num_Virt]
            
            # 5. Softmax 归一化 (对目标节点 dst_idx 进行归一化)
            decay_virtual = scatter_softmax(scores, dst_idx_virt)
            
            # 6. 赋值回总权重
            full_weights[num_induced_edges:] = decay_virtual

        # 应用权重到 edge_feats_master
        # edge_feats_master: [E, radial_dim]
        # full_weights: [E] -> [E, 1]
        edge_feats_master = edge_feats_master * full_weights.unsqueeze(-1)
        
        # --- MODIFICATION END ---

        # 4. Hierarchical update
        if self.conv.in_irreps == self.conv.out_irreps:
            # 使用加权后的 edge_feats_master 进行卷积
            h_master_update = self.conv(h_master, edge_index_master, edge_sh_master, edge_feats_master)
            h_hierarchical = h_master_update + F.pad(h_master, (0, h_master_update.shape[-1] - h_master.shape[-1]))
        else:
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
        hidden_irreps: Optional[e3nn.o3.Irreps] = None,
        mlp_dim: int = 64,
        num_embeddings: int = 1,
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
        # First layer
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
        # Subsequent layers
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
                # --- MODIFICATION: 显式传入 radial_dim ---
                radial_dim=self.radial_embedding.out_dim 
            )
            self.hmp_layers.append(layer)

        self.emb_in = nn.Embedding(num_embeddings, emb_dim)
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