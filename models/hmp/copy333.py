import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
from torch.nn import functional as F
from torch_scatter import scatter_softmax

# 假设这些是你项目中的引用，保持不变
from models.layers.gvp_layer import GVPConvLayer
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration

class HMPLayer(nn.Module):
    def __init__(self, backbone_layer, h_dim, s_dim, master_selection_hidden_dim, lambda_attn, master_rate):
        super().__init__()
        self.backbone_layer = backbone_layer
        self.s_dim = h_dim[0]
        self.v_dim = h_dim[1]
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim, ratio=master_rate)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)
        
        # --- FIX: 定义 Attention MLP ---
        # 输入维度: Source(s_dim) + Target(s_dim) + Distance_Embedding(s_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(3 * self.s_dim, self.s_dim),
            nn.ReLU(),
            nn.Linear(self.s_dim, 1)
        )

    def _edge_features(self, pos, edge_index):
        vectors = pos[edge_index[0]] - pos[edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        directions = torch.nan_to_num(vectors / lengths)

        s = lengths.repeat(1, self.s_dim)
        v = directions.unsqueeze(1).repeat(1, self.v_dim, 1)
        return s, v

    def forward(self, h, pos, edge_index, batch):
        # 1. Local Propagation
        num_nodes = h[0].size(0)
        edge_attr = self._edge_features(pos, edge_index)
        h_update = self.backbone_layer(h, edge_index, edge_attr)
        h_local = (h[0] + h_update[0], h[1] + h_update[1]) 
        pos_local = pos

        # 2. Invariant Topology Learning
        h_scalar = h_local[0][:, :self.s_dim]
        m, _ = self.master_selection(h_scalar) 

        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()

        # FIX: 使用 h[0].device 避免 tuple 报错
        device = h[0].device 

        if num_master_nodes <= 1:
            return h_local, pos_local, torch.zeros((0, 0), device=device), m

        master_indices = torch.where(master_nodes_mask)[0]

        # Induced Subgraph
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)

        h_master = (h_local[0][master_nodes_mask], h_local[1][master_nodes_mask])
        pos_master = pos_local[master_nodes_mask]
        h_master_scalar = h_master[0][:, :self.s_dim]

        # Generate virtual edges
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)

        # 3. Hierarchical Propagation
        edge_index_virtual, _ = dense_to_sparse(A_virtual)
        
        # 记录数量以便切片
        num_induced_edges = edge_index_induced.size(1)
        num_virtual_edges = edge_index_virtual.size(1)

        # 合并边索引
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        edge_attr_master = self._edge_features(pos_master, edge_index_master)
        
        # --- OPTIMIZATION START: 优化 Distance Decay Weights 计算 ---
        
        # 初始化权重为 1.0 (针对真实边)
        # full_decay_weights 形状: [Total_Edges]
        full_decay_weights = torch.ones(edge_index_master.size(1), dtype=torch.float, device=device)

        if num_virtual_edges > 0:
            # 仅提取虚拟边相关的数据进行计算，避免全图计算带来的开销
            # 虚拟边位于 edge_index_master 的后半部分
            
            # PyG edge_index 约定: [0]是源节点(Source/j), [1]是目标节点(Target/i)
            src_idx_virt = edge_index_virtual[0]
            dst_idx_virt = edge_index_virtual[1]
            
            h_src = h_master[0][src_idx_virt] # Source features
            h_dst = h_master[0][dst_idx_virt] # Target features
            
            # edge_attr_master 是 tuple (s, v)。我们需要 s 部分的距离特征。
            # s 的形状是 [Total_Edges, s_dim]。我们只取后半部分。
            d_feat_virt = edge_attr_master[0][num_induced_edges:]
            
            # 计算 Attention Score
            # Input: [Num_Virtual, 3 * s_dim] -> Output: [Num_Virtual, 1]
            attn_input = torch.cat([h_src, h_dst, d_feat_virt], dim=-1)
            scores = self.attn_mlp(attn_input).squeeze(-1)
            
            # Softmax 归一化
            # 关键修正：通常 Attention 是对汇聚到目标节点的信息进行归一化。
            # 所以应该 scatter on dst_idx (edge_index[1])。
            # 原始代码使用的是 row (edge_index[0])，那是对出度归一化，通常是不对的。
            decay_virtual = scatter_softmax(scores, dst_idx_virt)
            
            # 将计算好的权重赋值回总权重向量的后半部分
            full_decay_weights[num_induced_edges:] = decay_virtual

        # --- 应用权重到边特征 (处理 Tuple) ---
        # edge_attr_master 是 (s, v)
        # s: [E, s_dim], v: [E, v_dim, 3]
        # weights: [E] -> [E, 1]
        w = full_decay_weights.unsqueeze(-1)
        
        s_attr, v_attr = edge_attr_master
        
        # 对标量特征加权: [E, s_dim] * [E, 1]
        s_weighted = s_attr * w
        
        # 对向量特征加权: [E, v_dim, 3] * [E, 1, 1] (需要额外 unsqueeze)
        v_weighted = v_attr * w.unsqueeze(-1)
        
        edge_attr_master_weighted = (s_weighted, v_weighted)
        
        # --- OPTIMIZATION END ---

        h_master_update = self.backbone_layer(
            h_master, edge_index_master, edge_attr_master_weighted
        )
        
        h_hierarchical = (h_master[0] + h_master_update[0], h_master[1] + h_master_update[1])
        
        # 4. Feature Aggregation (保持不变)
        h_hierarchical_expanded = (torch.zeros_like(h_local[0]), torch.zeros_like(h_local[1]))
        h_hierarchical_expanded[0][master_nodes_mask] = h_hierarchical[0]
        h_hierarchical_expanded[1][master_nodes_mask] = h_hierarchical[1]

        m_expanded = m.unsqueeze(1)
        h_final_scalar = (1 - m_expanded) * h_local[0] + m_expanded * h_hierarchical_expanded[0]
        h_final_vector = (1 - m_expanded).unsqueeze(-1) * h_local[1] + m_expanded.unsqueeze(-1) * h_hierarchical_expanded[1]
        h_final = (h_final_scalar, h_final_vector)
        pos_final = pos_local

        return h_final, pos_final, A_virtual, m