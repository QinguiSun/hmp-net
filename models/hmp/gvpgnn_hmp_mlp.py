import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
from torch.nn import functional as F

from models.layers.gvp_layer import GVPConvLayer
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration
from torch_scatter import scatter_softmax

class HMPLayer(nn.Module):
    """
    A Hierarchical Message Passing (HMP) layer that wraps a backbone GNN layer.
    """
    def __init__(self, backbone_layer, h_dim, s_dim, master_selection_hidden_dim, lambda_attn, master_rate):
        super().__init__()
        self.backbone_layer = backbone_layer
        self.s_dim = h_dim[0]
        self.v_dim = h_dim[1]
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim, ratio=master_rate)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)
        
        # Distance Decay Weights MLP
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

        device = h[0].device 
        
        if num_master_nodes <= 1:
            return h_local, pos_local, torch.zeros((0, 0), device=device), m

        master_indices = torch.where(master_nodes_mask)[0]

        # Create induced subgraph for master nodes
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        
        # --- FIX START: 处理空的 induced subgraph ---
        if edge_index_induced.numel() > 0:
            adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        else:
            # 如果没有边，创建一个全零的邻接矩阵
            adj_induced = torch.zeros((num_master_nodes, num_master_nodes), device=device)
        # --- FIX END ---

        h_master = (h_local[0][master_nodes_mask], h_local[1][master_nodes_mask])
        pos_master = pos_local[master_nodes_mask]
        h_master_scalar = h_master[0][:, :self.s_dim]

        # Generate virtual edges
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)

        # 3. Hierarchical Propagation
        edge_index_virtual, _ = dense_to_sparse(A_virtual)
        
        num_induced_edges = edge_index_induced.size(1)
        num_virtual_edges = edge_index_virtual.size(1)
        
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        edge_attr_master = self._edge_features(pos_master, edge_index_master)
        
        # Distance Decay Weights Optimization
        full_decay_weights = torch.ones(edge_index_master.size(1), dtype=torch.float, device=device)
        
        if num_virtual_edges > 0:
            src_idx_virt = edge_index_virtual[0]
            dst_idx_virt = edge_index_virtual[1]
            
            h_src = h_master[0][src_idx_virt] 
            h_dst = h_master[0][dst_idx_virt] 
            
            # 使用切片获取虚拟边的特征，即使 num_induced_edges 为 0 也能正常工作
            d_feat_virt = edge_attr_master[0][num_induced_edges:]
            
            attn_input = torch.cat([h_src, h_dst, d_feat_virt], dim=-1)
            scores = self.attn_mlp(attn_input).squeeze(-1)
            
            decay_virtual = scatter_softmax(scores, dst_idx_virt)
            full_decay_weights[num_induced_edges:] = decay_virtual

        # Apply weights
        w = full_decay_weights.unsqueeze(-1)
        s_attr, v_attr = edge_attr_master
        s_weighted = s_attr * w
        v_weighted = v_attr * w.unsqueeze(-1)
        edge_attr_master_weighted = (s_weighted, v_weighted)

        h_master_update = self.backbone_layer(
            h_master, edge_index_master, edge_attr_master_weighted
        )
        
        h_hierarchical = (h_master[0] + h_master_update[0], h_master[1] + h_master_update[1])

        # 4. Feature Aggregation
        h_hierarchical_expanded = (torch.zeros_like(h_local[0]), torch.zeros_like(h_local[1]))
        h_hierarchical_expanded[0][master_nodes_mask] = h_hierarchical[0]
        h_hierarchical_expanded[1][master_nodes_mask] = h_hierarchical[1]

        m_expanded = m.unsqueeze(1)
        h_final_scalar = (1 - m_expanded) * h_local[0] + m_expanded * h_hierarchical_expanded[0]
        h_final_vector = (1 - m_expanded).unsqueeze(-1) * h_local[1] + m_expanded.unsqueeze(-1) * h_hierarchical_expanded[1]
        h_final = (h_final_scalar, h_final_vector)
        pos_final = pos_local

        return h_final, pos_final, A_virtual, m


class HMP_GVPGNNModel(torch.nn.Module):
    # ... (保持 HMP_GVPGNNModel 类代码不变) ...
    # 只需要确保上面的 HMPLayer 被正确导入和使用
    def __init__(
        self,
        num_layers: int = 5,
        s_dim: int = 32,
        v_dim: int = 16,
        num_embeddings: int = 1,
        out_dim: int = 1,
        master_selection_hidden_dim: int = 32,
        lambda_attn: float = 0.1,
        master_rate: float = 0.25,
    ):
        super().__init__()
        self.master_rate = master_rate
        self.s_dim = s_dim
        self.v_dim = v_dim

        self.emb_in_s = torch.nn.Embedding(num_embeddings, s_dim)
        self.emb_in_v = torch.nn.Embedding(num_embeddings, v_dim * 3)

        self.hmp_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            gvp_layer = GVPConvLayer((s_dim, v_dim), (s_dim, v_dim), activations=(F.relu, None), residual=True, vector_gate=True)
            hmp_layer = HMPLayer(
                backbone_layer=gvp_layer,
                h_dim=(s_dim, v_dim),
                s_dim=s_dim,
                master_selection_hidden_dim=master_selection_hidden_dim,
                lambda_attn=lambda_attn,
                master_rate=self.master_rate
            )
            self.hmp_layers.append(hmp_layer)

        self.pool = global_add_pool
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(s_dim, s_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(s_dim, out_dim)
        )

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
        h_scalar = self.emb_in_s(batch.atoms)
        h_vector = self.emb_in_v(batch.atoms).view(-1, self.v_dim, 3)
        h = (h_scalar, h_vector)
        pos = batch.pos

        virtual_adjs = []
        masks = []

        for layer in self.hmp_layers:
            h, pos, A_virtual, m = layer(h, pos, batch.edge_index, batch.batch)
            virtual_adjs.append(A_virtual)
            masks.append(m)

        pooled_h = self.pool(h[0], batch.batch)
        prediction = self.pred(pooled_h)
        return prediction