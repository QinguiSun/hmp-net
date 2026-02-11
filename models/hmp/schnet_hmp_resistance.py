# schnet_hmp.py
import torch
from torch import nn
from torch_geometric.nn.models.schnet import InteractionBlock
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool

from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration
from math import pi as PI
from torch_scatter import scatter_softmax


import networkx as nx
import numpy as np

def calculate_resistance_curvature(G):
    """
    Calculates a version of resistance curvature for a graph G.
    This is a simplified example and may need adjustments based on the specific formula.
    """
    # Initialize a dictionary to store curvature for each node
    node_curvatures = {}

    # Calculate effective resistance for all pairs of nodes
    # This is a computationally intensive step
    # (G must be connected for resistance to be well-defined)
    effective_resistance = nx.resistance_distance(G)

    # Loop through each node to calculate its curvature
    for v in G.nodes():
        # Example: Using a node-based curvature formula from literature [11]
        # We assume each edge has a resistance of 1 for simplicity in this example
        # In a real-world case, this would be specified or computed.
        sum_of_edge_resistances = 0
        for neighbor in G.neighbors(v):
            # Use the effective resistance between v and its neighbor.
            # In the paper, r_uv(c) is a specific resistance, so this may be different.
            # Here we use the pairwise resistance distance.
            sum_of_edge_resistances += effective_resistance[v][neighbor]

        # Apply the formula: p_v(c) = 1 - (1/2) * sum(r_uv(c))
        # For simplicity, we divide the sum by 2 and use effective resistance
        curvature = 1 - 0.5 * sum_of_edge_resistances
        node_curvatures[v] = curvature


# --- MODIFICATION START: 添加高斯基展开模块 ---
# 这个模块将替换掉有问题的 nn.Embedding
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        # 计算高斯函数的中心点 (mu)
        offset = torch.linspace(start, stop, num_gaussians)
        # 计算高斯函数的宽度 (beta)，与中心点的间距有关
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2 
        # 将 offset 注册为模型的缓冲区 (buffer)，这样它会被移动到正确的设备 (cpu/cuda)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # 输入 dist: [num_edges]
        # 1. 调整维度以进行广播: dist.view(-1, 1) -> [num_edges, 1]
        # 2. 从每个距离中减去所有高斯中心点: [num_edges, 1] - [num_gaussians] -> [num_edges, num_gaussians]
        # 3. 计算高斯函数: exp(coeff * (dist - offset)^2)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
# --- MODIFICATION END ---

# SchNet 的高斯核 相对非常窄，它在距离编码中使用更局部化的分布和高分辨率的特征。
class GaussianSmearing_schnet(torch.nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        # 计算高斯函数的中心点 (mu)
        offset = torch.linspace(start, stop, num_gaussians)
        # 计算高斯函数的宽度 (beta)，与中心点的间距有关
        self.coeff = -10.0  # SchNet 使用更窄的高斯核 
        # 将 offset 注册为模型的缓冲区 (buffer)，这样它会被移动到正确的设备 (cpu/cuda)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # 输入 dist: [num_edges]
        # 1. 调整维度以进行广播: dist.view(-1, 1) -> [num_edges, 1]
        # 2. 从每个距离中减去所有高斯中心点: [num_edges, 1] - [num_gaussians] -> [num_edges, num_gaussians]
        # 3. 计算高斯函数: exp(coeff * (dist - offset)^2)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
class SchNetInteraction(nn.Module):
    # 在构造函数中添加 distance_decay_d0
    def __init__(self, interaction_block, hidden_channels, num_gaussians, distance_decay_d0=None):
        super().__init__()
        self.interaction_block = interaction_block
        self.distance_decay_d0 = distance_decay_d0

        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + num_gaussians, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, h, pos, edge_index, virtual_edge_mask=None, size=None):
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.interaction_block.distance_expansion(edge_weight)
        
        # 默认情况下，所有边的权重都是1.0
        # 注意：这里的 full_decay_weights 是在局部坐标系(subgraph)中创建的
        full_decay_weights = torch.ones_like(edge_weight)

        # 仅当处理主节点子图时 (即 virtual_edge_mask 不是 None 时) 才计算注意力
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
                
                # --------------------- resistance based cosin ----------------------
                # 1. 计算 link resistance curvature k_ij
                
                
                
                
                
                
                
                
                #ics = pos[row]  # 起点坐标
                #jcs = pos[col]  # 终点坐标
                #mid_points = (ics + jcs) / 2  # 边的中点坐标    
                #vec_ij = jcs - ics  # 边的向量
                #vec_ij_norm = vec_ij / (torch.norm(vec_ij, dim=-1, keepdim=True) + 1e-8)  # 归一化边向量    
                # 计算中点到原点的向量
                #mid_to_origin = mid_points - torch.zeros_like(mid_points)
                # 计算中点到原点向量在边向量上的投影长度
                #proj_lengths = torch.sum(mid_to_origin * vec_ij_norm, dim=-1, keepdim=True)
                # 计算投影点坐标
                #proj_points = proj_lengths * vec_ij_norm
                # 计算曲率向量
                #curvature_vecs = mid_points - proj_points
                # 计算曲率大小 k_ij
                #k_ij = torch.norm(curvature_vecs, dim=-1)   
                
                # 2. 计算 max_k_ij
                #max_k_ij = torch.max(k_ij) + 1e-8  # 防止除以零     
                # 归一化曲率
                #k_ij_normalized = k_ij / max_k_ij
                
                # 3. 计算基于阻抗的调整因子
                #resistance_adjustment = 0.5 * (1 + torch.cos(torch.clamp(k_ij_normalized / self.distance_decay_d0 * PI, max=PI)))
                #resistance_adjustment = 0.5 * (1 + torch.cos(torch.clamp(edge_weight / self.distance_decay_d0 * PI, max=PI)))
                #full_decay_weights[virtual_edge_mask] = decay_virtual * resistance_adjustment[virtual_edge_mask]
                # -------------------------------------------------------------------
   

        # 用最终的权重（真实边为1，虚拟边为softmax结果）来调整边属性
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
        # 将 d0 存储起来（虽然在此类中不直接使用，但在构造 backbone_layer 时已传入）
        self.distance_decay_d0 = distance_decay_d0
        # --- MODIFICATION END ---

    def forward(self, h, pos, edge_index, batch):
        num_nodes = h.size(0)
        h_update, _ = self.backbone_layer(h, pos, edge_index)
        h_local = h + h_update
        pos_local = pos

        h_scalar = h_local[:, :self.s_dim]
        m, _ = self.master_selection(h_scalar)
        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()

        if num_master_nodes <= 1:
            return h_local, pos_local, torch.zeros((0, 0), device=h.device), m

        master_indices = torch.where(master_nodes_mask)[0]
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)

        if edge_index_induced.numel() > 0:
            adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        else:
            adj_induced = torch.zeros((num_master_nodes, num_master_nodes), device=h.device)
        
        h_master = h_local[master_nodes_mask]
        pos_master = pos_local[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]

        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)
        edge_index_virtual, _ = dense_to_sparse(A_virtual)
        
        # --- MODIFICATION START: 创建虚拟边掩码并将其传递给 backbone_layer ---
        # 记录 induced edges 和 virtual edges 的数量
        num_induced_edges = edge_index_induced.shape[1]
        num_virtual_edges = edge_index_virtual.shape[1]
        
        # 将 induced edges 和 virtual edges 合并
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        
        # 创建一个布尔掩码，用于标识哪些是虚拟边
        # 前 num_induced_edges 个是 False (真实边)，后 num_virtual_edges 个是 True (虚拟边)
        virtual_edge_mask = torch.cat([
            torch.zeros(num_induced_edges, dtype=torch.bool, device=h.device),
            torch.ones(num_virtual_edges, dtype=torch.bool, device=h.device)
        ])
        
        # 在调用 backbone_layer 时，传入新的 virtual_edge_mask 参数
        h_master_update, _ = self.backbone_layer(h_master, pos_master, edge_index_master, virtual_edge_mask=virtual_edge_mask)
        # --- MODIFICATION END ---
        
        h_hierarchical = h_master + h_master_update
        
        h_hierarchical_expanded = torch.zeros_like(h_local)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical

        m_expanded = m.unsqueeze(1)
        h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded

        return h_final, pos_local, A_virtual, m
    

class HMP_SchNetModel(torch.nn.Module):
    # (参数列表保持不变)
    def __init__(self, num_layers=5, hidden_channels=128, num_embeddings=100,
                 out_dim=1, 
                 num_filters=1, num_gaussians=1, cutoff = 1, s_dim=16,
                 master_selection_hidden_dim=32, lambda_attn=0.1, master_rate=0.25, 
                 distance_decay_d0=5.0): # d0 的默认值设为 5.0 埃
        super().__init__()
        self.master_rate = master_rate
        self.emb_in = nn.Embedding(num_embeddings, hidden_channels)
        
        # SchNet specific setup
        # --- MODIFICATION START: 使用正确的高斯基展开 ---
        self.num_gaussians = num_gaussians # 定义高斯基的数量
        self.cutoff = cutoff               # 定义截断半径，应与数据集预处理时一致
        
        # 将有问题的 nn.Embedding 替换为我们定义的 GaussianSmearing 模块
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)
        #self.distance_expansion = GaussianSmearing_schnet(0.0, self.cutoff, self.num_gaussians)
        # --- MODIFICATION END ---

        self.hmp_layers = nn.ModuleList()
        for _ in range(num_layers):
            # --- MODIFICATION START: InteractionBlock 的 num_gaussians 参数需要正确设置 ---
            # InteractionBlock 接收的是高斯展开后的特征维度，即 num_gaussians
            interaction = InteractionBlock(hidden_channels=hidden_channels, num_gaussians=self.num_gaussians,
                                           num_filters=num_filters, cutoff=self.cutoff)
            # --- MODIFICATION END ---
            
            # 这行代码的逻辑保持不变，但现在它传递的是一个正确的模块
            interaction.distance_expansion = self.distance_expansion
            
            schnet_layer = SchNetInteraction(interaction, hidden_channels=hidden_channels, num_gaussians=self.num_gaussians, 
                                             distance_decay_d0=distance_decay_d0)

            hmp_layer = HMPLayer(
                backbone_layer=schnet_layer, h_dim=hidden_channels, s_dim=s_dim,
                master_selection_hidden_dim=master_selection_hidden_dim,
                lambda_attn=lambda_attn, master_rate=self.master_rate,
                distance_decay_d0=distance_decay_d0)
            self.hmp_layers.append(hmp_layer)

        self.pool = global_add_pool
        # --- MODIFICATION START: SchNet 的输出层通常有特殊结构 ---
        # 原始的 SchNet 输出层结构
        self.pred = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(), # SchNet 使用 SiLU (或 swish) 激活函数
            nn.Linear(hidden_channels // 2, out_dim)
        )
        # --- MODIFICATION END ---

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
        atoms = getattr(batch, "z", None) # 在QM9/MD17中，原子序数通常在 'z' 属性里
        if atoms is None:
            atoms = getattr(batch, "atomic_number", None)
        if atoms is None:
            raise AttributeError("No atomic numbers tensor found. Expected 'batch.z' or 'batch.atomic_number'")
        
        # SchNet/DimeNet 等模型通常从1开始索引原子，但Embedding层需要从0开始
        # 假设这里的原子序数已经是类别索引了
        h = self.emb_in(atoms.long())
        
        pos = batch.pos

        for layer in self.hmp_layers:
            h, pos, A_virtual, m = layer(h, pos, batch.edge_index, batch.batch)

        pooled_h = self.pool(h, batch.batch)
        prediction = self.pred(pooled_h)
        return prediction