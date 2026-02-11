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

# --- MODIFICATION START: 添加高斯基展开模块 ---
# 这个模块将替换掉有问题的 nn.Embedding
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
# --- MODIFICATION END ---

# SchNet 的高斯核 相对非常窄，它在距离编码中使用更局部化的分布和高分辨率的特征�?
class GaussianSmearing_schnet(torch.nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        # 计算高斯函数的中心点 (mu)
        offset = torch.linspace(start, stop, num_gaussians)
        # 计算高斯函数的宽�? (beta)，与中心点的间距有关
        self.coeff = -10.0  # SchNet 使用更窄的高斯核 
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

class SchNetInteraction_old(nn.Module):
    """
    Wrapper for SchNet InteractionBlock to be used in HMPLayer.
    --- MODIFICATION START: 为虚拟边添加基于距离的权重衰�? ---
    我们为此封装类添加了 distance_decay_d0 参数，并修改�? forward 方法�?
    使其可以接受一�? virtual_edge_mask，从而对虚拟边的消息传递强度进行衰减�?
    --- MODIFICATION END ---
    """
    # --- MODIFICATION START ---
    # 在构造函数中添加 distance_decay_d0
    def __init__(self, interaction_block, hidden_channels, num_gaussians, distance_decay_d0=None):
        super().__init__()
        self.interaction_block = interaction_block
        self.distance_decay_d0 = distance_decay_d0
    # --- MODIFICATION END ---

        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + num_gaussians, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    # --- MODIFICATION START ---
    # �? forward 方法中添�? virtual_edge_mask 参数
    def forward(self, h, pos, edge_index, virtual_edge_mask=None, size=None):
    # --- MODIFICATION END ---
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        
        # SchNet 的核心部分：将距离进行高斯基展开，得�? edge_attr
        edge_attr = self.interaction_block.distance_expansion(edge_weight)
        
        # --- MODIFICATION START: 避免原地操作 ---
        # 旧的、错误的方式:
        # if self.distance_decay_d0 is not None and virtual_edge_mask is not None:
        #     virtual_distances = edge_weight[virtual_edge_mask]
        #     decay_weights = torch.exp(-virtual_distances / self.distance_decay_d0)
        #     # In-place operation - THIS CAUSES THE ERROR
        #     edge_attr[virtual_edge_mask] = edge_attr[virtual_edge_mask] * decay_weights.unsqueeze(-1)

        # 新的、正确的方式:
        if self.distance_decay_d0 is not None and virtual_edge_mask is not None:
            # 1. 创建一个与边的数量相同的权重张量，默认值全部为 1.0
            full_decay_weights = torch.ones_like(edge_weight)
            
            # 2. 仅计算虚拟边的衰减因�?
            virtual_distances = edge_weight[virtual_edge_mask]
            #decay_factors = torch.exp(-virtual_distances / self.distance_decay_d0)
            #decay_factors = 1.0 / (1.0 + virtual_distances**2)
            """
            # compute/keep a running maximum of virtual distances (safe, out-of-gradient)
            if virtual_distances.numel() > 0:
                # detach and move to cpu to store as a plain float
                current_max = float(virtual_distances.max().detach().cpu().item())
                # initialize or update running maximum
                if not hasattr(self, "max_virtual_distances") or self.max_virtual_distances is None:
                    self.max_virtual_distances = current_max if current_max > 0.0 else 1e-6
                else:
                    self.max_virtual_distances = max(self.max_virtual_distances, current_max)
            else:
                # ensure a sensible default exists
                if not hasattr(self, "max_virtual_distances") or self.max_virtual_distances is None:
                    self.max_virtual_distances = 1e-6

            # guard against zero to avoid division-by-zero later
            if self.max_virtual_distances == 0.0:
                self.max_virtual_distances = 1e-6
            #print("--- Max virtual distances (running)---: ", self.max_virtual_distances)
            """
            #decay_factors = 0.5 * (torch.cos(virtual_distances * PI / self.max_virtual_distances) + 1.0)

            # �?/目标节点特征和边特征
            h_i = h[row]   # 对应每条边的源节点特�?
            h_j = h[col]   # 对应每条边的目标节点特征
            d_feat = edge_attr

            score = self.attn_mlp(torch.cat([h_i, h_j, d_feat], dim=-1)).squeeze(-1)

            # 在每个源节点上仅对虚拟边�? softmax（只在虚拟边的子集上归一化）
            decay_factors = torch.zeros_like(score)
            if virtual_edge_mask.any():
                score_virtual = score[virtual_edge_mask]
                row_virtual = row[virtual_edge_mask]
                try:
                    decay_virtual = scatter_softmax(score_virtual, row_virtual)
                except Exception:
                    # 备选：�? PyTorch 实现（较慢）
                    decay_virtual = torch.zeros_like(score_virtual)
                    unique_rows = torch.unique(row_virtual)
                    for r in unique_rows:
                        mask_r = row_virtual == r
                        decay_virtual[mask_r] = torch.softmax(score_virtual[mask_r], dim=0)
                decay_factors[virtual_edge_mask] = decay_virtual
                #print("--Max decay factor (virtual edges)--: ", decay_virtual.max().item())
            # 非虚拟边保持 decay_factors �? 0（随�? full_decay_weights 上对应位置仍�? 1�?
            
            # 3. 将衰减因子填充到权重张量的对应位�?
            #    这一步本身是原地操作，但它操作的是一个新创建的�?
            #    不在关键梯度路径上的张量 full_decay_weights，所以是安全的�?
            #print("--- Decay Weights Debug Info ---")
            #print("decay_virtual.shape: ", decay_virtual.shape)
            #print("full_decay_weights.shape: ", full_decay_weights.shape)
            #print("virtual_edge_mask.sum(): ", virtual_edge_mask.sum().item())
            full_decay_weights[virtual_edge_mask] = decay_factors[virtual_edge_mask]
            
            # 4. 用权重张量创建一个全新的 edge_attr，而不是修改旧的�?
            #    通过广播机制 (unsqueeze) 进行逐元素相乘�?
            #    这是一个非原地操作 (out-of-place)，会创建一个新张量�?
            #    并正确地将梯度传播到 edge_attr �? full_decay_weights�?
            edge_attr = edge_attr * full_decay_weights.unsqueeze(-1)
        # --- MODIFICATION END ---
            
        h_update = self.interaction_block(h, edge_index, edge_weight, edge_attr)
        return h_update, pos # Return pos unchanged as SchNet is invariant

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
        # 记录 induced edges �? virtual edges 的数�?
        num_induced_edges = edge_index_induced.shape[1]
        num_virtual_edges = edge_index_virtual.shape[1]
        
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
                 master_selection_hidden_dim=32, lambda_attn=0.1, master_rate=0.25):
        super().__init__()
        self.master_rate = master_rate
        self.emb_in = nn.Embedding(num_embeddings, hidden_channels)
        
        # SchNet specific setup
        # --- MODIFICATION START: 使用正确的高斯基展开 ---
        self.num_gaussians = num_gaussians # 定义高斯基的数量
        self.cutoff = cutoff               # 定义截断半径，应与数据集预处理时一�?
        
        # 将有问题�? nn.Embedding 替换为我们定义的 GaussianSmearing 模块
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)
        #self.distance_expansion = GaussianSmearing_schnet(0.0, self.cutoff, self.num_gaussians)
        # --- MODIFICATION END ---

        self.hmp_layers = nn.ModuleList()
        for _ in range(num_layers):
            # --- MODIFICATION START: InteractionBlock �? num_gaussians 参数需要正确设�? ---
            # InteractionBlock 接收的是高斯展开后的特征维度，即 num_gaussians
            interaction = InteractionBlock(hidden_channels=hidden_channels, num_gaussians=self.num_gaussians,
                                           num_filters=num_filters, cutoff=self.cutoff)
            # --- MODIFICATION END ---
            
            # 这行代码的逻辑保持不变，但现在它传递的是一个正确的模块
            interaction.distance_expansion = self.distance_expansion
            
            schnet_layer = SchNetInteraction(interaction, hidden_channels=hidden_channels, num_gaussians=self.num_gaussians)

            hmp_layer = HMPLayer(
                backbone_layer=schnet_layer, h_dim=hidden_channels, s_dim=s_dim,
                master_selection_hidden_dim=master_selection_hidden_dim,
                lambda_attn=lambda_attn, master_rate=self.master_rate)
            self.hmp_layers.append(hmp_layer)

        self.pool = global_add_pool
        # --- MODIFICATION START: SchNet 的输出层通常有特殊结�? ---
        # 原始�? SchNet 输出层结�?
        self.pred = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(), # SchNet 使用 SiLU (�? swish) 激活函�?
            nn.Linear(hidden_channels // 2, out_dim)
        )
        # --- MODIFICATION END ---
    
    # ... (forward �? update_tau 方法无需改动) ...```

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
        atoms = getattr(batch, "atoms", None)
        if atoms is None:
            atoms = getattr(batch, "z", None) # 在QM9/MD17中，原子序数通常�? 'z' 属性里
        if atoms is None:
            atoms = getattr(batch, "atomic_number", None)
        if atoms is None:
            raise AttributeError("No atomic numbers tensor found. Expected 'batch.z' or 'batch.atomic_number'")
        
        # SchNet/DimeNet 等模型通常�?1开始索引原子，但Embedding层需要从0开�?
        # 假设这里的原子序数已经是类别索引�?
        h = self.emb_in(atoms.long())
        
        pos = batch.pos

        for layer in self.hmp_layers:
            h, pos, A_virtual, m = layer(h, pos, batch.edge_index, batch.batch)

        pooled_h = self.pool(h, batch.batch)
        prediction = self.pred(pooled_h)
        return prediction