# mace_hmp.py
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
import e3nn
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_scatter import scatter_softmax  # --- MODIFICATION: 引入 scatter_softmax ---

from models.mace import MACEModel
from models.mace_modules.irreps_tools import reshape_irreps
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration
from models.layers.tfn_layer import TensorProductConvLayer
from models.mace_modules.blocks import EquivariantProductBasisBlock

class HMP_MACELayer(nn.Module):
    """
    HMP Layer for the MACE architecture.
    """
    def __init__(self, local_conv, hierarchical_conv, reshape, prod, spherical_harmonics, radial_embedding, s_dim, master_selection_hidden_dim, lambda_attn):
        super().__init__()
        self.local_conv = local_conv
        self.hierarchical_conv = hierarchical_conv
        self.reshape = reshape
        self.prod = prod
        self.spherical_harmonics = spherical_harmonics
        self.radial_embedding = radial_embedding
        self.s_dim = s_dim
        
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)

        # --- MODIFICATION START: 定义 Attention MLP ---
        # 输入: Source_Scalar(s_dim) + Target_Scalar(s_dim) + Edge_Radial(radial_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * s_dim + radial_embedding.out_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, 1)
        )
        # --- MODIFICATION END ---

    def forward(self, h, pos, edge_index, edge_sh, edge_feats, batch):
        # 1. Local Propagation
        h_update = self.local_conv(h, edge_index, edge_sh, edge_feats)
        sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
        h_local = self.prod(self.reshape(h_update), sc, None)
        
        # 2. Invariant Topology Learning
        h_scalar = h_local[:, :self.s_dim]
        m, _ = self.master_selection(h_scalar)
        
        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()
        
        if num_master_nodes <= 1:
            empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=h.device)
            return h_local, torch.zeros((0, 0), device=h.device), m, empty_edge_index

        master_indices = torch.where(master_nodes_mask)[0]
        
        # Create induced subgraph for master nodes
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=h.size(0))

        if edge_index_induced.numel() > 0:
            adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        else:
            adj_induced = torch.zeros((num_master_nodes, num_master_nodes), device=h[0].device)

        h_master = h_local[master_nodes_mask]
        pos_master = pos[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]

        # Generate virtual edges
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)
        
        # 3. Hierarchical Propagation
        edge_reindex_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_virtual, _ = remove_self_loops(edge_reindex_virtual)
        edge_index_virtual = to_undirected(edge_index_virtual, num_nodes=h_master.size(0))
        
        # 记录真实边和虚拟边的数量，用于切片
        num_induced_edges = edge_index_induced.size(1)
        num_virtual_edges = edge_index_virtual.size(1)

        master_indices = master_indices.to(device=edge_reindex_virtual.device, dtype=torch.long)
        edge_index_virtual_global = master_indices[edge_reindex_virtual]
        
        # 合并边索引 [Induced, Virtual]
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        
        if edge_index_master.numel() > 0:
            assert edge_index_master.max() < h_master.size(0), \
                f"master edge index {edge_index_master.max().item()} >= num master {h_master.size(0)}"
        
        M = h_master.size(0)
        E = edge_index_master.size(1) if edge_index_master.numel() > 0 else 0
        
        assert (M <= 1) or (E >= 1), \
            f"No edges among master nodes: num_master={M}, num_edges={E}"
            
        deg = torch.bincount(edge_index_master.reshape(-1), minlength=M)
        isolated = (deg == 0).nonzero(as_tuple=False).view(-1)
        assert (M == 0) or (M == 1 and E == 0) or isolated.numel() == 0, \
            f"Isolated master nodes found (no incident edges): {isolated.tolist()} | num_master={M}, num_edges={E}"
        
        # Re-compute edge attributes for the master graph
        vectors_master = pos_master[edge_index_master[0]] - pos_master[edge_index_master[1]]
        lengths_master = torch.linalg.norm(vectors_master, dim=-1, keepdim=True)
        edge_sh_master = self.spherical_harmonics(vectors_master)
        edge_feats_master = self.radial_embedding(lengths_master)

        # --- MODIFICATION START: Distance Decay Weights (Attention) ---
        
        # 初始化权重为 1.0 (针对真实边)
        full_weights = torch.ones(E, dtype=torch.float, device=h.device)
        
        if num_virtual_edges > 0:
            # 仅提取虚拟边相关的数据进行计算 (利用切片 [num_induced_edges:])
            
            # 1. 获取虚拟边的 Source 和 Target 节点索引 (Master-Local 索引)
            # edge_index_master: [2, E], [0]=src, [1]=dst
            src_idx_virt = edge_index_master[0, num_induced_edges:]
            dst_idx_virt = edge_index_master[1, num_induced_edges:]
            
            # 2. 获取节点标量特征
            h_src = h_master_scalar[src_idx_virt]
            h_dst = h_master_scalar[dst_idx_virt]
            
            # 3. 获取边的径向特征
            d_feat_virt = edge_feats_master[num_induced_edges:]
            
            # 4. MLP 计算 Score
            # Input: [Num_Virt, s_dim*2 + radial_dim] -> Output: [Num_Virt, 1]
            attn_input = torch.cat([h_src, h_dst, d_feat_virt], dim=-1)
            scores = self.attn_mlp(attn_input).squeeze(-1)
            
            # 5. Softmax 归一化 (对目标节点 dst_idx 进行归一化)
            decay_virtual = scatter_softmax(scores, dst_idx_virt)
            
            # 6. 赋值回总权重
            full_weights[num_induced_edges:] = decay_virtual

        # 应用权重到 edge_feats_master
        # edge_feats_master: [E, radial_dim]
        # full_weights: [E] -> [E, 1]
        edge_feats_master = edge_feats_master * full_weights.unsqueeze(-1)
        
        # --- MODIFICATION END ---

        h_master_update = self.hierarchical_conv(h_master, edge_index_master, edge_sh_master, edge_feats_master)
        sc_master = F.pad(h_master, (0, h_master_update.shape[-1] - h_master.shape[-1]))
        h_hierarchical = self.prod(self.reshape(h_master_update), sc_master, None)
        
        # 4. Feature Aggregation
        h_hierarchical_expanded = torch.zeros_like(h_local)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical
        
        m_expanded = m.unsqueeze(1)
        h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded
        
        return h_final, A_virtual, m, edge_index_virtual_global


class HMP_MACEModel(MACEModel):
    """
    HMP-enhanced MACE model.
    """
    def __init__(self, 
                master_rate=0.25, 
                num_embeddings=1, 
                emb_dim=64,
                correlation=3, 
                residual=True,
                equivariant_pred=False, 
                out_dim=1,
                s_dim =0, 
                master_selection_hidden_dim=0, 
                lambda_attn=0, 
                s_dim_scale=1, 
                **kwargs):
        # call MACEModel constructor
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        
        self.emb_in = torch.nn.Embedding(num_embeddings, emb_dim)

        if getattr(self, "hidden_irreps", None) is None or self.hidden_irreps.dim == 0:
            sh_irreps = e3nn.o3.Irreps.spherical_harmonics(self.max_ell)
            self.hidden_irreps = (sh_irreps * self.emb_dim).sort()[0].simplify()
        self.master_rate = master_rate
        aggr = kwargs.get("aggr", "sum")
        
        sh_irreps = self.spherical_harmonics.irreps_out
        self.convs = torch.nn.ModuleList()
        self.convs.append(
                TensorProductConvLayer(
                    in_irreps=e3nn.o3.Irreps(f"{self.emb_dim}x0e"),
                    out_irreps=self.hidden_irreps,
                    sh_irreps=sh_irreps,
                    edge_feats_dim=self.radial_embedding.out_dim,
                    mlp_dim=self.mlp_dim,
                    aggr=aggr,
                    batch_norm=self.batch_norm,
                    gate=False,
                )
        )

        for _ in range(self.num_layers - 1):
            self.convs.append(
                TensorProductConvLayer(
                    in_irreps=self.hidden_irreps,
                    out_irreps=self.hidden_irreps,
                    sh_irreps=sh_irreps,
                    edge_feats_dim=self.radial_embedding.out_dim,
                    mlp_dim=self.mlp_dim,
                    aggr=aggr,
                    batch_norm=self.batch_norm,
                    gate=False,
                )
            )

        s_dim = 0
        for mul, ir in self.hidden_irreps:
            if ir.l == 0 and ir.p == 1:
                s_dim += mul
        s_dim *= s_dim_scale

        tensor_conv = TensorProductConvLayer(
            in_irreps=self.hidden_irreps,
            out_irreps=self.hidden_irreps,
            sh_irreps=self.spherical_harmonics.irreps_out,
            edge_feats_dim=self.radial_embedding.out_dim,
            mlp_dim=self.mlp_dim,
            aggr=aggr,
            batch_norm=self.batch_norm,
            gate=False,
        )
        
        self.reshapes = torch.nn.ModuleList([
            reshape_irreps(self.hidden_irreps)
        ])
        self.prods = torch.nn.ModuleList([
            EquivariantProductBasisBlock(
                node_feats_irreps=self.hidden_irreps,
                target_irreps=self.hidden_irreps,
                correlation=correlation,
                element_dependent=False,
                num_elements=num_embeddings,
                use_sc=residual
            )
        ])

        for _ in range(self.num_layers - 1):
            self.reshapes.append(reshape_irreps(self.hidden_irreps))
            self.prods.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=self.hidden_irreps,
                    target_irreps=self.hidden_irreps,
                    correlation=correlation,
                    element_dependent=False,
                    num_elements=num_embeddings,
                    use_sc=residual
                )
            )

        self.hmp_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            hmp_layer = HMP_MACELayer(
                local_conv=self.convs[i],
                hierarchical_conv=tensor_conv,
                reshape=self.reshapes[i],
                prod=self.prods[i],
                spherical_harmonics=self.spherical_harmonics,
                radial_embedding=self.radial_embedding,
                s_dim=s_dim,
                master_selection_hidden_dim=s_dim,
                lambda_attn=0.1 
            )
            self.hmp_layers.append(hmp_layer)

        del self.convs
        del self.prods
        del self.reshapes
        
        if self.equivariant_pred:
            self.pred = torch.nn.Linear(self.hidden_irreps.dim, out_dim)
        else:
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
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

    def forward(self, batch, return_virtual_edges=False):
        h = self.emb_in(batch.atoms)
        pos = batch.pos

        vectors = pos[batch.edge_index[0]] - pos[batch.edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        virtual_adjs = []
        masks = []
        
        last_layer_virtual_edges = None
        
        for i, layer in enumerate(self.hmp_layers):
            h, A_virtual, m, edge_index_virtual_global = layer(h, pos, batch.edge_index, edge_sh, edge_feats, batch.batch)
            virtual_adjs.append(A_virtual)
            masks.append(m)
            if i == len(self.hmp_layers) - 1:
                last_layer_virtual_edges = edge_index_virtual_global

        pooled_h = self.pool(h, batch.batch)
        
        if not self.equivariant_pred:
            pooled_h = pooled_h[:, :self.emb_dim]
        
        prediction = self.pred(pooled_h)
        
        if return_virtual_edges:
            if last_layer_virtual_edges is None:
                last_layer_virtual_edges = torch.empty((2, 0), dtype=torch.long, device=h.device)
            return prediction, last_layer_virtual_edges
        else:
            return prediction