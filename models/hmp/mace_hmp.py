# mace_hmp.py
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
import e3nn
from torch_geometric.utils import remove_self_loops, to_undirected

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

    def forward(self, h, pos, edge_index, edge_sh, edge_feats, batch):
        # 1. Local Propagation
        def safe_shape(x):
            try:
                return tuple(x.shape)
            except Exception:
                if isinstance(x, (list, tuple)):
                    return f"tuple(len={len(x)}): " + str([getattr(t, 'shape', type(t).__name__) for t in x])
                return type(x).__name__

        #print("h:", safe_shape(h))
        h_update = self.local_conv(h, edge_index, edge_sh, edge_feats)
        #print("h_update:", safe_shape(h_update))
        sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
        #print("sc:", safe_shape(sc))
        #print("self.reshape(h_update): ", safe_shape(self.reshape(h_update)))
        h_local = self.prod(self.reshape(h_update), sc, None)
        #print("h_local:", safe_shape(h_local)) 
        #print("h_local:\n", h_local)
        
        # 2. Invariant Topology Learning
        h_scalar = h_local[:, :self.s_dim]
        #print("h_scalar:", safe_shape(h_scalar)) 
        m, _ = self.master_selection(h_scalar)
        
        master_nodes_mask = m > 0.5
        #print("master_nodes_mask:\n", master_nodes_mask)
        num_master_nodes = master_nodes_mask.sum()
        #print("num_master_nodes:", num_master_nodes)
        
        if num_master_nodes <= 1:
            # MODIFICATION START: 当无法形成有意义的层级结构时，返回空的虚拟边索引
            empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=h.device)
            return h_local, torch.zeros((0, 0), device=h.device), m, empty_edge_index
            # MODIFICATION END

        master_indices = torch.where(master_nodes_mask)[0]
        #print("master_indices:", safe_shape(master_indices))
        #print("master_indices:\n", master_indices)
        
        # Create induced subgraph for master nodes
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=h.size(0))
        #print("edge_index_induced:\n", edge_index_induced)
        adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        #print("adj_induced:\n", adj_induced)        

        h_master = h_local[master_nodes_mask]
        #print("h_master:", safe_shape(h_master))
        #print("h_master:", h_master)
        pos_master = pos[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]
        #print("h_master_scalar:", safe_shape(h_master_scalar))    
        #print("h_master_scalar:\n", h_master_scalar)       
   

        # Generate virtual edges
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)
        # A_virtual 是 MxM（master 节点）的邻接；保持为 master-local 索引空间
        #A_virtual = A_virtual.clone()  # 用来避免你后续的原地操作（如 fill_diagonal_）影响到别处对 A_virtual 的引用；
                                       # 梯度仍然会正常回传。但它会多占一次内存拷贝，只有在你确实要对这个张量做原地修改时才有必要。
        #A_virtual.fill_diagonal_(0)    # 可选：去掉自环，避免零向量导致球谐数值问题
        
        # 3. Hierarchical Propagation
        edge_reindex_virtual, _ = dense_to_sparse(A_virtual)               # master graph的边索引（从0, 1, 2，… 重新编码的）
        edge_index_virtual, _ = remove_self_loops(edge_reindex_virtual)    # 可选："再次"确保无自环
        edge_index_virtual = to_undirected(edge_index_virtual, num_nodes=h_master.size(0))
        """
        edge_index_virtual = torch.zeros_like(edge_reindex_virtual)
        num_edge = 0
        for i, j in edge_reindex_virtual.t():   # 转置后按列遍历
            edge_index_virtual[num_edge] = [master_indices[i], master_indices[j]]  # original graph的边索引（应该是 2， 3， 5,… 编码的）
            num_edge += 1
        """
        
        master_indices = master_indices.to(device=edge_reindex_virtual.device, dtype=torch.long)
        edge_index_virtual_global = master_indices[edge_reindex_virtual]  # 仅用于全局用途
        
        #print("edge_index_virtual:", safe_shape(edge_index_virtual)) 
        #print("edge_reindex_virtual:\n", edge_reindex_virtual) 
        # 两类边现在都在同一个（master-local）索引空间
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        # 越界断言（避免空边时报 .max() 错）
        if edge_index_master.numel() > 0:
            assert edge_index_master.max() < h_master.size(0), \
                f"master edge index {edge_index_master.max().item()} >= num master {h_master.size(0)}"
        #print("edge_index_induced:\n", edge_index_induced)
        #print("edge_index_virtual:\n", edge_index_virtual)
        #print("edge_index_master:", safe_shape(edge_index_master))
        #print("edge_index_master:\n", edge_index_master) 
        # edge_index_master: [2, E]（master-local 索引空间）
        M = h_master.size(0)
        E = edge_index_master.size(1) if edge_index_master.numel() > 0 else 0
        
        # 版本 A：至少有一条边（若 M>=2 则要求 E>=1）
        assert (M <= 1) or (E >= 1), \
            f"No edges among master nodes: num_master={M}, num_edges={E}"
            
        # 版本 B：每个 master 节点至少有一条相连的边（无孤立点）
        # deg[k] 是第 k 个 master 节点的度
        deg = torch.bincount(edge_index_master.reshape(-1), minlength=M)
        isolated = (deg == 0).nonzero(as_tuple=False).view(-1)
        assert (M == 0) or (M == 1 and E == 0) or isolated.numel() == 0, \
            f"Isolated master nodes found (no incident edges): {isolated.tolist()} | num_master={M}, num_edges={E}"
        
        # Need to re-compute edge attributes for the master graph
        vectors_master = pos_master[edge_index_master[0]] - pos_master[edge_index_master[1]]
        lengths_master = torch.linalg.norm(vectors_master, dim=-1, keepdim=True)
        edge_sh_master = self.spherical_harmonics(vectors_master)
        edge_feats_master = self.radial_embedding(lengths_master)

        h_master_update = self.hierarchical_conv(h_master, edge_index_master, edge_sh_master, edge_feats_master)
        #print("h_master_update:", safe_shape(h_master_update))
        sc_master = F.pad(h_master, (0, h_master_update.shape[-1] - h_master.shape[-1]))
        #print("sc_master:", safe_shape(sc_master))
        #print("self.reshape(h_master_update):", safe_shape(self.reshape(h_master_update)))
        h_hierarchical = self.prod(self.reshape(h_master_update), sc_master, None)
        #print("////////////////////////////////////////////////////////////////")
        
        # 4. Feature Aggregation
        h_hierarchical_expanded = torch.zeros_like(h_local)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical
        
        m_expanded = m.unsqueeze(1)
        h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded
        
        # MODIFICATION START: 返回在全局索引空间下的虚拟边
        return h_final, A_virtual, m, edge_index_virtual_global
        # MODIFICATION END


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
        # **kwargs: 'num_layers': args.L, 'correlation': 3, 'max_ell': 3, 'out_dim': 1, 
        # 'num_layers': args.L, 'num_embeddings': num_atom_classes, 'emb_dim': 128, 'correlation': 3, 'max_ell': 3, 'out_dim': 1, **hmp_params
        self.emb_dim = emb_dim
        
        self.emb_in = torch.nn.Embedding(num_embeddings, emb_dim)

        # ------------------------------------------------------------------
        # Ensure hidden_irreps is computed.  If hiddnode_featsen_irreps=None or has zero
        # dimension (no valid tensor‑product instructions) recompute it from
        # the spherical harmonics irreps and the embedding dimension.
        if getattr(self, "hidden_irreps", None) is None or self.hidden_irreps.dim == 0:
            sh_irreps = e3nn.o3.Irreps.spherical_harmonics(self.max_ell)
            # replicate each irrep emb_dim times and simplify
            self.hidden_irreps = (sh_irreps * self.emb_dim).sort()[0].simplify()
        self.master_rate = master_rate
        aggr = kwargs.get("aggr", "sum")
        
        # ------------------------------------------------------------------
        # Override backbone with TensorProductConvLayer
        # First layer: scalar only -> tensor
        #print(f"emb_dim: {emb_dim}")
        #print(f"self.hidden_irreps:{self.hidden_irreps}")
        sh_irreps = self.spherical_harmonics.irreps_out
        #print(f"sh_irreps: {sh_irreps}")
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

        # Intermediate layers: tensor -> tensor
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

        # s_dim is the number of 0e irreps in the hidden representation
        # (only scalars drive master-node selection).  Compute it directly
        # from hidden_irreps.
        s_dim = 0
        for mul, ir in self.hidden_irreps:
            if ir.l == 0 and ir.p == 1:
                s_dim += mul
        s_dim *= s_dim_scale

        # Create a single tensor-to-tensor convolution layer for hierarchical propagation
        # This will be shared across all HMP layers for the hierarchical step
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
                master_selection_hidden_dim=s_dim,  # A reasonable default
                lambda_attn=0.1  # A reasonable default
            )
            self.hmp_layers.append(hmp_layer)

        # Remove original layers to avoid confusion and duplicate parameters
        del self.convs
        del self.prods
        del self.reshapes
        
        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(self.hidden_irreps.dim, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
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

    # MODIFICATION START: 添加可选参数 return_virtual_edges 以控制返回值
    def forward(self, batch, return_virtual_edges=False):
    # MODIFICATION END
        h = self.emb_in(batch.atoms)
        pos = batch.pos

        vectors = pos[batch.edge_index[0]] - pos[batch.edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        virtual_adjs = []
        masks = []
        
        # MODIFICATION START: 准备一个变量来存储最后一层的虚拟边
        last_layer_virtual_edges = None
        # MODIFICATION END
        
        # MODIFICATION START: 迭代时获取索引，以便识别最后一层
        for i, layer in enumerate(self.hmp_layers):
            # 解包从 HMP_MACELayer 返回的新增值 edge_index_virtual_global
            h, A_virtual, m, edge_index_virtual_global = layer(h, pos, batch.edge_index, edge_sh, edge_feats, batch.batch)
            virtual_adjs.append(A_virtual)
            masks.append(m)
            # 如果是最后一层，则保存其虚拟边
            if i == len(self.hmp_layers) - 1:
                last_layer_virtual_edges = edge_index_virtual_global
        # MODIFICATION END

        pooled_h = self.pool(h, batch.batch)
        
        if not self.equivariant_pred:
            pooled_h = pooled_h[:, :self.emb_dim]
        
        prediction = self.pred(pooled_h)
        
        # MODIFICATION START: 根据参数决定返回值
        if return_virtual_edges:
            # 确保即使没有生成虚拟边也返回一个格式正确的空张量
            if last_layer_virtual_edges is None:
                last_layer_virtual_edges = torch.empty((2, 0), dtype=torch.long, device=h.device)
            return prediction, last_layer_virtual_edges
        else:
            # 默认行为，仅返回预测值，以兼容训练循环
            return prediction
        # MODIFICATION END
