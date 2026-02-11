import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool

from models.layers.spherenet_layer import *
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration

            
class HMPLayer(nn.Module):
    """
    A Hierarchical Message Passing (HMP) layer that wraps a backbone GNN layer.
    """
    def __init__(self, backbone_layer, h_dim, s_dim, master_selection_hidden_dim, lambda_attn, master_rate):
        super().__init__()
        self.backbone_layer = backbone_layer
        self.s_dim = s_dim
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim, ratio=master_rate)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)

    def forward(self, h, pos, edge_index, batch):
        # 1. Local Propagation
        num_nodes = h.size(0)
        h_update = self.backbone_layer(h, pos, edge_index)
        h_local = h + h_update  # Residual connection for features
        pos_local = pos

        # 2. Invariant Topology Learning
        h_scalar = h_local[:, :self.s_dim]
        m, _ = self.master_selection(h_scalar) # m is the soft mask

        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()

        if num_master_nodes <= 1:
            # Not enough master nodes, skip hierarchical message passing
            return h_local, pos_local, torch.zeros((0, 0), device=h.device), m

        master_indices = torch.where(master_nodes_mask)[0]

        # Create induced subgraph for master nodes
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        #adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)
        
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
        pos_master = pos_local[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]

        # Generate virtual edges
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)

        # 3. Hierarchical Propagation
        edge_index_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)

        h_master_update = self.backbone_layer(
            h_master, pos_master, edge_index_master
        )
        h_hierarchical = h_master + h_master_update
        pos_hierarchical = pos_master

        # 4. Feature Aggregation
        h_hierarchical_expanded = torch.zeros_like(h_local)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical

        m_expanded = m.unsqueeze(1)
        h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded
        pos_final = pos_local

        return h_final, pos_final, A_virtual, m


class HMP_SphereNetModel(torch.nn.Module):
    """
    HMP-enhanced SphereNet model.
    """
    def __init__(
        self,
        num_layers: int = 5,
        emb_dim: int = 128,
        num_embeddings: int = 1,
        out_dim: int = 1,
        s_dim: int = 16, # Dimension of scalar features for attention
        master_selection_hidden_dim: int = 32,
        lambda_attn: float = 0.1,
        master_rate: float = 0.25,
        # SphereNet specific params
        int_emb_size: int = 64,
        basis_emb_size_dist: int = 8,
        basis_emb_size_angle: int = 8,
        basis_emb_size_torsion: int = 8,
        out_emb_channels: int = 256,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act=swish,
        output_init: str = 'GlorotOrthogonal',
        use_node_features: bool = True,
    ):
        super().__init__()
        
        self.master_rate = master_rate
        self.num_layers = num_layers
        self.cutoff = cutoff

        self.emb = emb(num_spherical, num_radial, self.cutoff, envelope_exponent)
        self.init_e = init(num_embeddings, num_radial, emb_dim, act, use_node_features=use_node_features)
        self.init_v = update_v(emb_dim, out_emb_channels, emb_dim, num_output_layers, act, output_init)
    
        self.hmp_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            spherenet_layer = SphereNetLayer(
                hidden_channels=emb_dim,
                out_emb_channels=out_emb_channels,
                int_emb_size=int_emb_size,
                basis_emb_size_dist=basis_emb_size_dist,
                basis_emb_size_angle=basis_emb_size_angle,
                basis_emb_size_torsion=basis_emb_size_torsion,
                num_spherical=num_spherical,
                num_radial=num_radial,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                act=act,
                num_output_layers=num_output_layers,
                output_init=output_init,
                out_channels=emb_dim,
            )
            hmp_layer = HMPLayer(
                backbone_layer=spherenet_layer,
                h_dim=emb_dim,
                s_dim=s_dim,
                master_selection_hidden_dim=master_selection_hidden_dim,
                lambda_attn=lambda_attn,
                master_rate=self.master_rate
            )
            self.hmp_layers.append(hmp_layer)

        self.pool = global_add_pool
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, out_dim)
        )

    def update_tau(self, epoch, n_epochs):
        """Update the Gumbel-Softmax temperature `tau` for all HMP layers."""
        initial_tau = 1.0
        final_tau = 0.1
        anneal_epochs = n_epochs // 2

        if epoch <= anneal_epochs:
            tau = initial_tau - (initial_tau - final_tau) * (epoch / anneal_epochs)
        else:
            tau = final_tau

        for layer in self.hmp_layers:
            layer.master_selection.tau.fill_(tau)

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        edge_index = batch_data.edge_index
        num_nodes = z.size(0)

        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)
        
        emb = self.emb(dist, angle, torsion, idx_kj)

        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i, dim_size=num_nodes)
        
        virtual_adjs = []
        masks = []
        
        for hmp_layer in self.hmp_layers:
            # The HMPLayer is not directly compatible with SphereNet's update mechanism.
            # We manually implement the HMP logic here to resolve the discrepancy.

            spherenet_layer = hmp_layer.backbone_layer

            # 1. Local Propagation
            e_local, v_update_local = spherenet_layer(e, v, i, emb, idx_kj, idx_ji, dim_size=num_nodes)
            v_local = v + v_update_local

            # 2. Invariant Topology Learning (Master Node Selection)
            h_scalar = v_local[:, :hmp_layer.s_dim]
            m, _ = hmp_layer.master_selection(h_scalar)

            master_nodes_mask = m > 0.5
            num_master_nodes = master_nodes_mask.sum()

            if num_master_nodes <= 1:
                # Skip hierarchical message passing
                e, v = e_local, v_local
                virtual_adjs.append(torch.zeros((0, 0), device=e[0].device))
                masks.append(m)
                continue

            master_indices = torch.where(master_nodes_mask)[0]

            # 3. Hierarchical Propagation
            # Create master subgraph
            edge_index_master, _, edge_mask_master = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes, return_edge_mask=True)
            '''
            sub_out = subgraph(
                master_indices, edge_index, e,  # 显式传入 edge_attr=e
                relabel_nodes=True,
                num_nodes=num_nodes,
                return_edge_mask=True
            )

            # 兼容：有 edge_attr 返回三元组；无 edge_attr 返回二元组
            if isinstance(sub_out, tuple) and len(sub_out) == 3:
                edge_index_master, e_master, edge_mask_master = sub_out
            elif isinstance(sub_out, tuple) and len(sub_out) == 2:
                edge_index_master, edge_mask_master = sub_out
                e_master = e[edge_mask_master] if e is not None else None
            else:
                # 理论上不会走到这里
                raise RuntimeError(f"Unexpected subgraph() output: {type(sub_out)} with length {getattr(sub_out, '__len__', lambda: 'NA')()}")
            '''
            # 后续一律用子图的边特征 e_master，而不是原图的 e
            pos_master = pos[master_nodes_mask]
          
            def safe_shape(x):
                try:
                    return tuple(x.shape)
                except Exception:
                    if isinstance(x, (list, tuple)):
                        return f"tuple(len={len(x)}): " + str([getattr(t, 'shape', type(t).__name__) for t in x])
                    return type(x).__name__

            #print("edge_index_master:", safe_shape(edge_index_master))
            #print("e_master:", safe_shape(e_master))
            #print("edge_mask_master:", safe_shape(edge_mask_master))


            # Geometric features for master subgraph
            dist_master, angle_master, torsion_master, i_master, j_master, idx_kj_master, idx_ji_master = xyz_to_dat(pos_master, edge_index_master, num_master_nodes, use_torsion=True)
            emb_master = self.emb(dist_master, angle_master, torsion_master, idx_kj_master)

            # Edge and node features for master subgraph
            e_master = (e_local[0][edge_mask_master], e_local[1][edge_mask_master])
            v_master = v_local[master_nodes_mask]

            # Hierarchical update
            e_hier, v_update_hier = spherenet_layer(e_master, v_master, i_master, emb_master, idx_kj_master, idx_ji_master, dim_size=num_master_nodes)
            v_hier = v_master + v_update_hier

            # 4. Feature Aggregation
            v_hier_expanded = torch.zeros_like(v_local)
            v_hier_expanded[master_nodes_mask] = v_hier

            m_expanded = m.unsqueeze(1)
            v_final = (1 - m_expanded) * v_local + m_expanded * v_hier_expanded

            # We need to aggregate edge features too, but HMPLayer doesn't do this.
            # For simplicity, we use the locally propagated edge features.
            e_final = e_local

            v = v_final
            e = e_final

            masks.append(m)
            # virtual_adjs are not computed in this simplified integration.
            virtual_adjs.append(torch.zeros((0, 0), device=e[0].device))


        pooled_h = self.pool(v, batch)
        prediction = self.pred(pooled_h)
        return prediction
