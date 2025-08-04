import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
import e3nn

from models.mace import MACEModel
from models.mace_modules.irreps_tools import reshape_irreps
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration

class HMP_MACELayer(nn.Module):
    """
    HMP Layer for the MACE architecture.
    """
    def __init__(self, conv, reshape, prod, spherical_harmonics, radial_embedding, s_dim, master_selection_hidden_dim, lambda_attn):
        super().__init__()
        self.conv = conv
        self.reshape = reshape
        self.prod = prod
        self.spherical_harmonics = spherical_harmonics
        self.radial_embedding = radial_embedding
        self.s_dim = s_dim
        
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)

    def forward(self, h, pos, edge_index, edge_sh, edge_feats, batch):
        # 1. Local Propagation
        h_update = self.conv(h, edge_index, edge_sh, edge_feats)
        sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
        h_local = self.prod(self.reshape(h_update), sc, None)

        # 2. Invariant Topology Learning
        h_scalar = h_local[:, :self.s_dim]
        m, _ = self.master_selection(h_scalar)
        
        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()

        if num_master_nodes <= 1:
            return h_local, torch.zeros((0, 0), device=h.device), m

        master_indices = torch.where(master_nodes_mask)[0]
        
        # Create induced subgraph for master nodes
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=h.size(0))
        adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)

        h_master = h_local[master_nodes_mask]
        pos_master = pos[master_nodes_mask]
        h_master_scalar = h_master[:, :self.s_dim]

        # Generate virtual edges
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)
        
        # 3. Hierarchical Propagation
        edge_index_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        
        # Need to re-compute edge attributes for the master graph
        vectors_master = pos_master[edge_index_master[0]] - pos_master[edge_index_master[1]]
        lengths_master = torch.linalg.norm(vectors_master, dim=-1, keepdim=True)
        edge_sh_master = self.spherical_harmonics(vectors_master)
        edge_feats_master = self.radial_embedding(lengths_master)

        h_master_update = self.conv(h_master, edge_index_master, edge_sh_master, edge_feats_master)
        sc_master = F.pad(h_master, (0, h_master_update.shape[-1] - h_master.shape[-1]))
        h_hierarchical = self.prod(self.reshape(h_master_update), sc_master, None)

        # 4. Feature Aggregation
        h_hierarchical_expanded = torch.zeros_like(h_local)
        h_hierarchical_expanded[master_nodes_mask] = h_hierarchical
        
        m_expanded = m.unsqueeze(1)
        h_final = (1 - m_expanded) * h_local + m_expanded * h_hierarchical_expanded
        
        return h_final, A_virtual, m


class HMP_MACEModel(MACEModel):
    """
    HMP-enhanced MACE model.
    """
    def __init__(self, master_rate=0.25, s_dim_scale=1, **kwargs):
        super().__init__(**kwargs)
        self.master_rate = master_rate
        
        # s_dim is the scalar feature dimension, which is emb_dim for MACE
        s_dim = self.emb_dim * s_dim_scale

        self.hmp_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            hmp_layer = HMP_MACELayer(
                conv=self.convs[i],
                reshape=self.reshapes[i],
                prod=self.prods[i],
                spherical_harmonics=self.spherical_harmonics,
                radial_embedding=self.radial_embedding,
                s_dim=s_dim,
                master_selection_hidden_dim=s_dim, # A reasonable default
                lambda_attn=0.1 # A reasonable default
            )
            self.hmp_layers.append(hmp_layer)
        
        # Remove original layers to avoid confusion and duplicate parameters
        del self.convs
        del self.prods
        del self.reshapes

    def forward(self, batch):
        h = self.emb_in(batch.atoms)
        pos = batch.pos

        vectors = pos[batch.edge_index[0]] - pos[batch.edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        virtual_adjs = []
        masks = []

        for layer in self.hmp_layers:
            h, A_virtual, m = layer(h, pos, batch.edge_index, edge_sh, edge_feats, batch.batch)
            virtual_adjs.append(A_virtual)
            masks.append(m)

        pooled_h = self.pool(h, batch.batch)
        
        if not self.equivariant_pred:
            pooled_h = pooled_h[:, :self.emb_dim]
        
        prediction = self.pred(pooled_h)
        
        l_struct = sum(torch.norm(A, p=1) for A in virtual_adjs if A.numel() > 0)
        if virtual_adjs:
            l_struct = l_struct / len(virtual_adjs)
            
        l_rate = 0
        if masks:
            for m in masks:
                rate = m.sum() / m.size(0)
                l_rate += (rate - self.master_rate)**2
            l_rate = l_rate / len(masks)

        return {
            'pred': prediction,
            'l_struct': l_struct,
            'l_rate': l_rate,
        }
