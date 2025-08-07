import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool

from models.layers.spherenet_layer import SphereNetLayer
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
        adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)

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
        in_dim: int = 1,
        out_dim: int = 1,
        s_dim: int = 16, # Dimension of scalar features for attention
        master_selection_hidden_dim: int = 32,
        lambda_attn: float = 0.1,
        master_rate: float = 0.25,
    ):
        super().__init__()
        self.master_rate = master_rate

        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        self.hmp_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            spherenet_layer = SphereNetLayer(emb_dim, emb_dim, 7, 5)
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

    def forward(self, batch):
        h = self.emb_in(batch.atoms)
        pos = batch.pos

        virtual_adjs = []
        masks = []

        for layer in self.hmp_layers:
            h, pos, A_virtual, m = layer(h, pos, batch.edge_index, batch.batch)
            virtual_adjs.append(A_virtual)
            masks.append(m)

        pooled_h = self.pool(h, batch.batch)
        prediction = self.pred(pooled_h)
        return prediction
