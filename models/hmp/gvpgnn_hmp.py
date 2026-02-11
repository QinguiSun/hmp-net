import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_geometric.nn import global_add_pool
from torch.nn import functional as F

from models.layers.gvp_layer import GVPConvLayer
from models.hmp.master_selection import MasterSelection
from models.hmp.virtual_generation import VirtualGeneration

class HMPLayer(nn.Module):
    """
    A Hierarchical Message Passing (HMP) layer that wraps a backbone GNN layer.
    """
    def __init__(self, backbone_layer, h_dim, s_dim, master_selection_hidden_dim, lambda_attn, master_rate):
        super().__init__()
        self.backbone_layer = backbone_layer
        # ``h_dim`` stores the scalar and vector dimensionalities expected by the
        # underlying GVP layer.  We keep them around so that we can construct edge
        # attributes with matching sizes on the fly during the forward pass.
        self.s_dim = h_dim[0]
        self.v_dim = h_dim[1]
        self.master_selection = MasterSelection(in_dim=s_dim, hidden_dim=master_selection_hidden_dim, ratio=master_rate)
        self.virtual_generation = VirtualGeneration(in_dim=s_dim, lambda_attn=lambda_attn)

    def _edge_features(self, pos, edge_index):
        """Create scalar and vector edge features from node positions.

        The original implementation passed the raw node positions directly to
        ``GVPConvLayer`` as ``edge_attr``.  ``GVPConvLayer`` expects a tuple of
        scalar and vector edge features of shape ``(n_edges, s_dim)`` and
        ``(n_edges, v_dim, 3)`` respectively.  Supplying only the positions causes
        the tuple unpacking inside ``tuple_cat`` to misinterpret the tensor,
        ultimately leading to mismatched tensor sizes during concatenation.  Here
        we derive simple geometric features – edge lengths and normalised
        directions – and broadcast them to the required dimensions.
        """

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
        h_local = (h[0] + h_update[0], h[1] + h_update[1]) # Residual connection for features
        pos_local = pos

        # 2. Invariant Topology Learning
        h_scalar = h_local[0][:, :self.s_dim]
        m, _ = self.master_selection(h_scalar) # m is the soft mask

        master_nodes_mask = m > 0.5
        num_master_nodes = master_nodes_mask.sum()

        if num_master_nodes <= 1:
            # Not enough master nodes, skip hierarchical message passing
            return h_local, pos_local, torch.zeros((0, 0), device=h[0].device), m

        master_indices = torch.where(master_nodes_mask)[0]

        # Create induced subgraph for master nodes
        edge_index_induced, _ = subgraph(master_indices, edge_index, relabel_nodes=True, num_nodes=num_nodes)
        adj_induced = to_dense_adj(edge_index_induced, max_num_nodes=num_master_nodes).squeeze(0)

        h_master = (h_local[0][master_nodes_mask], h_local[1][master_nodes_mask])
        pos_master = pos_local[master_nodes_mask]
        h_master_scalar = h_master[0][:, :self.s_dim]

        # Generate virtual edges
        A_virtual = self.virtual_generation(h_master_scalar, adj_induced)

        # 3. Hierarchical Propagation
        edge_index_virtual, _ = dense_to_sparse(A_virtual)
        edge_index_master = torch.cat([edge_index_induced, edge_index_virtual], dim=1)
        edge_attr_master = self._edge_features(pos_master, edge_index_master)

        h_master_update = self.backbone_layer(
            h_master, edge_index_master, edge_attr_master
        )
        h_hierarchical = (h_master[0] + h_master_update[0], h_master[1] + h_master_update[1])
        pos_hierarchical = pos_master

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
    """
    HMP-enhanced GVP-GNN model.
    """
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
        # Each vector feature consists of 3 components.  The original
        # implementation only allocated ``v_dim`` channels which results in a
        # tensor of shape ``(n_nodes, 1, v_dim)`` after an ``unsqueeze`` in the
        # forward pass.  Downstream ``GVPConv`` layers expect the vector features
        # in the canonical ``(n_nodes, v_dim, 3)`` layout.  The mismatch caused a
        # runtime error when the tensor was reshaped inside the convolution
        # because the last dimension was ``v_dim`` instead of ``3``.  We allocate
        # ``v_dim * 3`` embedding dimensions here so that the tensor can be
        # properly viewed into ``(n_nodes, v_dim, 3)``.
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
        h_scalar = self.emb_in_s(batch.z)
        # Reshape the embedding output into ``(n_nodes, v_dim, 3)`` so that it is
        # compatible with the expectations of ``GVPConv`` layers.
        h_vector = self.emb_in_v(batch.z).view(-1, self.v_dim, 3)
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
