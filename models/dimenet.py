from typing import Callable, Tuple, Union

import torch
from torch.nn import functional as F
#from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.nn import DimeNetPlusPlus
from torch_scatter import scatter

from torch_geometric.typing import  SparseTensor
from torch import Tensor

def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

class DimeNetPPModel(DimeNetPlusPlus):
    """
    DimeNet model from "Directional message passing for molecular graphs".

    This class extends the DimeNetPlusPlus base class for PyG.
    """
    def __init__(
        self, 
        in_dim: int = 1,
        hidden_channels: int = 128, 
        num_embeddings: int = 1,
        out_dim: int = 1, 
        num_layers: int = 4,    # num_blocks
        int_emb_size: int = 64, 
        basis_emb_size: int = 8,    # num_bilinear
        out_emb_channels: int = 256, 
        num_spherical: int = 7, 
        num_radial: int = 6, 
        cutoff: float = 10, 
        max_num_neighbors: int = 32, 
        envelope_exponent: int = 5, 
        num_before_skip: int = 1, 
        num_after_skip: int = 2, 
        num_output_layers: int = 3, 
        act: Union[str, Callable] = 'swish'
    ):
        """
        Initializes an instance of the DimeNetPPModel class with the provided parameters.

        Parameters:
        - hidden_channels (int): Number of channels in the hidden layers (default: 128)
        - num_embeddings (int): (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - num_layers (int): Number of layers in the model (default: 4)
        - int_emb_size (int): Embedding size for interaction features (default: 64)
        - basis_emb_size (int): Embedding size for basis functions (default: 8)
        - out_emb_channels (int): Number of channels in the output embeddings (default: 256)
        - num_spherical (int): Number of spherical harmonics (default: 7)
        - num_radial (int): Number of radial basis functions (default: 6)
        - cutoff (float): Cutoff distance for interactions (default: 10)
        - max_num_neighbors (int): Maximum number of neighboring atoms to consider (default: 32)
        - envelope_exponent (int): Exponent of the envelope function (default: 5)
        - num_before_skip (int): Number of layers before the skip connections (default: 1)
        - num_after_skip (int): Number of layers after the skip connections (default: 2)
        - num_output_layers (int): Number of output layers (default: 3)
        - act (Union[str, Callable]): Activation function (default: 'swish' or callable)

        Note:
        - The `act` parameter can be either a string representing a built-in activation function,
        or a callable object that serves as a custom activation function.
        """
        super().__init__(
            hidden_channels, 
            out_dim, 
            num_layers, 
            int_emb_size, 
            basis_emb_size, 
            out_emb_channels, 
            num_spherical, 
            num_radial, 
            cutoff, 
            max_num_neighbors, 
            envelope_exponent, 
            num_before_skip, 
            num_after_skip, 
            num_output_layers, 
            act
        )
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hidden_channels)

    def forward(self, batch):
        
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            batch.edge_index, num_nodes=batch.num_nodes)

        # Calculate distances.
        dist = (batch.pos[i] - batch.pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = batch.pos[idx_i]
        pos_ji, pos_ki = batch.pos[idx_j] - pos_i, batch.pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # 取原子序号：优先 atoms，其次 z，再次 atomic_numbers
        atoms = getattr(batch, "atoms", None)
        if atoms is None:
            atoms = getattr(batch, "z", None)
        if atoms is None:
            atoms = getattr(batch, "atomic_numbers", None)
        if atoms is None:
            raise AttributeError(
                "No atomic numbers tensor found. Expected one of: batch.atoms, batch.z, batch.atomic_numbers"
            )
        atoms = atoms.long()  # 确保是 Long
        
        # Embedding block.
        x = self.emb(atoms, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=batch.pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P.sum(dim=0) if batch is None else scatter(P, batch.batch, dim=0)
