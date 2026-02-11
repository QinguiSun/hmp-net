# In a new file, e.g., models/hmp/hmp_equiformer.py

import torch
import e3nn
from e3nn import o3
from torch_geometric.utils import to_dense_batch

# Import the modules we've defined/copied
from .pool import TopKPooling # Assuming pool.py is in the same folder
from equiformer_lib.equiformer.nets.graph_attention_transformer import GraphAttentionTransformer
from equiformer_lib.equiformer.nets.tensor_product_rescale import LinearRS


def create_fully_connected_edges(batch: torch.Tensor, device: torch.device):
    """
    For a batch of nodes, creates a fully-connected edge_index for each graph.
    """
    # Get the number of nodes in each graph
    num_nodes_per_graph = torch.bincount(batch)
    # Get the starting index for each graph in the batch
    cumulative_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
    
    edge_indices = []
    for i in range(len(num_nodes_per_graph)):
        num_nodes = num_nodes_per_graph[i]
        start_idx = cumulative_nodes[i]
        end_idx = cumulative_nodes[i+1]
        
        # Create a grid of all possible edges within the graph's node indices
        adj = torch.ones(num_nodes, num_nodes, device=device)
        # Remove self-loops
        adj.fill_diagonal_(0)
        
        # Convert adjacency matrix to edge_index format and offset by the graph's start index
        edge_index = adj.nonzero(as_tuple=False).t() + start_idx
        edge_indices.append(edge_index)
        
    return torch.cat(edge_indices, dim=1)


class HMP_Equiformer_Layer(torch.nn.Module):
    def __init__(self, local_mp: torch.nn.Module, global_mp: torch.nn.Module, pool: torch.nn.Module):
        super().__init__()
        
        # --- This modular design directly addresses your future plans ---
        # 1. You can pass any Equiformer instance here.
        self.local_mp = local_mp
        
        # 2. You can pass any pooling module here.
        self.pool = pool
        
        # 3. Another Equiformer for the global step.
        self.global_mp = global_mp
        
        # 4. A simple linear layer for the "unpooling" step.
        # It needs to project the global features back to the local feature space.
        # The irreps must match the output of global_mp and the input of the next layer's local_mp.
        # For the MVP, we assume they are the same.
        self.unpool_proj = LinearRS(self.global_mp.irreps_node_embedding, 
                                    self.local_mp.irreps_node_embedding)

    def forward(self, node_features, pos, edge_index, batch, node_atom):
        # The input `node_features` must be a plain torch.Tensor with the correct feature dimension
        # that the `local_mp` Equiformer expects.
        
        # 1. LOCAL MESSAGE PASSING
        # -----------------------
        # The local_mp Equiformer operates on the original graph structure.
        # We need to compute the necessary edge attributes on the fly.
        
        edge_src, edge_dst = edge_index
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_sh = o3.spherical_harmonics(l=self.local_mp.irreps_edge_attr,
                                         x=edge_vec, normalize=True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.local_mp.rbf(edge_length)
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        h_local = self.local_mp.blocks[0]( # Assuming 1 block for MVP
            node_input=node_features, node_attr=node_attr, 
            edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
            edge_scalars=edge_length_embedding, batch=batch)
            
        # 2. POOLING
        # ----------
        # The pooling module selects master nodes based on local features.
        master_indices, master_mask = self.pool(h_local, batch)

        # If a graph has no master nodes, we can't do global MP.
        # In this case, we just return the features from local MP.
        if master_indices.numel() == 0:
            return h_local, pos, edge_index, batch, node_atom

        # 3. PREPARE FOR GLOBAL MESSAGE PASSING
        # -------------------------------------
        # We now create a new, smaller graph consisting only of the master nodes.
        pos_master = pos[master_mask]
        batch_master = batch[master_mask]
        h_master_initial = h_local[master_mask]

        # Your key idea: create a fully-connected graph for the master nodes.
        edge_index_global = create_fully_connected_edges(batch_master, device=pos.device)
        
        # 4. GLOBAL MESSAGE PASSING
        # -------------------------
        # The global_mp Equiformer operates on this new, dense graph.
        # We need to re-calculate edge attributes for this new graph.
        edge_src_g, edge_dst_g = edge_index_global
        edge_vec_g = pos_master[edge_src_g] - pos_master[edge_dst_g]
        edge_sh_g = o3.spherical_harmonics(l=self.global_mp.irreps_edge_attr,
                                           x=edge_vec_g, normalize=True, normalization='component')
        edge_length_g = edge_vec_g.norm(dim=1)
        edge_length_embedding_g = self.global_mp.rbf(edge_length_g)
        node_attr_g = torch.ones_like(h_master_initial.narrow(1, 0, 1))
        
        h_global_out = self.global_mp.blocks[0]( # Assuming 1 block for MVP
            node_input=h_master_initial, node_attr=node_attr_g, 
            edge_src=edge_src_g, edge_dst=edge_dst_g, edge_attr=edge_sh_g, 
            edge_scalars=edge_length_embedding_g, batch=batch_master)
            
        # 5. UNPOOLING / BROADCAST
        # ------------------------
        # We update the features of the master nodes in the original graph
        # with the refined features from the global context.
        
        # Project the global features back to the right dimension (usually the same)
        h_update = self.unpool_proj(h_global_out)
        
        # Create the final output tensor with a residual connection.
        h_out = h_local.clone()
        h_out[master_mask] = h_out[master_mask] + h_update
        
        # The graph structure itself hasn't changed, so we pass the original ones through.
        return h_out, pos, edge_index, batch, node_atom