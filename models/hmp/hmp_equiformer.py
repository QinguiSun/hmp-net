# In file: hmp-net-clean/models/hmp/hmp_equiformer.py

import torch
from torch_cluster import radius_graph

import e3nn
from e3nn import o3

# Import our building blocks
from .hmp_layer import HMP_Equiformer_Layer # We'll rename our file to hmp_layer.py
from .pool import TopKPooling

# Import Equiformer's own building blocks that we need
from equiformer_lib.equiformer.nets.graph_attention_transformer import (
    GraphAttentionTransformer, 
    NodeEmbeddingNetwork, 
    EdgeDegreeEmbeddingNetwork,
    ScaledScatter,
    _MAX_ATOM_TYPE,
    _AVG_DEGREE,
    _AVG_NUM_NODES
)
from equiformer_lib.equiformer.nets.gaussian_rbf import GaussianRadialBasisLayer
from equiformer_lib.equiformer.nets.tensor_product_rescale import LinearRS
from equiformer_lib.equiformer.nets.fast_activation import Activation

# Main network class
class HMP_Equiformer_Net(torch.nn.Module):
    def __init__(self,
                 num_hmp_layers: int = 2,
                 pooling_ratio: float = 0.5,
                 # We can expose more Equiformer hyperparameters here later
                 irreps_node_embedding='128x0e+64x1e+32x2e',
                 number_of_basis=128,
                 max_radius=5.0,
                 # MD17 specific parameters
                 task_mean=0.0,
                 task_std=1.0,
                 atomref=None,
                 ):
        super().__init__()
        
        # --- 1. CONFIGURATION ---
        self.max_radius = max_radius
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(self.irreps_node_embedding.lmax)
        
        # --- MD17: Store mean, std, and atomref ---
        self.task_mean = task_mean
        self.task_std = task_std
        self.register_buffer('atomref', atomref)
        
        
        # --- FIX #2: Correctly define fc_neurons from number_of_basis ---
        fc_neurons = [number_of_basis, 64, 64]
        
        # --- 2. INITIAL EMBEDDING LAYERS (Copied from Equiformer) ---
        # MD17 uses a different embedding strategy (atomic number, not one-hot)
        # For simplicity in the MVP, we will reuse the QM9 NodeEmbeddingNetwork
        # This is a reasonable starting point.
        # Converts atom numbers into initial feature vectors.
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=self.max_radius)
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, fc_neurons, _AVG_DEGREE)
            
        # --- 3. HIERARCHICAL MESSAGE PASSING LAYERS ---
        # This is the core of our new architecture.
        self.hmp_layers = torch.nn.ModuleList()
        
        # ==================== START OF THE FIX ====================
        
        # Correctly calculate the number of scalar features (l=0, p=1) from the irreps string.
        num_scalar_features = 0
        for mul, ir in self.irreps_node_embedding:
            if ir.l == 0 and ir.p == 1: # This is the condition for '0e'
                num_scalar_features += mul
        
        if num_scalar_features == 0:
            raise ValueError(f"The irreps '{self.irreps_node_embedding}' must contain scalar ('0e') features for pooling.")

        # ===================== END OF THE FIX =====================
        
        for _ in range(num_hmp_layers):
            # For each HMP layer, we need a local_mp, a global_mp, and a pooler.
            
            # For the MVP, we use a minimal Equiformer (1 block) for local and global steps.
            # This is where your UNet/ordering ideas can be implemented in the future.

            # NEW code
            # We must ensure the block does not change the irreps.
            # We do this by setting `irreps_feature` to be the same as `irreps_node_embedding`.
            shared_config = {
                'num_layers': 1,
                'irreps_node_embedding': self.irreps_node_embedding,
                'irreps_feature': self.irreps_node_embedding, # <--- THIS IS THE FIX
                'max_radius': self.max_radius,
                'number_of_basis': number_of_basis,
                'fc_neurons': fc_neurons, # <-- Pass the correct fc_neurons
                # Add other default args from the original Equiformer if needed
                # to prevent other errors. Let's start with this.
                'mean': self.task_mean, # Pass mean/std down
                'std': self.task_std,
            }

            local_mp_block = GraphAttentionTransformer(**shared_config)
            global_mp_block = GraphAttentionTransformer(**shared_config)
            
            local_mp_block.head = torch.nn.Identity()
            local_mp_block.norm = torch.nn.Identity()
            global_mp_block.head = torch.nn.Identity()
            global_mp_block.norm = torch.nn.Identity()
            
            # The pooling layer acts on the scalar features of the node embeddings.
            #num_scalar_features = self.irreps_node_embedding['0e'].dim
            #pooler = TopKPooling(in_channels=num_scalar_features, ratio=pooling_ratio)
            
            # ==================== RELATED FIX ====================
            # Use the calculated number of features instead of a hard-coded value.
            pooler = TopKPooling(in_channels=num_scalar_features, ratio=pooling_ratio)
            # ===================================================
            
            layer = HMP_Equiformer_Layer(
                local_mp=local_mp_block,
                global_mp=global_mp_block,
                pool=pooler
            )
            self.hmp_layers.append(layer)

        # --- 4. FINAL PREDICTION HEAD (Copied from Equiformer) ---
        self.final_irreps = self.irreps_node_embedding
        self.norm = torch.nn.LayerNorm(self.final_irreps.dim)

        # ==================== START OF THE FIX ====================

        # Create the list of activation functions.
        # We apply SiLU to the scalar part (0e) and identity (None) to all other parts.
        head_activations = []
        for mul, ir in self.final_irreps:
            if ir.l == 0: # Apply SiLU only to scalar types
                head_activations.append(torch.nn.SiLU())
            else: # Do nothing to vector/tensor types
                head_activations.append(None)

        self.head = torch.nn.Sequential(
            LinearRS(self.final_irreps, self.final_irreps), 
            # Pass the correctly sized list of activations
            Activation(self.final_irreps, acts=head_activations),
            LinearRS(self.final_irreps, o3.Irreps('1x0e'))) 
            
        # ===================== END OF THE FIX =====================
            
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)


    def forward(self, pos, batch, node_atom, f_in=None, **kwargs):
        # f_in is ignored, but included for compatibility with the existing training script.
        # --- ENABLE GRADIENTS FOR FORCE CALCULATION ---
        pos.requires_grad_(True)
        
        # --- 1. INITIAL GRAPH CONSTRUCTION & EMBEDDING ---
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch, max_num_neighbors=1000)
        edge_index = torch.stack([edge_src, edge_dst])
        
        # This initial embedding logic is identical to the start of Equiformer's forward pass.
        node_atom_mapped = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        atom_embedding, _, _ = self.atom_embed(node_atom_mapped)
        
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        
        # This is the initial rich feature vector for all nodes.
        node_features = atom_embedding + edge_degree_embedding

        # --- 2. HIERARCHICAL PROCESSING ---
        # Pass the data through our stack of HMP layers.
        """
        # ----- for QM9 -----
        for layer in self.hmp_layers:
            node_features, pos, edge_index, batch, node_atom = layer(
                node_features, pos, edge_index, batch, node_atom
            )
        """
        # ----- for MD17 -----
        for layer in self.hmp_layers:
            node_features, _, _, _, _ = layer( # We don't need the returned graph structure
                node_features, pos, edge_index, batch, node_atom
            )
            
        # --- 3. FINAL NODE-LEVEL PREDICTION ---
        # First, apply the final normalization to the node features.
        # Note: e3nn norm layers expect IrrepsTensor, so we use a standard one.
        # This might need refinement if results are poor.
        node_features = self.norm(node_features) 
        
        # Then, use the head to predict a single scalar energy value for EACH node.
        # This is the line that was missing or incorrect.
        node_energies = self.head(node_features)
        
        # Apply atomref if it exists (adds a baseline energy for each atom type).
        if self.atomref is not None:
            node_energies = node_energies + self.atomref[node_atom]
            
        # --- 4. ENERGY SUMMATION & FORCE CALCULATION ---
        # Sum the node energies to get the total system energy (still normalized).
        energy_normalized = self.scale_scatter(node_energies, batch, dim=0)

        # Compute forces by taking the negative gradient of the normalized energy.
        grad_outputs = [torch.ones_like(energy_normalized)]
        dy = -torch.autograd.grad(
            outputs=[energy_normalized],
            inputs=[pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Denormalize the final energy value for the output.
        #energy = energy_normalized * self.task_std + self.task_mean

        # ===================== END OF THE FIX =====================

        return energy_normalized, dy
    
# In file: hmp-net-clean/models/hmp/hmp_equiformer.py
# ... (keep all the existing code and imports) ...

# ===================================================================
# ADD THE NEW ABLATION MODEL CLASS BELOW THE EXISTING HMP_Equiformer_Net
# ===================================================================

class HMP_Equiformer_Net_Ablation(torch.nn.Module):
    """
    Ablation Baseline Model: A "flat" Equiformer with the same total number of 
    message-passing blocks as the HMP model, but with NO global MP step.
    This tests the value of the hierarchical structure itself.
    """
    def __init__(self,
                 num_total_blocks: int = 8, # Total number of MP blocks to stack
                 irreps_node_embedding='128x0e+64x1e+32x2e',
                 number_of_basis=128,
                 max_radius=5.0,
                 task_mean=0.0,
                 task_std=1.0,
                 atomref=None,
                 ):
        super().__init__()
        
        self.max_radius = max_radius
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(self.irreps_node_embedding.lmax)
        self.task_mean = task_mean
        self.task_std = task_std
        self.register_buffer('atomref', atomref)
        
        fc_neurons = [number_of_basis, 64, 64]

        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=self.max_radius)
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, fc_neurons, _AVG_DEGREE)
            
        # --- Simplified "Flat" Message Passing ---
        # Instead of HMP layers, we just stack the TransBlocks directly.
        self.blocks = torch.nn.ModuleList()
        for _ in range(num_total_blocks):
            # Each "block" is a single Equiformer TransBlock.
            mp_block = GraphAttentionTransformer(
                num_layers=1,
                irreps_node_embedding=self.irreps_node_embedding,
                irreps_feature=self.irreps_node_embedding,
                max_radius=self.max_radius,
                number_of_basis=number_of_basis,
                fc_neurons=fc_neurons,
                mean=self.task_mean,
                std=self.task_std,
            )
            # We only need the core TransBlock from this instance
            mp_block.head = torch.nn.Identity()
            mp_block.norm = torch.nn.Identity()
            self.blocks.append(mp_block.blocks[0]) # Append the TransBlock itself

        # --- Final Prediction Head (Identical to main model) ---
        self.final_irreps = self.irreps_node_embedding
        self.norm = torch.nn.LayerNorm(self.final_irreps.dim)
        head_activations = [torch.nn.SiLU() if ir.l == 0 else None for _, ir in self.final_irreps]
        self.head = torch.nn.Sequential(
            LinearRS(self.final_irreps, self.final_irreps), 
            Activation(self.final_irreps, acts=head_activations),
            LinearRS(self.final_irreps, o3.Irreps('1x0e')))
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)


    def forward(self, node_atom, pos, batch, **kwargs):
        pos.requires_grad_(True)
        
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch, max_num_neighbors=1000)
        
        node_atom_mapped = node_atom.new_tensor([-1, 0, -1, -1, -1, -1, 1, 2, 3, 4])[node_atom]
        atom_embedding, _, _ = self.atom_embed(node_atom_mapped)
        
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,
            x=edge_vec, normalize=True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        
        edge_degree_embedding = self.edge_deg_embed(atom_embedding, edge_sh, 
            edge_length_embedding, edge_src, edge_dst, batch)
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        # --- Simplified "Flat" Forward Pass ---
        for block in self.blocks:
            node_features = block(
                node_input=node_features, node_attr=node_attr, 
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh, 
                edge_scalars=edge_length_embedding, batch=batch
            )
        
        # --- Final Prediction (Identical to main model) ---
        node_features = self.norm(node_features)
        node_energies = self.head(node_features)
        
        if self.atomref is not None:
             node_energies = node_energies + self.atomref[node_atom]

        energy_normalized = self.scale_scatter(node_energies, batch, dim=0)

        grad_outputs = [torch.ones_like(energy_normalized)]
        dy = -torch.autograd.grad(
            outputs=[energy_normalized],
            inputs=[pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # energy = energy_normalized * self.task_std + self.task_mean

        return energy_normalized, dy