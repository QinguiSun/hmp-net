# In file: hmp-net-clean/models/hmp/hmp_equiformer.py

# ... (keep all the imports as they are) ...

# Main network class - MODIFIED FOR MD17
class HMP_Equiformer_Net(torch.nn.Module):
    def __init__(self,
                 num_hmp_layers: int = 2,
                 pooling_ratio: float = 0.5,
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
        
        # --- 2. INITIAL EMBEDDING LAYERS ---
        # MD17 uses a different embedding strategy (atomic number, not one-hot)
        # For simplicity in the MVP, we will reuse the QM9 NodeEmbeddingNetwork
        # This is a reasonable starting point.
        self.atom_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _MAX_ATOM_TYPE)
        self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=self.max_radius)
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, 
            self.irreps_edge_attr, [number_of_basis, 64, 64], _AVG_DEGREE)
            
        # --- 3. HIERARCHICAL MESSAGE PASSING LAYERS ---
        self.hmp_layers = torch.nn.ModuleList()
        
        num_scalar_features = 0
        for mul, ir in self.irreps_node_embedding:
            if ir.l == 0 and ir.p == 1:
                num_scalar_features += mul
        
        if num_scalar_features == 0:
            raise ValueError(f"Irreps '{self.irreps_node_embedding}' must contain scalar features for pooling.")

        for _ in range(num_hmp_layers):
            shared_config = {
                'num_layers': 1,
                'irreps_node_embedding': self.irreps_node_embedding,
                'irreps_feature': self.irreps_node_embedding,
                'max_radius': self.max_radius,
                'number_of_basis': number_of_basis,
                'task_mean': self.task_mean, # Pass mean/std down
                'task_std': self.task_std,
            }
            local_mp_block = GraphAttentionTransformer(**shared_config)
            global_mp_block = GraphAttentionTransformer(**shared_config)
            
            local_mp_block.head = torch.nn.Identity()
            local_mp_block.norm = torch.nn.Identity()
            global_mp_block.head = torch.nn.Identity()
            global_mp_block.norm = torch.nn.Identity()
            
            pooler = TopKPooling(in_channels=num_scalar_features, ratio=pooling_ratio)
            
            layer = HMP_Equiformer_Layer(
                local_mp=local_mp_block,
                global_mp=global_mp_block,
                pool=pooler
            )
            self.hmp_layers.append(layer)

        # --- 4. FINAL PREDICTION HEAD ---
        self.final_irreps = self.irreps_node_embedding
        self.norm = torch.nn.LayerNorm(self.final_irreps.dim)
        
        head_activations = [torch.nn.SiLU() if ir.l == 0 else None for _, ir in self.final_irreps]
        self.head = torch.nn.Sequential(
            LinearRS(self.final_irreps, self.final_irreps), 
            Activation(self.final_irreps, acts=head_activations),
            LinearRS(self.final_irreps, o3.Irreps('1x0e')))
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)


    def forward(self, node_atom, pos, batch, **kwargs):
        # --- ENABLE GRADIENTS FOR FORCE CALCULATION ---
        pos.requires_grad_(True)
        
        # --- 1. INITIAL GRAPH CONSTRUCTION & EMBEDDING ---
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch, max_num_neighbors=1000)
        edge_index = torch.stack([edge_src, edge_dst])
        
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

        # --- 2. HIERARCHICAL PROCESSING ---
        for layer in self.hmp_layers:
            node_features, _, _, _, _ = layer( # We don't need the returned graph structure
                node_features, pos, edge_index, batch, node_atom
            )
        
        # --- 3. FINAL ENERGY PREDICTION ---
        node_features = self.norm(node_features)
        node_energies = self.head(node_features)
        
        if self.atomref is not None:
             node_energies = node_energies + self.atomref[node_atom]

        energy = self.scale_scatter(node_energies, batch, dim=0)

        # --- 4. FORCE CALCULATION ---
        grad_outputs = [torch.ones_like(energy)]
        dy = -torch.autograd.grad(
            outputs=[energy],
            inputs=[pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Denormalize energy prediction
        energy = energy * self.task_std + self.task_mean

        return energy, dy