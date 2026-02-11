class HMP_Equiformer_Layer(torch.nn.Module):
    def __init__(self, equiformer_config_local, equiformer_config_global, pooling_config):
        super().__init__()
        
        # Local MP: Operates on original graph edges
        self.local_mp = GraphAttentionTransformer(**equiformer_config_local)
        
        # Pooling Layer: Selects master nodes
        self.pool = GumbelSoftmaxPooling(**pooling_config)
        
        # Global MP: Operates on a fully-connected graph of master nodes
        self.global_mp = GraphAttentionTransformer(**equiformer_config_global)
        
        # A projection layer for the "unpooling"/broadcast step   
        self.unpool_proj = LinearRS(...) # Or some other update mechanism

    def create_fully_connected_graph(self, indices, batch):
        # Helper function to create a dense edge_index for master nodes
        # ... implementation needed ...
        return edge_index_global
        
    def forward(self, h, pos, edge_index, batch):
        # 1. Local Message Passing
        # Note: We need to adapt the input format here!
        # This is why Step 1 is so important.
        h_local = self.local_mp(f_in=..., pos=pos, batch=batch, node_atom=...)
        
        # 2. Differentiable Pooling
        master_indices, master_mask = self.pool(h_local.tensor, batch)
        
        # If no nodes are selected, skip global MP
        if master_indices.numel() == 0:
            return h_local
            
        # 3. Prepare for Global Message Passing
        h_master = h_local[master_mask]
        pos_master = pos[master_mask]
        batch_master = batch[master_mask]
        
        # Create a new, fully-connected edge_index for the master nodes
        edge_index_global = self.create_fully_connected_graph(master_indices, batch_master)
        
        # 4. Global Message Passing
        # Again, adapt the input format
        h_global_out = self.global_mp(f_in=..., pos=pos_master, batch=batch_master, node_atom=...)

        # 5. Unpooling / Broadcast & Update
        # Project the refined master node features back
        h_update = self.unpool_proj(h_global_out)
        
        # Create the final output feature tensor using a residual connection
        h_out = h_local.clone()
        h_out.tensor[master_mask] += h_update.tensor # Or a more complex update
        
        return h_out