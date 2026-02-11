# You can create a new file, e.g., models/hmp/pool.py

import torch
from torch_scatter import scatter_max

class TopKPooling(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        # A simple linear layer to predict a single "importance" score for each node.
        # We use the scalar part of the input features for scoring.
        # Let's assume the first 128 features are scalars (0e).
        # We need to verify this assumption later.
        num_scalar_features = 128 # This should be a parameter
        self.score_net = torch.nn.Linear(num_scalar_features, 1)        # 这个分数是学出来的。从结构中推出来，会不会更好？

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        # x is the node_features tensor, shape [num_nodes, 480]
        # We only use the scalar part for scoring to maintain equivariance.
        x_scalar = x[:, :128] 
        scores = self.score_net(x_scalar).squeeze(-1) # Shape: [num_nodes]

        # Add a small random noise to break ties, helps with training stability.
        # This is a simpler alternative to Gumbel noise that works well for top-k.
        if self.training:
            noise = torch.rand_like(scores) * 1e-5
            scores = scores + noise

        # Batched top-k pooling
        # Calculate the number of nodes to keep for each graph
        num_nodes_per_graph = torch.bincount(batch)
        num_to_keep = (num_nodes_per_graph.float() * self.ratio).ceil().to(torch.long)

        # To perform batched top-k, we can use a clever trick with offsets
        # This avoids loops and is much faster on GPU
        # Add a large offset to scores of nodes in different graphs
        cumulative_nodes = torch.cat([batch.new_zeros(1), num_nodes_per_graph.cumsum(dim=0)])
        max_score = scores.max() * 2 
        # Create an offset that is different for each graph but constant within a graph
        offset = max_score * batch.float()
        
        # Get the top scores across the entire batch
        _, perm = torch.topk(scores + offset, k=scores.size(0))
        
        # Now, select the top 'num_to_keep' for each graph
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(num_nodes_per_graph.size(0)):
            graph_perm = perm[(batch[perm] == i)]
            graph_top_k = graph_perm[:num_to_keep[i]]
            mask[graph_top_k] = True

        master_indices = torch.where(mask)[0]
        
        return master_indices, mask

    def __repr__(self):
        return f'{self.__class__.__name__}(ratio={self.ratio})'