import torch
import torch.nn.functional as F

class GumbelSoftmaxPooling(torch.nn.Module):
    def __init__(self, in_channels, pooling_ratio=0.5):
        super().__init__()
        self.pooling_ratio = pooling_ratio
        # A simple linear layer to predict a single "importance" score for each node
        self.score_net = torch.nn.Linear(in_channels, 1)

    def forward(self, x, batch, training=True):
        # x: [num_nodes, in_channels], batch: [num_nodes]
        
        node_scores = self.score_net(x) # [num_nodes, 1]
        
        # Use Gumbel-Softmax for differentiable sampling
        # 'hard=True' means forward pass is one-hot, backward pass uses softmax gradient
        gumbel_probs = F.gumbel_softmax(node_scores, tau=1.0, hard=True, dim=0)
        
        # In practice, gumbel_softmax is better for categories. 
        # A simpler approach for top-k might be better.
        # Let's try a simpler, learnable top-k approach.
        
        # --- 更适合图池化的可微Top-K方法 ---
        node_scores = self.score_net(x).squeeze(-1) # [num_nodes]
        
        # Add a random component for stochasticity, similar to Gumbel
        if self.training:
            random_noise = -torch.log(-torch.log(torch.rand_like(node_scores)))
            node_scores = node_scores + random_noise

        # For each graph in the batch, select the top-k nodes
        num_nodes_per_graph = torch.bincount(batch)
        num_to_pool_per_graph = (num_nodes_per_graph * self.pooling_ratio).ceil().to(torch.long)

        # A bit tricky to do this in a batched way. We can use a loop or advanced indexing.
        # The core idea is to get a mask of the master nodes.
        master_mask = torch.zeros_like(node_scores, dtype=torch.bool)
        
        current_idx = 0
        for i, num_nodes in enumerate(num_nodes_per_graph):
            num_to_pool = num_to_pool_per_graph[i]
            graph_scores = node_scores[current_idx : current_idx + num_nodes]
            
            # Find the indices of the top-k scores within this graph
            _, top_k_indices = torch.topk(graph_scores, k=num_to_pool)
            
            # Set the mask for these master nodes
            master_mask[current_idx + top_k_indices] = True
            current_idx += num_nodes
            
        master_indices = torch.where(master_mask)[0]
        
        return master_indices, master_mask