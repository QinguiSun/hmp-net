# virtual_generation.py
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation function.

    Projects ``logits`` onto the probability simplex resulting in sparse
    probabilities.  Implementation follows Martins & Astudillo (2016).
    """

    logits = logits.transpose(dim, -1)
    z = logits - logits.max(dim=-1, keepdim=True).values
    sort_z, _ = torch.sort(z, descending=True, dim=-1)
    cssv = sort_z.cumsum(dim=-1) - 1
    range_ = torch.arange(
        1, sort_z.size(-1) + 1, device=logits.device, dtype=logits.dtype
    )
    cond = sort_z > cssv / range_
    k = cond.sum(dim=-1, keepdim=True)
    tau = cssv.gather(-1, k - 1) / k
    out = torch.clamp(z - tau, min=0)
    return out.transpose(dim, -1)


class VirtualGeneration(nn.Module):
    """Generate virtual edges between master nodes via attention.

    Parameters
    ----------
    in_dim: int
        Dimension of the scalar node features used for computing
        attention scores.
    lambda_attn: float, optional (default=0.0)
        Weight applied to the induced adjacency matrix when combining
        with the learned attention scores.
    negative_slope: float, optional (default=0.2)
        Negative slope for the LeakyReLU used in attention computation.
    """

    def __init__(
        self,
        in_dim: int,
        lambda_attn: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.lambda_attn = lambda_attn
        self.proj = nn.Linear(2 * in_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.phi = nn.Linear(in_dim, 2*in_dim, bias=False)  # 共享

    def forward(
        self,
        s: torch.Tensor,
        adj_induced: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a sparse, invariant virtual adjacency matrix.

        Parameters
        ----------
        s: torch.Tensor, shape ``(m, d)``
            Scalar features for ``m`` master nodes.
        adj_induced: torch.Tensor, optional, shape ``(m, m)``
            Adjacency matrix induced from the original molecular graph.

        Returns
        -------
        torch.Tensor, shape ``(m, m)``
            Row-normalised sparse adjacency matrix representing the
            generated virtual edges.
        """

        m = s.size(0)
        s_i = s.unsqueeze(1).expand(-1, m, -1)
        s_j = s.unsqueeze(0).expand(m, -1, -1)
        # 有向的分数 score 
        pair = torch.cat([s_i, s_j], dim=-1)
        scores = self.leaky_relu(self.proj(pair).squeeze(-1))
                
        if adj_induced is not None:
            scores = scores + self.lambda_attn * adj_induced
        attn = sparsemax(scores, dim=-1)
        #attn = torch.softmax(scores, dim=-1)
        attn = attn * (1 - torch.eye(m, device=s.device))
        return attn

    