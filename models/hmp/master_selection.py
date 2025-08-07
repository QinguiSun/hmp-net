'''
class MasterSelection(adjascent, features):
    def __init__(adjascent, features):
        pass
    def MasterSelection(self, features, adjascent):
        adjascent_mask = self.GumbelSoftmax()
        return adjascent_mask
    
    def GumbelSoftmax():
        pass
    '''
    
import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


class MasterSelection(nn.Module):
    """Differentiable master node selector using Gumbel-Softmax.

    This module scores each node with a small MLP and produces a binary
    selection mask.  During training the mask is obtained through a
    Straight-Through (ST) Gumbel-Softmax estimator so that gradients can
    flow through the discrete sampling operation.

    Parameters
    ----------
    in_dim: int
        Dimension of the input node features.
    hidden_dim: int
        Hidden size of the internal MLP used for computing selection
        logits.
    tau: float, optional (default=1.0)
        Initial temperature for the Gumbel-Softmax distribution.  Lower
        temperatures yield harder, more discrete samples.
    ratio: float, optional (default=None)
        If given, selects the top ``ratio`` fraction of nodes in the hard
        mask.  When ``ratio`` is ``None`` every node is compared against
        0.5 to decide whether it is selected.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        tau: float = 1.0,
        ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer('tau', torch.tensor(tau))
        self.ratio = ratio

    def forward(
        self,
        h: torch.Tensor,
        tau: Optional[float] = None,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select master nodes from a set of node features.

        Parameters
        ----------
        h: torch.Tensor, shape ``(n, d)``
            Invariant features for ``n`` nodes.
        tau: float, optional
            Temperature used for the Gumbel-Softmax.  If ``None`` the
            module's stored temperature is used.
        hard: bool, optional
            Whether to return a discrete mask using the Straight-Through
            estimator.  If ``False`` the soft probabilities are returned.

        Returns
        -------
        mask: torch.Tensor, shape ``(n,)``
            Binary (or soft) mask indicating which nodes are selected as
            masters.
        probs: torch.Tensor, shape ``(n,)``
            Selection probabilities for each node before sampling.
        """

        logits = self.mlp(h).squeeze(-1)
        probs = torch.sigmoid(logits)
        temperature = self.tau if tau is None else tau

        if self.training:
            # Sample Gumbel noise and compute the relaxed Bernoulli sample.
            gumbel = -torch.empty_like(logits).exponential_().log()
            y = torch.sigmoid((logits + gumbel) / temperature)
        else:
            y = probs

        if hard:
            if self.ratio is not None:
                k = max(1, int(math.ceil(self.ratio * h.size(0))))
                topk = torch.topk(y, k=k, dim=0)[1]
                hard_mask = torch.zeros_like(y)
                hard_mask[topk] = 1.0
            else:
                hard_mask = (y > 0.5).float()
            y = hard_mask - y.detach() + y
        return y, probs
