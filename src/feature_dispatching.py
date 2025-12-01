import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDispatcher(nn.Module):
    """
    Implements the feature dispatching mechanism:
    - Computes cosine similarity between cluster centers C and frame features X
    - Builds a hard mask on the most similar cluster per frame
    - Aggregates and redistributes features to encourage monotonic alignment
    """

    def __init__(self, sim_alpha_init: float = 1.0, sim_beta_init: float = 0.0, learn_params: bool = True):
        super().__init__()
        self.sim_alpha = nn.Parameter(torch.ones(1) * sim_alpha_init, requires_grad=learn_params)
        self.sim_beta = nn.Parameter(torch.zeros(1) + sim_beta_init, requires_grad=learn_params)

    @staticmethod
    def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: (B, K, D), x2: (B, T, D) -> (B, K, T)
        return torch.matmul(x1, x2.transpose(-2, -1))

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        X: (B, T, D) frame features
        C: (B, K, D) cluster centers (typically self.clusters[None, ...])

        Returns:
            x: (B, T, D) refined features after dispatching
        """
        similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * self.pairwise_cos_sim(C, X))
        _, max_idx = similarity.max(dim=1, keepdim=True)
        mask = torch.zeros_like(similarity)
        mask.scatter_(1, max_idx, 1.0)

        similarity_out = similarity * mask
        # aggregate to cluster centers + residual C
        out = ((X.unsqueeze(1) * similarity_out.unsqueeze(-1)).sum(dim=2) + C) / (
            mask.sum(dim=-1, keepdim=True) + 1.0
        )
        # redistribute to frames
        out = (out.unsqueeze(2) * similarity.unsqueeze(-1)).sum(dim=1)
        x = X + out
        return x
