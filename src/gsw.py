import torch
import numpy as np
import torch.nn.functional as F
import ot
class SlicedWassersteinCostMatrix:
    def __init__(self, radius=2, nofprojections=150, use_cuda=True):
        self.radius = radius
        self.nofprojections = nofprojections
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    def compute_cost_matrix(self, X, Y, temp_prior= None):
        """
        Computes the pairwise cost matrix for the Sliced Wasserstein Distance
        with a circular defining function, supporting batched inputs.
        
        X: B x N x d tensor (batches of distributions)
        Y: M x d tensor (second distribution, no batches)
        
        Returns:
        Cost matrix of shape B x N x M.
        """
        B, N, d = X.shape
        M, dm = Y.shape
        assert dm == d, "X and Y must have the same feature dimension (d)."
        self.device = X.device         
        # Random projections on the unit sphere
        theta = self._random_projections(d)

        # Circular slices
        X_slices = self._circular_projection(X, theta)          # (B, N, P)
        Y_slices = self._circular_projection(Y.unsqueeze(0), theta)  # (1, M, P)

        X_slices = F.normalize(X_slices, p=2, dim=-1)
        Y_slices = F.normalize(Y_slices, p=2, dim=-1)

        # Ordenar por frames
        projected_X = X_slices.sort(dim=1)[0]  # (B, N, P)
        projected_Y = Y_slices.sort(dim=1)[0]  # (1, M, P)

        # Distancias entre proyecciones
        cost_matrix = torch.cdist(X_slices, Y_slices, p=2)  # (B, N, M)
        if temp_prior is not None:
            cost_matrix = cost_matrix + temp_prior

        cost_matrix_sort = torch.cdist(projected_X, projected_Y, p=2)  # (B, N, M)
        return cost_matrix_sort
        
    def get_opt(self, cost_matrix: torch.Tensor):
        """
        Example of how to solve an OT plan given a cost matrix using POT.
        Not strictly necessary if only the cost matrix is used.
        """
        B, T, K = cost_matrix.shape
        opt_codes = torch.zeros((B, T, K), device=cost_matrix.device)
        dx = torch.ones((B, T), device=cost_matrix.device) / T
        dy = torch.ones((B, K), device=cost_matrix.device) / K

        for b in range(B):
            opt_codes[b] = ot.solve(
                cost_matrix[b].cpu(),
                dx[b].cpu(),
                dy[b].cpu(),
                reg=0.1,
                reg_type="l2",
                unbalanced=0.2,
                unbalanced_type="l2",
            ).plan

        return opt_codes

    def _random_projections(self, dim: int) -> torch.Tensor:
        """Generate random projections normalized to the unit sphere."""
        theta = torch.randn((self.nofprojections, dim), device=self.device)
        theta = theta / torch.norm(theta, p=2, dim=-1, keepdim=True)
        return theta

    def _circular_projection(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Circular projection: sqrt(sum((X - radius * theta)^2, axis=2)).

        X: B x N x d tensor
        theta: P x d tensor
        Returns:
            B x N x P tensor
        """
        if len(theta.shape) == 1:
            return torch.sqrt(torch.sum((X - self.radius * theta) ** 2, dim=2))
        else:
            return torch.stack(
                [torch.sqrt(torch.sum((X - self.radius * t) ** 2, dim=2)) for t in theta],
                dim=2,
            )
