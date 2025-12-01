import torch
import numpy as np
import torch.nn.functional as F
import ot
class SlicedWassersteinCostMatrix:
    def __init__(self, radius=2, nofprojections=150, use_cuda=True):
        self.radius = radius
        self.nofprojections = nofprojections
        self.device = torch.device("cpu" )
        '''
            Mayor radio mayor amplifica las diferencias entre los puentos del espacio royectado. 
            aumenta sensibilidad) puede causar desbordamientos (overflow) o underflow en cálculos 
            con punto flotante.
            En proyecciones circulares, la idea del "círculo" es mantener un tamaño razonable 
            para capturar relaciones locales entre puntos. Un radio extremadamente grande puede 
            hacer que el círculo sea "tan grande" que actúe más como una proyección lineal que 
            como una circular.
        '''
    def compute_cost_matrix(self, X, Y, temp_prior= None):
        """
        Computes the pairwise cost matrix for the Sliced Wasserstein Distance
        with a circular defining function, supporting batched inputs.
        
        X: B x N x d tensor (batches of distributions)
        Y: M x d tensor (second distribution, no batches)
        

        
        Returns:
        Cost matrices of shape B x N x M.
        """
        B, N, d = X.shape
        M, dm = Y.shape
        assert dm == d, "X and Y must have the same feature dimension (d)."
        self.device = X.device
        # Generate random projections (theta)
        theta = self._random_projections(d)
        
        # Compute circular slices for each point in X and Y
        X_slices = self._circular_projection(X, theta)  # Shape: B x N x nofprojections
        Y_slices = self._circular_projection(Y.unsqueeze(0), theta)  # Shape: 1 x M x nofprojections
        
        X_slices = F.normalize(X_slices, p=2, dim=-1)
        Y_slices = F.normalize(Y_slices, p=2, dim=-1)
        
        projected_X = X_slices.sort(dim=1)[0]  # Ordenar por frames (B, T, n_projections)
        projected_Y = Y_slices.sort(dim=1)[0]
        
        # Compute pairwise cost matrices for each batch
        cost_matrix = torch.cdist(X_slices, Y_slices, p=2)  # Shape: B x N x M
        if temp_prior is not None: 
            cost_matrix += temp_prior
        cost_matrix_sort = torch.cdist(projected_X, projected_Y, p=2)  # Shape: B x N x M
        #return cost_matrix, self.get_opt(cost_matrix), cost_matrix_sort
        return cost_matrix, 0, cost_matrix_sort
    def get_opt(self, cost_matrix):
        B,T,K = cost_matrix.shape
        opt_codes = torch.zeros((B, T, K), device=self.device)
        dx = torch.ones((B, T,), device=self.device)/T
        dy = torch.ones((B, K,), device=self.device)/K
        for b in range(B):
            opt_codes[b] = ot.solve(cost_matrix[b].cpu(), dx[b].cpu(), dy[b].cpu(), reg=0.1, reg_type="l2", unbalanced=0.2, unbalanced_type="l2").plan #Unbalanced L2 with L2 Reg.
        return opt_codes

    def _random_projections(self, dim):
        """Generate random projections normalized to the unit sphere."""
        theta = torch.randn((self.nofprojections, dim), device=self.device)
        theta = torch.stack([t / torch.norm(t, p=2) for t in theta])  # Normalize
        return theta

    def _circular_projection(self, X, theta):
        """
        Circular projection: sqrt(sum((X - radius * theta)^2, axis=2)).
        X: B x N x d tensor (batches of distributions) or M x d tensor (second distribution)
        theta: nofprojections x d tensor
        """
        if len(theta.shape) == 1:
            return torch.sqrt(torch.sum((X - self.radius * theta) ** 2, dim=2))
        else:
            return torch.stack([torch.sqrt(torch.sum((X - self.radius * t) ** 2, dim=2)) for t in theta], dim=2)
