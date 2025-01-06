import torch
import torch.linalg.eig as eig
import torch.linalg.pinv as pinv
from sklearn.metrics.pairwise import pairwise_kernels

class ELTO_MD:

    def __init__(self, trained_X_t, window_size, d, device='cuda', is_deep_kernel=False):
        super().__init__()

        self.device = device
        self.is_deep_kernel = is_deep_kernel
        if isinstance(trained_X_t, torch.Tensor):
            self.X1 = trained_X_t.T[:, :-1]
            self.X2 = trained_X_t.T[:, 1:]
        else:
            self.X1 = torch.tensor(trained_X_t.T[:, :-1], device=self.device)
            self.X2 = torch.tensor(trained_X_t.T[:, 1:], device=self.device)
    
    def compute_eigval(self):
        G = pairwise_kernels(self.X1, self.X1, metric='rbf', gamma=0.01)
        A = pairwise_kernels(self.X1, self.X2, metric='rbf', gamma=0.01)
        eigvals, _ = eig(pinv(G) @ A)
        return eigvals

