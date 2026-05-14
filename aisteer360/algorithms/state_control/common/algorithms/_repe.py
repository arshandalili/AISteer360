from sklearn.decomposition import PCA

import torch
from torch import Tensor

from aisteer360.algorithms.state_control.common._steer_base import VecSteer


class RepE(VecSteer):
    def __init__(self):
        super().__init__()
        self.pca = PCA(n_components=1)

    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> "RepE":
        if len(pos_X) != len(neg_X):
            n = min(len(pos_X), len(neg_X))
            pos_X, neg_X = pos_X[:n], neg_X[:n]
        diff = (pos_X - neg_X).detach().cpu()
        self.pca.fit(diff.numpy())
        self.steer_vec = torch.as_tensor(self.pca.components_[0], device=pos_X.device)
        return self
