from sklearn.linear_model import LogisticRegression

import torch
from torch import Tensor

from aisteer360.algorithms.state_control.common._steer_base import VecSteer


class ITI(VecSteer):
    def __init__(self):
        super().__init__()
        self.clf = LogisticRegression(max_iter=1000)

    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> "ITI":
        if len(neg_X_or_labels.shape) == 1:
            self._fit_labels(pos_X, neg_X_or_labels)
        elif neg_X_or_labels.shape[1] == pos_X.shape[1]:
            Xs = torch.cat([pos_X, neg_X_or_labels], dim=0)
            labels = torch.cat([torch.ones(pos_X.shape[0]), torch.zeros(neg_X_or_labels.shape[0])])
            self._fit_labels(Xs, labels)
        else:
            raise ValueError("neg_X_or_labels must be 1D labels or 2D activations matching pos_X.")
        return self

    @torch.no_grad()
    def _fit_labels(self, Xs: Tensor, labels: Tensor) -> None:
        self.clf.fit(Xs.detach().cpu().numpy(), labels.detach().cpu().numpy())
        v = torch.as_tensor(self.clf.coef_.ravel(), device=Xs.device, dtype=Xs.dtype)
        self.steer_vec = v / v.norm().item()
