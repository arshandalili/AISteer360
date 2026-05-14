import torch
from torch import Tensor

from aisteer360.algorithms.state_control.common._steer_base import VecSteer


class CAA(VecSteer):
    @torch.no_grad()
    def fit(self, pos_X: Tensor, neg_X: Tensor) -> "CAA":
        self.steer_vec = pos_X.mean(dim=0) - neg_X.mean(dim=0)
        return self
