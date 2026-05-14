from abc import abstractmethod
from typing import Literal

import torch
from torch import Tensor
from torchdiffeq import odeint

from aisteer360.algorithms.state_control.common._steer_base import Steer
from aisteer360.algorithms.state_control.common.kernels import (
    KernelClassifier,
    NormedPolyClassifier,
    RFFClassifier,
)


class BaseODESteer(Steer):
    def __init__(self, solver: Literal["euler", "midpoint", "rk4"] = "euler", steps: int = 10, **kwargs):
        super().__init__()
        self.solver = solver
        self.steps = steps
        self.clf = self._init_clf(**kwargs)

    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> "BaseODESteer":
        self.clf.fit(pos_X, neg_X_or_labels)
        return self

    @torch.no_grad()
    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        if T == 0.0:
            return X
        return odeint(
            func=lambda t, state: self.vector_field(state),
            y0=X,
            t=torch.tensor([0.0, T], device=X.device),
            method=self.solver,
            options={"step_size": T / self.steps},
        )[1]

    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        g = self.clf.grad(X)
        return g / (g.norm(dim=-1, keepdim=True) + 1e-10)

    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError


class ODESteer(BaseODESteer):
    def _init_clf(self, **kwargs) -> NormedPolyClassifier:
        return NormedPolyClassifier(**kwargs)


class RFFODESteer(BaseODESteer):
    def _init_clf(self, **kwargs) -> RFFClassifier:
        return RFFClassifier(**kwargs)
