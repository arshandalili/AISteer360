from abc import abstractmethod

from torch import Tensor

from aisteer360.algorithms.state_control.common._steer_base import Steer
from aisteer360.algorithms.state_control.common.kernels import (
    KernelClassifier,
    NormedPolyClassifier,
    RFFClassifier,
)


class BaseStepODESteer(Steer):
    def __init__(self, **kwargs):
        super().__init__()
        self.clf = self._init_clf(**kwargs)

    def fit(self, pos_X: Tensor, neg_X_or_labels: Tensor) -> "BaseStepODESteer":
        self.clf.fit(pos_X, neg_X_or_labels)
        return self

    def steer(self, X: Tensor, T: float = 1.0) -> Tensor:
        return X + T * self.vector_field(X)

    def vector_field(self, X: Tensor) -> Tensor:
        self.clf.to(X.device)
        g = self.clf.grad(X)
        return g / (g.norm(dim=-1, keepdim=True) + 1e-10)

    @abstractmethod
    def _init_clf(self, **kwargs) -> KernelClassifier:
        raise NotImplementedError


class StepODESteer(BaseStepODESteer):
    def _init_clf(self, **kwargs) -> NormedPolyClassifier:
        return NormedPolyClassifier(**kwargs)


class RFFStepODESteer(BaseStepODESteer):
    def _init_clf(self, **kwargs) -> RFFClassifier:
        return RFFClassifier(**kwargs)
