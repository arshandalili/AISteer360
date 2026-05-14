import torch
from torch import Tensor, nn

from aisteer360.algorithms.state_control.common._steer_base import Steer


class LinOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fitted = False

    def fit(self, X_target: Tensor, X_source: Tensor):
        n = min(len(X_target), len(X_source))
        X_target, X_source = X_target[:n], X_source[:n]
        m_t, m_s = X_target.mean(dim=0), X_source.mean(dim=0)
        s_t = (X_target - m_t).sort(dim=0).values
        s_s = (X_source - m_s).sort(dim=0).values
        w = torch.sum(s_t * s_s, dim=0) / (torch.sum(s_s ** 2, dim=0) + 1e-10)
        self.register_buffer("w", w)
        self.register_buffer("b", m_t - w * m_s)
        self.fitted = True
        return self

    def forward(self, X: Tensor) -> Tensor:
        return X * self.w + self.b


class LinAcT(Steer):
    def __init__(self):
        super().__init__()
        self.lin_ot = LinOT()

    def fit(self, pos_X: Tensor, neg_X: Tensor):
        self.lin_ot.fit(pos_X, neg_X)
        return self

    def steer(self, X: Tensor, T: float = 1.0):
        self.lin_ot.to(X.device)
        return self.lin_ot(X)

    def vector_field(self, X: Tensor):
        self.lin_ot.to(X.device)
        return self.lin_ot(X) - X
