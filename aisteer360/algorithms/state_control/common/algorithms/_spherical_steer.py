import torch
import torch.nn.functional as F
from torch import Tensor

from aisteer360.algorithms.state_control.common._steer_base import Steer


class SphericalSteer(Steer):
    def __init__(self, kappa: float = 20.0, alpha: float = 0.15, beta: float = 0.1):
        super().__init__()
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.mu_T = None
        self.mu_H = None
        self.cos_sim = None

    def fit(self, pos_X: Tensor, neg_X: Tensor):
        diff = pos_X.mean(dim=0) - neg_X.mean(dim=0)
        self.mu_T = F.normalize(diff, dim=0)
        self.mu_H = -self.mu_T
        self.cos_sim = torch.dot(self.mu_T, self.mu_H)
        return self

    def _rotate(self, x, mu_T, mu_H):
        orig_dtype = x.dtype
        x = x.float()
        mu_T = mu_T.float()
        mu_H = mu_H.float()

        orig_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        x_hat = x / orig_norm

        cos_T = (x_hat * mu_T.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
        cos_H = (x_hat * mu_H.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)

        logits = torch.stack([self.kappa * cos_T, self.kappa * cos_H], dim=-1)
        probs = F.softmax(logits, dim=-1)
        delta = probs[:, 1] - probs[:, 0]
        trigger = delta > self.beta

        out = x.clone()
        if trigger.any():
            t = (self.alpha * (delta - self.beta) / (1.0 - self.beta)).clamp(0.0, 1.0)
            theta = torch.acos(cos_T)
            valid = trigger & (theta >= 1e-4)
            if valid.any():
                theta_new = (1.0 - t) * theta
                sin_theta = torch.sin(theta).clamp_min(1e-12)
                u = (x_hat - cos_T.unsqueeze(-1) * mu_T.unsqueeze(0)) / sin_theta.unsqueeze(-1)
                cand = (torch.cos(theta_new).unsqueeze(-1) * mu_T.unsqueeze(0)
                        + torch.sin(theta_new).unsqueeze(-1) * u) * orig_norm
                out[valid] = cand[valid]
        return out.to(orig_dtype)

    def steer(self, X: Tensor, T: float = 1.0):
        return self._rotate(X, self.mu_T.to(X.device), self.mu_H.to(X.device))

    def vector_field(self, X: Tensor):
        return None
