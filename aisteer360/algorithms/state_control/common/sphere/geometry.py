from __future__ import annotations

import torch
from torch import Tensor


_EPS = 1e-7
_SAME_POINT_THR = 1e-8


def geodesic_dist(p: Tensor, q: Tensor, R: float) -> Tensor:
    cos_theta = (p * q).sum(dim=-1) / (R ** 2)
    cos_theta = cos_theta.clamp(-1.0 + _EPS, 1.0 - _EPS)
    return R * torch.acos(cos_theta)


def log_map(p: Tensor, q: Tensor, R: float) -> Tensor:
    cos_theta = (p * q).sum(dim=-1, keepdim=True) / (R ** 2)
    cos_theta = cos_theta.clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    coeff = torch.where(
        theta.abs() < _SAME_POINT_THR,
        torch.ones_like(theta),
        theta / sin_theta.clamp(min=_SAME_POINT_THR),
    )
    tangent = coeff * (q - cos_theta * p)
    return torch.where(theta.abs() < _SAME_POINT_THR, torch.zeros_like(tangent), tangent)


def exp_map(p: Tensor, v: Tensor, R: float) -> Tensor:
    v_norm = v.norm(dim=-1, keepdim=True)
    angle = v_norm / R
    safe = v_norm.clamp(min=_SAME_POINT_THR)
    v_unit = v / safe
    p_norm = p.norm(dim=-1, keepdim=True)
    result = torch.cos(angle) * p + torch.sin(angle) * p_norm * v_unit
    return torch.where(v_norm < _SAME_POINT_THR, p, result)


def knn_geodesic_dist(H: Tensor, R: float, k: int) -> Tensor:
    k = min(k, H.size(0) - 1)
    cos_mat = (H @ H.T) / (R ** 2)
    cos_mat = cos_mat.clamp(-1.0 + _EPS, 1.0 - _EPS)
    dist_mat = R * torch.acos(cos_mat)
    dist_mat.fill_diagonal_(float("inf"))
    kth, _ = dist_mat.topk(k, dim=1, largest=False)
    return kth[:, -1]
