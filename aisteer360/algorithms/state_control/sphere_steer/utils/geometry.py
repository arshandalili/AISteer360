r"""Riemannian geometry primitives for the hypersphere S^{d-1}_R.

All operations act on a hypersphere of radius R (i.e.
S = \{x \in \mathbb{R}^d : \|x\| = R\}).
Every function is numerically stable via explicit clamping / torch.where guards.

References
----------
- Turner et al., "Activation Addition", 2023 (standard additive baseline)
- "Hypersphere Steering" paper (this implementation)
- Absil, Mahony & Sepulchre, "Optimization Algorithms on Matrix Manifolds", 2007
"""
from __future__ import annotations

import torch

# Small epsilon for numerical stability (arccos / asin domains)
_EPS = 1e-7
# Threshold below which we treat two points as identical
_SAME_POINT_THR = 1e-8


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------

def geodesic_dist(p: torch.Tensor, q: torch.Tensor, R: float) -> torch.Tensor:
    r"""Geodesic (arc-length) distance on S^{d-1}_R between p and q.

    d_S(p, q) = R \cdot \arccos(\mathrm{clamp}(p^\top q / R^2,\, -1+\varepsilon,\, 1-\varepsilon))

    Args:
        p: Tensor of shape [..., H].
        q: Tensor of shape [..., H] (broadcast-compatible with p).
        R: Sphere radius (scalar > 0).

    Returns:
        Tensor of shape [...] containing distances.
    """
    cos_theta = (p * q).sum(dim=-1) / (R ** 2)
    cos_theta = cos_theta.clamp(-1.0 + _EPS, 1.0 - _EPS)
    return R * torch.acos(cos_theta)


# ---------------------------------------------------------------------------
# Logarithmic map
# ---------------------------------------------------------------------------

def log_map(p: torch.Tensor, q: torch.Tensor, R: float) -> torch.Tensor:
    r"""Riemannian logarithmic map \log_p(q) \in T_p S.

    Returns the tangent vector at p whose geodesic flow reaches q in unit time.

        \theta = \arccos(p^\top q / R^2)
        \log_p(q) = (\theta / \sin\theta) \cdot (q - \cos\theta \cdot p)

    When p \approx q\ (\theta \approx 0) the formula is 0/0 — we return the
    zero vector.

    Args:
        p: Base point, shape [..., H], on the sphere (\|p\| \approx R).
        q: Target point, shape [..., H] (broadcast-compatible with p).
        R: Sphere radius.

    Returns:
        Tangent vector \log_p(q), shape [..., H].
        Satisfies (\mathrm{result})^\top p = 0.
    """
    cos_theta = (p * q).sum(dim=-1, keepdim=True) / (R ** 2)
    cos_theta = cos_theta.clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(cos_theta)           # shape [..., 1]
    sin_theta = torch.sin(theta)            # shape [..., 1]

    # coefficient = \theta / \sin\theta  (\to 1 as \theta \to 0, so numerically safe with where)
    coeff = torch.where(
        theta.abs() < _SAME_POINT_THR,
        torch.ones_like(theta),
        theta / sin_theta.clamp(min=_SAME_POINT_THR),
    )

    tangent = coeff * (q - cos_theta * p)

    # Zero out when p \approx q
    return torch.where(
        theta.abs() < _SAME_POINT_THR,
        torch.zeros_like(tangent),
        tangent,
    )


# ---------------------------------------------------------------------------
# Exponential map
# ---------------------------------------------------------------------------

def exp_map(h: torch.Tensor, v: torch.Tensor, R: float) -> torch.Tensor:
    r"""Riemannian exponential map \mathrm{Exp}_h(v) on S^{d-1}_R.

    Rotates h in the direction v by geodesic angle \|v\| / R.  The formula is
    the standard one from the paper scaled to be exactly norm-preserving for
    any h (not just those with \|h\| = R):

        \mathrm{Exp}_h(v) = \cos(\|v\|/R) \cdot h
                          + \sin(\|v\|/R) \cdot (\|h\| / \|v\|) \cdot v

    Proof of norm preservation:
        \|result\|^2 = \cos^2(\cdot)\|h\|^2 + \sin^2(\cdot)\|h\|^2 = \|h\|^2

    When h is on the sphere (\|h\| = R) this reduces to the paper's formula:
        \cos(\|v\|/R) \cdot h + \sin(\|v\|/R) \cdot (R/\|v\|) \cdot v

    When \|v\| \approx 0 we return h unchanged (no rotation).

    Args:
        h: Base point, shape [..., H].  Need not be exactly on S_R.
        v: Tangent vector at h (v^\top h \approx 0), shape [..., H].
        R: Sphere radius (controls rotation angle, not output norm).

    Returns:
        Rotated point, shape [..., H], with \|result\| = \|h\| exactly.
    """
    v_norm = v.norm(dim=-1, keepdim=True)           # [..., 1]
    angle = v_norm / R                               # [..., 1]

    # Safe normalised direction: v / \|v\| (avoid 0/0)
    safe_v_norm = v_norm.clamp(min=_SAME_POINT_THR)
    v_unit = v / safe_v_norm                         # [..., H]

    # Use \|h\| as scale so \|result\| = \|h\| for any h
    h_norm = h.norm(dim=-1, keepdim=True)            # [..., 1]
    result = torch.cos(angle) * h + torch.sin(angle) * h_norm * v_unit

    # If \|v\| is negligible, return h unchanged
    return torch.where(v_norm < _SAME_POINT_THR, h, result)


# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------

def parallel_transport_batch(
    w: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    R: float,
) -> torch.Tensor:
    r"""Parallel-transport tangent vectors w \in T_p S along geodesic p \to q.

    Formula (exact on sphere):
        \Gamma_{p \to q}(w) = w - (w^\top q / (R^2 + p^\top q)) \cdot (p + q)

    Properties guaranteed:
    - \|\Gamma_{p \to q}(w)\| = \|w\|  (isometry)
    - \Gamma_{p \to q}(w)^\top q = 0   (result lies in T_q S)

    This batched version handles:
        w : [N, H]   — N training tangent vectors
        p : [N, H]   — N training base-points (h_i^-)
        q : [B, T, H]— B x T query hidden states
    and returns the transported vectors at each query position.

    To avoid materialising an [B, T, N, H] tensor we decompose the result:

        \Gamma(w)[b,t,n] = w[n] - \mathrm{coeff}[b,t,n] \cdot (p[n] + q[b,t])

    and compute the **weighted sum over n** inline (caller supplies weights):

        v[b,t] = \sum_n \mathrm{weight}[b,t,n] \cdot \Gamma(w)[b,t,n]
               = \sum_n \mathrm{weight}[b,t,n] \cdot w[n]
               - \sum_n (\mathrm{weight}[b,t,n] \cdot \mathrm{coeff}[b,t,n]) \cdot p[n]
               - q[b,t] \cdot \sum_n \mathrm{weight}[b,t,n] \cdot \mathrm{coeff}[b,t,n]

    So this function returns the coefficient tensor transport_coeff [B, T, N]
    rather than the full [B, T, N, H] result.  The caller does the einsum.

    Args:
        w: Tangent vectors to transport, shape [N, H].
        p: Source base-points, shape [N, H].
        q: Target points, shape [B, T, H].
        R: Sphere radius.

    Returns:
        transport_coeff: shape [B, T, N].
            \mathrm{coeff}[b,t,n] = (w[n]^\top q[b,t]) / (R^2 + p[n]^\top q[b,t])
    """
    # w_dot_q[b,t,n] = w[n] \cdot q[b,t]
    w_dot_q = torch.einsum("nh,bth->btn", w, q)          # [B, T, N]

    # p_dot_q[b,t,n] = p[n] \cdot q[b,t]
    p_dot_q = torch.einsum("nh,bth->btn", p, q)          # [B, T, N]

    denom = R ** 2 + p_dot_q                              # [B, T, N]
    denom = denom.clamp(min=_SAME_POINT_THR)

    return w_dot_q / denom                                # [B, T, N]


# ---------------------------------------------------------------------------
# Full weighted steering step (vectorised, O(B·T·N·H) free)
# ---------------------------------------------------------------------------

def sphere_steer_step(
    h: torch.Tensor,
    h_neg: torch.Tensor,
    xi: torch.Tensor,
    R: float,
    sigma: float,
    alpha: float,
) -> torch.Tensor:
    r"""Apply one SphereSteer step to hidden states h.

    Computes the kernel-weighted average of parallel-transported log-map
    displacements and applies the exponential map:

        v(h) = \alpha \cdot \sum_i w_i(h) \cdot \Gamma_{h_i^- \to h}(\xi_i)
        f(h) = \mathrm{Exp}_h(v(h))

    **Exact norm preservation** for arbitrary h:
    All Riemannian operations (geodesic distance, parallel transport, log/exp
    maps) are defined for points exactly on S_R.  To handle hidden states whose
    norms may deviate slightly from R (e.g. due to residual-stream growth), we
    project h onto S_R before the geometric computation, apply the steering,
    and scale the result back to the original norm \|h\|:

        \hat{h} = h \cdot R / \|h\|             (project onto S_R)
        v(\hat{h}) = \text{geometric tangent field at } \hat{h}
        \hat{f} = \mathrm{Exp}_{\hat{h}}(v(\hat{h}))   (on S_R,\; \|\hat{f}\| = R)
        f = \hat{f} \cdot \|h\| / R             (restore original norm,\; \|f\| = \|h\|)

    Memory-efficient: never allocates [B, T, N, H].  Intermediate tensors
    are at most [B, T, N].

    Args:
        h:      Hidden states, shape [B, T, H].  \|h\| need not equal R.
        h_neg:  Negative training activations, shape [N, H].  On S_R.
        xi:     Log-map displacements (\xi_i = \log_{h_i^-}(h_i^+)),
                shape [N, H].  Tangent vectors on S_R.
        R:      Sphere radius (mean \|h_\mathrm{neg}\| from training).
        sigma:  Kernel bandwidth (geodesic RBF).
        alpha:  Steering strength scalar.

    Returns:
        Steered hidden states, shape [B, T, H].
        \|result[b,t]\| = \|h[b,t]\|  (exact, regardless of \|h\| vs R).
    """
    B, T, H = h.shape
    dtype = h.dtype
    device = h.device

    h_neg = h_neg.to(device=device, dtype=dtype)
    xi = xi.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # 0. Project h onto S_R for exact geometry, remember scale to restore
    # ------------------------------------------------------------------
    h_norm = h.norm(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR)  # [B, T, 1]
    h_hat = h * (R / h_norm)                                           # [B, T, H], on S_R

    # ------------------------------------------------------------------
    # 1. Geodesic kernel weights w_i(\hat{h})  [B, T, N]
    # ------------------------------------------------------------------
    # cos of geodesic angle: \hat{h}[b,t] \cdot h_neg[n] / R^2
    cos_ang = torch.einsum("nh,bth->btn", h_neg, h_hat) / (R ** 2)  # [B, T, N]
    cos_ang = cos_ang.clamp(-1.0 + _EPS, 1.0 - _EPS)
    theta = torch.acos(cos_ang)                                       # [B, T, N]
    dist = R * theta                                                   # [B, T, N]

    # RBF kernel; subtract max for numerical stability (log-sum-exp trick)
    log_k = -(dist ** 2) / (2.0 * sigma ** 2)                        # [B, T, N]
    log_k = log_k - log_k.amax(dim=-1, keepdim=True)                 # stabilise
    kernel = torch.exp(log_k)                                         # [B, T, N]
    weights = kernel / kernel.sum(dim=-1, keepdim=True).clamp(min=_SAME_POINT_THR)

    # ------------------------------------------------------------------
    # 2. Parallel-transport coefficient at \hat{h}:
    #    (\xi_i \cdot \hat{h}) / (R^2 + h_{\mathrm{neg},i} \cdot \hat{h})
    # ------------------------------------------------------------------
    transport_coeff = parallel_transport_batch(xi, h_neg, h_hat, R)  # [B, T, N]

    # weighted transport coefficient  [B, T, N]
    wtc = weights * transport_coeff

    # ------------------------------------------------------------------
    # 3. Weighted tangent field at \hat{h}
    #    v(\hat{h}) = \sum_n w_n \cdot \xi_n
    #               - \sum_n \mathrm{wtc}_n \cdot h_{\mathrm{neg},n}
    #               - \hat{h} \cdot \sum_n \mathrm{wtc}_n
    # ------------------------------------------------------------------
    v = (
        torch.einsum("btn,nh->bth", weights, xi)        # \sum w \cdot \xi
        - torch.einsum("btn,nh->bth", wtc, h_neg)       # \sum \mathrm{wtc} \cdot h_\mathrm{neg}
        - h_hat * wtc.sum(dim=-1, keepdim=True)          # \sum \mathrm{wtc} \cdot \hat{h}
    )                                                    # [B, T, H]

    v = alpha * v

    # ------------------------------------------------------------------
    # 4. Exponential map at \hat{h}: \hat{f} = \mathrm{Exp}_{\hat{h}}(v),
    #    \|\hat{f}\| = R
    # ------------------------------------------------------------------
    f_hat = exp_map(h_hat, v, R)   # [B, T, H], \|f_hat\| = R

    # ------------------------------------------------------------------
    # 5. Scale back to original norm: f = \hat{f} \cdot \|h\| / R
    # ------------------------------------------------------------------
    return f_hat * (h_norm / R)
