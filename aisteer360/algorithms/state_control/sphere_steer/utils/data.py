"""SphereSteerCache: per-layer storage of training activations and displacements."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SphereSteerCache:
    r"""Per-layer data extracted during the offline (steer) phase of SphereSteer.

    For each targeted layer l the cache stores:
    - h_neg[l]  : negative training activations  [N, H]
    - xi[l]     : log-map displacements \xi_i = \log_{h_i^-}(h_i^+)  [N, H]
    - radius[l] : mean norm R = \mathrm{mean}_i \|h_i^-\|  (scalar)

    These tensors are loaded at inference time by the forward hook.

    Attributes:
        model_type: HuggingFace model_type string (e.g. "llama").
        h_neg: Dict mapping layer_id → tensor [N, H] of negative activations.
        xi:    Dict mapping layer_id → tensor [N, H] of tangent displacements.
        radius: Dict mapping layer_id → float sphere radius for that layer.
    """

    model_type: str
    h_neg: dict[int, torch.Tensor] = field(default_factory=dict)
    xi: dict[int, torch.Tensor] = field(default_factory=dict)
    radius: dict[int, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str, dtype: torch.dtype) -> "SphereSteerCache":
        """Return a new cache with all tensors moved to *device* and cast to *dtype*."""
        return SphereSteerCache(
            model_type=self.model_type,
            h_neg={lid: t.to(device=device, dtype=dtype) for lid, t in self.h_neg.items()},
            xi={lid: t.to(device=device, dtype=dtype) for lid, t in self.xi.items()},
            radius=dict(self.radius),
        )

    def validate(self) -> None:
        """Raise ValueError if the cache is not properly populated."""
        if not self.model_type:
            raise ValueError("SphereSteerCache.model_type must be set.")
        if not self.h_neg:
            raise ValueError("SphereSteerCache.h_neg is empty.")
        if self.h_neg.keys() != self.xi.keys():
            raise ValueError("SphereSteerCache.h_neg and xi must have the same layer keys.")
        if self.h_neg.keys() != self.radius.keys():
            raise ValueError("SphereSteerCache.h_neg and radius must have the same layer keys.")
        for lid, hn in self.h_neg.items():
            xi = self.xi[lid]
            if hn.shape != xi.shape:
                raise ValueError(
                    f"Layer {lid}: h_neg shape {hn.shape} != xi shape {xi.shape}."
                )
            if hn.ndim != 2:
                raise ValueError(f"Layer {lid}: expected 2-D tensors [N, H], got {hn.ndim}-D.")
            R = self.radius[lid]
            if R <= 0:
                raise ValueError(f"Layer {lid}: radius must be > 0, got {R}.")
