"""SphereSteer argument validation."""
from __future__ import annotations

from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.state_control.common.specs import (
    ContrastivePairs,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from aisteer360.algorithms.state_control.common.token_scope import TokenScope

from .utils.data import SphereSteerCache


@dataclass
class SphereSteerArgs(BaseArgs):
    """Arguments for SphereSteer (Hypersphere Steering).

    SphereSteer steers model behaviour by operating on the Riemannian
    hypersphere that LLM hidden states (post-RMSNorm) concentrate on.
    It computes a locally-adaptive tangent vector field via kernel-weighted
    parallel transport of contrastive log-map displacements, then applies the
    exponential map — preserving the hidden-state norm exactly.

    Users provide EITHER a pre-computed cache OR raw contrastive training data.
    If data is provided the cache is fitted during steer().

    Attributes:
        cache: Pre-computed SphereSteerCache. If provided, skip training.
        data:  Contrastive pairs for training. Required if cache is None.
        train_spec: Controls extraction method and batch size.
        layer_id:  Layer to apply steering at. If None, uses 40 % depth
            heuristic (same as CAA).
        alpha: Steering strength. Scales the tangent vector v(h) before the
            exponential map; larger values produce stronger rotation.
            Default 1.0.
        sigma: Geodesic RBF kernel bandwidth.  If None (default), auto-set to
            the mean pairwise geodesic distance among training negatives,
            which adapts to the scale of the activation space.
        token_scope: Which tokens to steer. "after_prompt" (default) steers
            only generated tokens.
        last_k: Required when token_scope == "last_k".
        from_position: Required when token_scope == "from_position".
    """

    # source (provide exactly one)
    cache: SphereSteerCache | None = None
    data: ContrastivePairs | dict | None = None

    # training configuration
    train_spec: VectorTrainSpec | dict = field(
        default_factory=lambda: VectorTrainSpec(method="mean_diff", accumulate="last_token")
    )

    # inference configuration
    layer_id: int | None = None
    alpha: float = 1.0
    sigma: float | None = None          # None → auto
    token_scope: TokenScope = "after_prompt"
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self) -> None:
        # exactly one of cache or data
        if self.cache is None and self.data is None:
            raise ValueError("Provide either `cache` or `data`.")
        if self.cache is not None and self.data is not None:
            raise ValueError("Provide `cache` or `data`, not both.")

        # validate cache
        if self.cache is not None:
            self.cache.validate()

        # normalise dict → ContrastivePairs
        if self.data is not None and not isinstance(self.data, ContrastivePairs):
            object.__setattr__(self, "data", as_contrastive_pairs(self.data))

        # normalise dict → VectorTrainSpec
        if isinstance(self.train_spec, dict):
            object.__setattr__(self, "train_spec", VectorTrainSpec(**self.train_spec))

        # parameter range checks
        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("`layer_id` must be >= 0.")
        if self.alpha <= 0:
            raise ValueError("`alpha` must be > 0.")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError("`sigma` must be > 0 when provided.")

        # token_scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("`last_k` must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (
            self.from_position is None or self.from_position < 0
        ):
            raise ValueError(
                "`from_position` must be >= 0 when token_scope is 'from_position'."
            )
