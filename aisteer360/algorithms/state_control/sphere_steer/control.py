"""SphereSteer: hypersphere-geometry activation steering."""
from __future__ import annotations

import logging
from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control.common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control.common.selectors import (
    FixedLayerSelector,
    FractionalDepthSelector,
)
from aisteer360.algorithms.state_control.common.token_scope import (
    compute_prompt_lens,
    make_token_mask,
)

from .args import SphereSteerArgs
from .utils.data import SphereSteerCache
from .utils.estimator import SphereSteerEstimator
from .utils.geometry import geodesic_dist, sphere_steer_step

logger = logging.getLogger(__name__)


class SphereSteer(StateControl):
    r"""SphereSteer: geometry-aware activation steering on the hypersphere.

    After each RMSNorm operation the hidden states of a transformer layer
    concentrate on a hypersphere S = \{x \in \mathbb{R}^d : \|x\| = R\}.
    Standard additive methods (e.g. CAA) shift activations off this sphere,
    degrading the utility of subsequent computations.

    SphereSteer respects the sphere geometry throughout:

    **Offline phase (steer)**
        Given N contrastive pairs (x_i, y_i^-, y_i^+), extract last-token
        hidden states h_i^- / h_i^+ at the target layer and compute the
        contrastive log-map displacements

            \xi_i = \log_{h_i^-}(h_i^+)  \in  T_{h_i^-} S

        which encode the geodesic direction from each negative to its
        paired positive.  These are stored alongside h_i^-.

    **Online phase (forward hook)**
        For each hidden state h at inference time:

        1. Compute geodesic-RBF kernel weights

               w_i(h) = K_\sigma(d_S(h, h_i^-)) / \sum_{i'} K_\sigma(d_S(h, h_{i'}^-))
               K_\sigma(d) = \exp(-d^2 / 2\sigma^2)

        2. Parallel-transport each ξ_i to the tangent space at h

               \Gamma_{h_i^- \to h}(\xi_i) = \xi_i - (\xi_i^\top h)/(R^2 + h_i^- \cdot h) \cdot (h_i^- + h)

        3. Form the weighted tangent field and apply the exponential map

               v(h) = \alpha \cdot \sum_i w_i \cdot \Gamma_{h_i^- \to h}(\xi_i)
               f(h) = \mathrm{Exp}_h(v(h)) = \cos(\|v\|/R) \cdot h + \sin(\|v\|/R) \cdot (R/\|v\|) \cdot v

        The exponential map preserves the norm exactly: \|f(h)\| = \|h\| = R.
        The kernel weighting gives locality: if h already resembles a positive
        training point (h ≈ h_i^+), ξ_i ≈ 0 and no correction is applied.

    Args:
        cache (SphereSteerCache | None): Pre-computed offline data.
            Supply this to skip the training forward passes.
        data (ContrastivePairs | dict | None): Contrastive text pairs used to
            fit the cache during steer().  Required when cache is None.
        train_spec (VectorTrainSpec): Controls batch size for extraction.
        layer_id (int | None): Layer to intervene at.  Defaults to 40 % depth.
        alpha (float): Steering strength.  Scales v(h) before the exp map.
        sigma (float | None): RBF bandwidth.  None = auto (mean geodesic
            distance between training negatives).
        token_scope (TokenScope): Which token positions to steer.
        last_k (int | None): Used when token_scope == "last_k".
        from_position (int | None): Used when token_scope == "from_position".
    """

    Args = SphereSteerArgs
    supports_batching = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._cache: SphereSteerCache | None = None
        self._layer_ids: list[int] = []
        self._layer_names: list[str] = []
        self._sigmas: dict[int, float] = {}
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None

        # KV-cache position tracking
        self._position_offset: int = 0
        self._initial_seq_len: int = 0

    # ------------------------------------------------------------------
    # steer(): offline phase
    # ------------------------------------------------------------------

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Fit or load the SphereSteer cache, then prepare the forward hook.

        Args:
            model:     Base language model.
            tokenizer: Tokenizer (required when ``data`` is provided).

        Returns:
            The input model, unchanged.
        """
        device = next(model.parameters()).device
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        num_layers = len(layer_names)

        # ------------------------------------------------------------------
        # 1. Resolve target layers
        # ------------------------------------------------------------------
        if self.layer_id is not None:
            self._layer_ids = [
                FixedLayerSelector(lid).select(num_layers=num_layers)
                for lid in self.layer_id
            ]
        else:
            self._layer_ids = [FractionalDepthSelector(fraction=0.4).select(num_layers=num_layers)]

        # ------------------------------------------------------------------
        # 2. Fit or load cache
        # ------------------------------------------------------------------
        if self.cache is not None:
            cache = self.cache
        else:
            if tokenizer is None:
                raise ValueError(
                    "A tokenizer must be provided when SphereSteer is initialised with `data`."
                )
            estimator = SphereSteerEstimator()
            cache = estimator.fit(
                model,
                tokenizer,
                data=self.data,
                spec=self.train_spec,
                layer_ids=self._layer_ids,
            )

        cache.validate()
        self._cache = cache.to(device, dtype=model.dtype)

        # ------------------------------------------------------------------
        # 3. Resolve sigma (kernel bandwidth) per layer
        # ------------------------------------------------------------------
        for lid in self._layer_ids:
            if self.sigma is not None:
                self._sigmas[lid] = float(self.sigma)
            else:
                self._sigmas[lid] = self._auto_sigma(
                    self._cache.h_neg[lid],
                    self._cache.radius[lid],
                )
            logger.debug(
                "SphereSteer ready: layer=%d, N=%d, R=%.4f, sigma=%.4f, alpha=%.4f",
                lid,
                self._cache.h_neg[lid].shape[0],
                self._cache.radius[lid],
                self._sigmas[lid],
                self.alpha,
            )

        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
        return model

    # ------------------------------------------------------------------
    # get_hooks(): register forward hook
    # ------------------------------------------------------------------

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **__,
    ) -> dict[str, list]:
        """Create a forward hook that applies SphereSteer at the target layer.

        Args:
            input_ids:       Input token IDs.
            runtime_kwargs:  Runtime parameters (currently unused).

        Returns:
            Hook dict with "pre", "forward", "backward" keys.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)
        self._initial_seq_len = ids.size(1)
        self._position_offset = 0

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for lid in self._layer_ids:
            hooks["forward"].append({
                "module": self._layer_names[lid],
                "hook_func": partial(
                    self._forward_hook,
                    layer_id=lid,
                    cache=self._cache,
                    sigma=self._sigmas[lid],
                    alpha=self.alpha,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    prompt_lens=prompt_lens,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    control_ref=self,
                ),
            })
        return hooks

    # ------------------------------------------------------------------
    # Forward hook (static to avoid closure-reference issues)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_hook(
        module,
        args,
        kwargs,
        output,
        *,
        layer_id: int,
        cache: SphereSteerCache,
        sigma: float,
        alpha: float,
        gate,
        token_scope: str,
        prompt_lens: torch.LongTensor,
        last_k: int | None,
        from_position: int | None,
        control_ref: "SphereSteer",
    ):
        """Apply SphereSteer transformation to the layer output.

        Args:
            module:       The hooked transformer layer.
            args:         Forward positional args (unused).
            kwargs:       Forward keyword args (unused).
            output:       Layer output — either hidden_states tensor or
                          (hidden_states, ...) tuple.
            layer_id:     Index of the target layer.
            cache:        Pre-computed SphereSteerCache.
            sigma:        Kernel bandwidth.
            alpha:        Steering strength.
            gate:         AlwaysOpenGate instance.
            token_scope:  Which token positions to steer.
            prompt_lens:  Per-batch prompt lengths [B].
            last_k:       Used when token_scope == "last_k".
            from_position: Used when token_scope == "from_position".
            control_ref:  Reference to the SphereSteer instance for state.

        Returns:
            Modified output with the same structure as input.
        """
        # Unpack hidden states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if hidden is None:
            return output

        seq_len = hidden.size(1)

        # KV-cache position tracking (same logic as CAA)
        if seq_len < control_ref._initial_seq_len:
            position_offset = control_ref._position_offset
            control_ref._position_offset += seq_len
        else:
            position_offset = 0
            control_ref._position_offset = seq_len

        mask = make_token_mask(
            token_scope,
            seq_len=seq_len,
            prompt_lens=prompt_lens.to(hidden.device),
            last_k=last_k,
            from_position=from_position,
            position_offset=position_offset,
        )

        if gate.is_open() and mask.any():
            hidden = _apply_sphere_steer(
                hidden=hidden,
                mask=mask,
                cache=cache,
                layer_id=layer_id,
                sigma=sigma,
                alpha=alpha,
            )

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset position-tracking state between generation calls."""
        self._gate.reset()
        self._position_offset = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_sigma(h_neg: torch.Tensor, R: float) -> float:
        """Compute sigma as mean pairwise geodesic distance among h_neg.

        Uses a random subsample of at most 256 pairs to keep cost O(256²).

        Args:
            h_neg: Negative training activations [N, H].
            R:     Sphere radius.

        Returns:
            Scalar sigma (float).
        """
        N = h_neg.size(0)
        MAX_SUBSAMPLE = 256
        if N > MAX_SUBSAMPLE:
            idx = torch.randperm(N, device=h_neg.device)[:MAX_SUBSAMPLE]
            h_sub = h_neg[idx]
        else:
            h_sub = h_neg

        # pairwise geodesic distances
        # cos_mat[i,j] = h_sub[i] · h_sub[j] / R^2
        cos_mat = (h_sub @ h_sub.T) / (R ** 2)
        cos_mat = cos_mat.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        dist_mat = R * torch.acos(cos_mat)      # [K, K]

        # upper-triangle (exclude diagonal)
        K = h_sub.size(0)
        triu_mask = torch.triu(torch.ones(K, K, dtype=torch.bool, device=h_sub.device), diagonal=1)
        pairwise = dist_mat[triu_mask]

        sigma = pairwise.mean().item()
        if sigma < 1e-6:
            # Fallback: use mean norm (scale of activations)
            sigma = R * 0.1
        logger.debug("Auto-sigma: %.4f (from %d pairwise distances)", sigma, pairwise.numel())
        return sigma


# ---------------------------------------------------------------------------
# Module-level helper (keeps _forward_hook signature clean)
# ---------------------------------------------------------------------------

def _apply_sphere_steer(
    hidden: torch.Tensor,
    mask: torch.BoolTensor,
    cache: SphereSteerCache,
    layer_id: int,
    sigma: float,
    alpha: float,
) -> torch.Tensor:
    """Apply SphereSteer only to masked token positions.

    Args:
        hidden:   Full hidden states [B, T, H].
        mask:     Boolean mask [B, T]; True where steering should be applied.
        cache:    SphereSteerCache for the current layer.
        layer_id: Target layer index.
        sigma:    Kernel bandwidth.
        alpha:    Steering strength.

    Returns:
        Hidden states [B, T, H] with steered values at masked positions.
    """
    h_neg = cache.h_neg[layer_id]   # [N, H]
    xi = cache.xi[layer_id]         # [N, H]
    R = cache.radius[layer_id]

    # Work in float32 for numerical precision
    orig_dtype = hidden.dtype
    hidden_f = hidden.float()
    h_neg_f = h_neg.float()
    xi_f = xi.float()

    # If all positions are masked, steer the whole tensor at once
    if mask.all():
        steered = sphere_steer_step(hidden_f, h_neg_f, xi_f, R, sigma, alpha)
        return steered.to(orig_dtype)

    # Otherwise, gather masked positions, steer, scatter back
    # mask: [B, T]; find positions where mask is True
    B, T, H = hidden_f.shape
    # Flatten to [B*T, H] for easier indexing
    hidden_flat = hidden_f.view(B * T, H)
    mask_flat = mask.view(B * T)                       # [B*T]

    masked_h = hidden_flat[mask_flat].unsqueeze(0)     # [1, M, H]  (fake batch)
    steered_m = sphere_steer_step(masked_h, h_neg_f, xi_f, R, sigma, alpha)
    steered_m = steered_m.squeeze(0)                   # [M, H]

    out_flat = hidden_flat.clone()
    out_flat[mask_flat] = steered_m
    return out_flat.view(B, T, H).to(orig_dtype)
