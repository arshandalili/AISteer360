"""SphereSteerEstimator: offline extraction of h_neg and ξ_i for SphereSteer."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.common.estimators.base import BaseEstimator
from aisteer360.algorithms.state_control.common.estimators.contrastive_direction_estimator import (
    _layerwise_tokenwise_hidden,
)
from aisteer360.algorithms.state_control.common.estimators.utils import (
    get_last_token_positions,
    select_at_positions,
    tokenize_pairs,
)
from aisteer360.algorithms.state_control.common.specs import ContrastivePairs, VectorTrainSpec

from .data import SphereSteerCache
from .geometry import log_map

logger = logging.getLogger(__name__)


class SphereSteerEstimator(BaseEstimator[SphereSteerCache]):
    r"""Offline estimator for the SphereSteer method.

    For each contrastive pair (x_i, y_i^-, y_i^+) and a specified layer l:
    1. Extracts last-token hidden states h_i^- and h_i^+ at layer l.
    2. Computes \xi_i = \log_{h_i^-}(h_i^+) — the tangent vector on the
       hypersphere that points from h_i^- to h_i^+.
    3. Stores h_i^-, \xi_i, and R = \mathrm{mean}(\|h_i^-\|) in a SphereSteerCache.

    The cache is used at inference time for kernel-weighted parallel
    transport and exponential-map steering.
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: ContrastivePairs,
        spec: VectorTrainSpec,
        layer_ids: list[int],
    ) -> SphereSteerCache:
        """Extract SphereSteer offline data for the given layers.

        Args:
            model:     Model to extract hidden states from.
            tokenizer: Tokenizer for encoding the contrastive pairs.
            data:      Contrastive (positive, negative) text pairs.
            spec:      Training configuration; only ``batch_size`` is used
                       (``accumulate`` is always ``"last_token"``).
            layer_ids: List of layer indices to extract data for.

        Returns:
            SphereSteerCache populated for each requested layer.
        """
        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")

        # ----------------------------------------------------------------
        # 1.  Build full texts from contrastive pairs
        # ----------------------------------------------------------------
        if data.prompts is not None:
            pos_texts = [p + c for p, c in zip(data.prompts, data.positives)]
            neg_texts = [p + c for p, c in zip(data.prompts, data.negatives)]
        else:
            pos_texts = list(data.positives)
            neg_texts = list(data.negatives)

        N = len(pos_texts)
        logger.debug(
            "SphereSteerEstimator: tokenising %d positive and %d negative examples", N, N
        )

        # ----------------------------------------------------------------
        # 2.  Tokenise pairs (interleaved for consistent padding)
        # ----------------------------------------------------------------
        enc_pos, enc_neg = tokenize_pairs(tokenizer, pos_texts, neg_texts, device)

        # ----------------------------------------------------------------
        # 3.  Extract hidden states for all layers in one pass
        # ----------------------------------------------------------------
        logger.debug(
            "SphereSteerEstimator: extracting hidden states (batch_size=%d)", spec.batch_size
        )
        hs_pos = _layerwise_tokenwise_hidden(model, enc_pos, batch_size=spec.batch_size)
        hs_neg = _layerwise_tokenwise_hidden(model, enc_neg, batch_size=spec.batch_size)

        # Attention masks (CPU) for position selection
        attn_pos = enc_pos.get("attention_mask")
        attn_neg = enc_neg.get("attention_mask")
        if attn_pos is not None:
            attn_pos = attn_pos.cpu()
        if attn_neg is not None:
            attn_neg = attn_neg.cpu()

        # ----------------------------------------------------------------
        # 4.  For each requested layer: select last-token activations,
        #     compute R and ξ_i = log_{h_i^-}(h_i^+)
        # ----------------------------------------------------------------
        cache = SphereSteerCache(model_type=model_type)

        for lid in layer_ids:
            if lid not in hs_pos or lid not in hs_neg:
                raise ValueError(
                    f"Layer {lid} not found in extracted hidden states "
                    f"(available: 0–{len(hs_pos)-1})."
                )

            hp = hs_pos[lid]   # [N, T, H]
            hn = hs_neg[lid]   # [N, T, H]

            # Select last non-pad token for each example
            pos_positions = get_last_token_positions(attn_pos, hp.size(1), N)
            neg_positions = get_last_token_positions(attn_neg, hn.size(1), N)

            h_pos_sel = select_at_positions(hp, pos_positions).float()  # [N, H]
            h_neg_sel = select_at_positions(hn, neg_positions).float()  # [N, H]

            # Sphere radius: mean norm of negative activations
            R = h_neg_sel.norm(dim=-1).mean().item()
            if R < 1e-6:
                raise ValueError(
                    f"Layer {lid}: computed sphere radius R={R:.4e} is too small. "
                    "Check that the model produces non-zero hidden states."
                )

            logger.debug("Layer %d: R=%.4f, N=%d", lid, R, N)

            # Log-map displacements: \xi_i = \log_{h_i^-}(h_i^+)
            xi = log_map(h_neg_sel, h_pos_sel, R)  # [N, H]

            cache.h_neg[lid] = h_neg_sel
            cache.xi[lid] = xi
            cache.radius[lid] = R

        logger.debug("SphereSteerEstimator: done. Layers extracted: %s", list(cache.h_neg.keys()))
        return cache
