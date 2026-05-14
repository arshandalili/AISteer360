from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from aisteer360.evaluation.metrics.base import Metric


class GPT2XLPerplexity(Metric):
    def __init__(
        self,
        model_id: str = "gpt2-xl",
        model: PreTrainedModel | None = None,
        tokenizer=None,
        batch_size: int = 10,
        dtype: torch.dtype | None = None,
        min_tokens: int = 4,
        **extras: Any,
    ) -> None:
        super().__init__(**extras)
        self.name = "perplexity"
        # Skip outputs tokenizing to fewer than min_tokens — PPL on 1-2 token
        # responses is dominated by single rare bigrams and not meaningful.
        self.min_tokens = min_tokens
        if model is None:
            kwargs = {"device_map": "auto"}
            if dtype is not None:
                kwargs["torch_dtype"] = dtype
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._owns_model = True
        else:
            if tokenizer is None:
                raise ValueError("tokenizer required when model is supplied.")
            self._model = model
            self._tokenizer = tokenizer
            self._owns_model = False
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._batch_size = batch_size

    def free(self) -> None:
        if not self._owns_model:
            return
        try:
            self._model.to("meta")
        except Exception:
            pass
        self._model = None
        self._tokenizer = None
        from aisteer360.evaluation.use_cases.truthful_qa._memory import hard_free
        hard_free()

    @torch.no_grad()
    def _ppl_batch(self, texts: list[str]) -> list[float]:
        enc = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids: Tensor = enc.input_ids.to(self._model.device)
        attn: Tensor = enc.attention_mask.to(self._model.device)
        logits: Tensor = self._model(input_ids, attention_mask=attn, labels=input_ids).logits.detach()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attn[..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss(reduction="none")(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
        per_example = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        return torch.exp(per_example).tolist()

    def compute(self, responses=None, prompts=None, **kwargs: Any) -> dict[str, Any]:
        texts = [g["response"] for g in (responses or []) if g.get("response")]
        if self.min_tokens > 1:
            texts = [t for t in texts if len(self._tokenizer.encode(t)) >= self.min_tokens]
        if not texts:
            return {"perplexity": float("nan"), "ppls": []}
        ppls: list[float] = []
        for i in range(0, len(texts), self._batch_size):
            try:
                ppls.extend(self._ppl_batch(texts[i: i + self._batch_size]))
            except Exception:
                ppls.extend([float("nan")] * len(texts[i: i + self._batch_size]))
        ppls = [p for p in ppls if not np.isnan(p)]
        return {"perplexity": float(np.mean(ppls)) if ppls else float("nan"), "ppls": ppls}


def _dist_n(text: str, n: int) -> float:
    if not text:
        return 0.0
    words = text.strip().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


class DistinctN(Metric):
    def __init__(self, n: int = 1, **extras: Any) -> None:
        super().__init__(**extras)
        self.n = int(n)
        self.name = f"dist_{self.n}"

    def compute(self, responses=None, prompts=None, **kwargs: Any) -> dict[str, Any]:
        texts = [g["response"] for g in (responses or [])]
        scores = [_dist_n(t, self.n) for t in texts]
        return {f"dist_{self.n}": float(np.mean(scores)) if scores else 0.0, "scores": scores}
