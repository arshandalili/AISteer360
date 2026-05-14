from __future__ import annotations

from functools import partial

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import HookSpec, StateControl
from aisteer360.algorithms.state_control.common._steer_base import Steer
from aisteer360.algorithms.state_control.common.hook_utils import get_model_layer_list


def _split_hidden(output):
    if isinstance(output, tuple):
        hidden, rest = output[0], output[1:]
        return hidden, lambda h: (h, *rest)
    if hasattr(output, "last_hidden_state"):
        def reassemble(h):
            output.last_hidden_state = h
            return output
        return output.last_hidden_state, reassemble
    return output, (lambda h: h)


class WrappedSteerControl(StateControl):
    supports_batching = True

    _steer_cls: type[Steer] = None
    _steer_kwarg_names: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steer: Steer | None = None
        self._layer_names: list[str] = []
        self._layer_id: int = 0

    def _build_steer(self) -> Steer:
        return self._steer_cls(**{n: getattr(self, n) for n in self._steer_kwarg_names})

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        *,
        pos_activations: Tensor | None = None,
        neg_activations: Tensor | None = None,
        **__,
    ) -> None:
        _, self._layer_names = get_model_layer_list(model)
        self._layer_id = int(self.layer_id)
        self._steer = self._build_steer()
        if pos_activations is not None and neg_activations is not None:
            self._steer.fit(
                pos_activations.detach().to(torch.float32).cpu(),
                neg_activations.detach().to(torch.float32).cpu(),
            )

    def get_hooks(self, input_ids, runtime_kwargs=None, **__) -> dict[str, list[HookSpec]]:
        return {
            "pre": [],
            "forward": [{
                "module": self._layer_names[self._layer_id],
                "hook_func": partial(self._forward_hook, steer=self._steer, T=float(getattr(self, "T", 1.0))),
            }],
            "backward": [],
        }

    @staticmethod
    def _forward_hook(module, args, kwargs, output, *, steer: Steer, T: float):
        hidden, reassemble = _split_hidden(output)
        if hidden is None or steer is None:
            return output
        last = hidden[:, -1, :]
        steered = steer.steer(last.to(torch.float32), T=T).to(hidden.dtype)
        new = hidden.clone()
        new[:, -1, :] = steered
        return reassemble(new)
