from __future__ import annotations

from aisteer360.algorithms.state_control.common.algorithms import COBRAS as COBRASCore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import (
    WrappedSteerControl,
)

from .args import COBRASArgs


class COBRAS(WrappedSteerControl):
    Args = COBRASArgs
    _steer_cls = COBRASCore
    _steer_kwarg_names = (
        "k_bw",
        "n_sinkhorn",
        "alpha_sigma",
        "epsilon",
        "max_iters",
        "vmf_kappa",
        "vmf_beta",
    )
