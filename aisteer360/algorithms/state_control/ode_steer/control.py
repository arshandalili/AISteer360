from __future__ import annotations

from aisteer360.algorithms.state_control.common.algorithms import ODESteer as ODESteerCore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import (
    WrappedSteerControl,
)

from .args import ODESteerArgs


class ODESteer(WrappedSteerControl):
    Args = ODESteerArgs
    _steer_cls = ODESteerCore
    _steer_kwarg_names = (
        "solver",
        "steps",
        "n_components",
        "degree",
        "gamma",
        "coef0",
        "lin_clf_type",
    )
