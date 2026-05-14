from __future__ import annotations

from aisteer360.algorithms.state_control.common.algorithms import (
    StepODESteer as StepODESteerCore,
)
from aisteer360.algorithms.state_control.common.wrapped_steer_control import (
    WrappedSteerControl,
)

from .args import StepODESteerArgs


class StepODESteer(WrappedSteerControl):
    Args = StepODESteerArgs
    _steer_cls = StepODESteerCore
    _steer_kwarg_names = (
        "n_components",
        "degree",
        "gamma",
        "coef0",
        "lin_clf_type",
    )
