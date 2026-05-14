from __future__ import annotations

from aisteer360.algorithms.state_control.common.algorithms import (
    RFFStepODESteer as RFFStepODESteerCore,
)
from aisteer360.algorithms.state_control.common.wrapped_steer_control import (
    WrappedSteerControl,
)

from .args import RFFStepODESteerArgs


class RFFStepODESteer(WrappedSteerControl):
    Args = RFFStepODESteerArgs
    _steer_cls = RFFStepODESteerCore
    _steer_kwarg_names = ("n_components", "sigma", "lin_clf_type")
