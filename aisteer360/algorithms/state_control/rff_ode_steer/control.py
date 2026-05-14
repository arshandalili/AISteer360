from __future__ import annotations

from aisteer360.algorithms.state_control.common.algorithms import (
    RFFODESteer as RFFODESteerCore,
)
from aisteer360.algorithms.state_control.common.wrapped_steer_control import (
    WrappedSteerControl,
)

from .args import RFFODESteerArgs


class RFFODESteer(WrappedSteerControl):
    Args = RFFODESteerArgs
    _steer_cls = RFFODESteerCore
    _steer_kwarg_names = ("solver", "steps", "n_components", "sigma", "lin_clf_type")
