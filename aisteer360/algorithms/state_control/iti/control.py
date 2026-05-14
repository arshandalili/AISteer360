from aisteer360.algorithms.state_control.common.algorithms import ITI as ITICore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import WrappedSteerControl

from .args import ITIArgs


class ITI(WrappedSteerControl):
    Args = ITIArgs
    _steer_cls = ITICore
    _steer_kwarg_names = ()
