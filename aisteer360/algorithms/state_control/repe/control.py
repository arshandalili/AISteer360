from aisteer360.algorithms.state_control.common.algorithms import RepE as RepECore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import WrappedSteerControl

from .args import RepEArgs


class RepE(WrappedSteerControl):
    Args = RepEArgs
    _steer_cls = RepECore
    _steer_kwarg_names = ()
