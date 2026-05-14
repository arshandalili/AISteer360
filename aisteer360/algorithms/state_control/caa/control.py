from aisteer360.algorithms.state_control.common.algorithms import CAA as CAACore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import WrappedSteerControl

from .args import CAAArgs


class CAA(WrappedSteerControl):
    Args = CAAArgs
    _steer_cls = CAACore
    _steer_kwarg_names = ()
