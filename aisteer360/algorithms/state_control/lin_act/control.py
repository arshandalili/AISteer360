from aisteer360.algorithms.state_control.common.algorithms import LinAcT as LinAcTCore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import WrappedSteerControl

from .args import LinAcTArgs


class LinAcT(WrappedSteerControl):
    Args = LinAcTArgs
    _steer_cls = LinAcTCore
    _steer_kwarg_names = ()
