from aisteer360.algorithms.state_control.common.algorithms import MiMiC as MiMiCCore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import WrappedSteerControl

from .args import MiMiCArgs


class MiMiC(WrappedSteerControl):
    Args = MiMiCArgs
    _steer_cls = MiMiCCore
    _steer_kwarg_names = ()
