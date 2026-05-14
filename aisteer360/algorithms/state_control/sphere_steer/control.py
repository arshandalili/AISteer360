from aisteer360.algorithms.state_control.common.algorithms import SphericalSteer as SphericalSteerCore
from aisteer360.algorithms.state_control.common.wrapped_steer_control import WrappedSteerControl

from .args import SphereSteerArgs


class SphereSteer(WrappedSteerControl):
    Args = SphereSteerArgs
    _steer_cls = SphericalSteerCore
    _steer_kwarg_names = ("kappa", "alpha", "beta")
