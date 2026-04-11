"""SphereSteer: hypersphere-geometry activation steering."""
from .args import SphereSteerArgs
from .control import SphereSteer
from .utils.data import SphereSteerCache

STEERING_METHOD = {
    "category": "state_control",
    "name": "sphere_steer",
    "control": SphereSteer,
    "args": SphereSteerArgs,
}

__all__ = [
    "SphereSteer",
    "SphereSteerArgs",
    "SphereSteerCache",
    "STEERING_METHOD",
]
