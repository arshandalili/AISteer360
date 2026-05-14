from .args import SphereSteerArgs
from .control import SphereSteer

STEERING_METHOD = {
    "category": "state_control",
    "name": "sphere_steer",
    "control": SphereSteer,
    "args": SphereSteerArgs,
}
