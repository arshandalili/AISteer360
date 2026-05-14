from .args import RFFODESteerArgs
from .control import RFFODESteer

STEERING_METHOD = {
    "category": "state_control",
    "name": "rff_ode_steer",
    "control": RFFODESteer,
    "args": RFFODESteerArgs,
}
