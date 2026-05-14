from .args import RFFStepODESteerArgs
from .control import RFFStepODESteer

STEERING_METHOD = {
    "category": "state_control",
    "name": "rff_step_ode_steer",
    "control": RFFStepODESteer,
    "args": RFFStepODESteerArgs,
}
