from .args import ODESteerArgs
from .control import ODESteer

STEERING_METHOD = {
    "category": "state_control",
    "name": "ode_steer",
    "control": ODESteer,
    "args": ODESteerArgs,
}
