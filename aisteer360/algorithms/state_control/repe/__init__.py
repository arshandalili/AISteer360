from .args import RepEArgs
from .control import RepE

STEERING_METHOD = {
    "category": "state_control",
    "name": "repe",
    "control": RepE,
    "args": RepEArgs,
}
