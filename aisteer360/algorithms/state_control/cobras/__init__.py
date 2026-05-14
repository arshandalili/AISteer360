from .args import COBRASArgs
from .control import COBRAS

STEERING_METHOD = {
    "category": "state_control",
    "name": "cobras",
    "control": COBRAS,
    "args": COBRASArgs,
}
