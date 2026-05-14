from .args import MiMiCArgs
from .control import MiMiC

STEERING_METHOD = {
    "category": "state_control",
    "name": "mimic",
    "control": MiMiC,
    "args": MiMiCArgs,
}
