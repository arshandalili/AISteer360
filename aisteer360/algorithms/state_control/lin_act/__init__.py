from .args import LinAcTArgs
from .control import LinAcT

STEERING_METHOD = {
    "category": "state_control",
    "name": "lin_act",
    "control": LinAcT,
    "args": LinAcTArgs,
}
