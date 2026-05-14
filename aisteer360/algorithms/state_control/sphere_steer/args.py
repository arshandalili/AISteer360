from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class SphereSteerArgs(BaseArgs):
    layer_id: int = field(metadata={"help": "Layer index to steer."})
    kappa: float = field(default=20.0, metadata={"help": "vMF concentration."})
    alpha: float = field(default=0.7, metadata={"help": "Max steering strength (0-1)."})
    beta: float = field(default=-0.15, metadata={"help": "Trigger threshold."})
    T: float = field(default=1.0, metadata={"help": "Strength multiplier (unused in SphereSteer's steer)."})

    def __post_init__(self):
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")
