from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class CAAArgs(BaseArgs):
    layer_id: int = field(metadata={"help": "Layer index to steer."})
    T: float = field(default=1.0, metadata={"help": "Strength multiplier."})

    def __post_init__(self):
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")
