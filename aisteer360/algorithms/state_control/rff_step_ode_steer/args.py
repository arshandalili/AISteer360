from dataclasses import dataclass, field
from typing import Union

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class RFFStepODESteerArgs(BaseArgs):
    layer_id: int = field(metadata={"help": "Transformer block index."})
    n_components: int = field(default=8000, metadata={"help": "RFF components."})
    sigma: Union[float, str] = field(default="median", metadata={"help": "RFF bandwidth."})
    lin_clf_type: str = field(default="lr", metadata={"help": "Linear classifier ('lr' or 'svm')."})
    T: float = field(default=1.0, metadata={"help": "Step magnitude."})

    def __post_init__(self):
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")
