from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class StepODESteerArgs(BaseArgs):
    layer_id: int = field(metadata={"help": "Transformer block index."})
    n_components: int = field(default=8000, metadata={"help": "Sketch components."})
    degree: int = field(default=2, metadata={"help": "Polynomial degree."})
    gamma: float = field(default=0.1, metadata={"help": "Polynomial scale."})
    coef0: float = field(default=1.0, metadata={"help": "Polynomial bias."})
    lin_clf_type: str = field(default="lr", metadata={"help": "Linear classifier ('lr' or 'svm')."})
    T: float = field(default=1.0, metadata={"help": "Step magnitude."})

    def __post_init__(self):
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")
