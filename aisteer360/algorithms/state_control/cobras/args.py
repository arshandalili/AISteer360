from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class COBRASArgs(BaseArgs):
    layer_id: int = field(metadata={"help": "Layer index."})
    k_bw: int = field(default=5, metadata={"help": "KNN bandwidth k."})
    n_sinkhorn: int = field(default=5, metadata={"help": "Sinkhorn iterations."})
    alpha_sigma: float = field(default=1e-3, metadata={"help": "Centroid regularization."})
    epsilon: float = field(default=0.0, metadata={"help": "Langevin noise scale."})
    max_iters: int = field(default=10, metadata={"help": "Geodesic iterations."})
    vmf_kappa: float = field(default=20.0, metadata={"help": "vMF concentration."})
    vmf_beta: float = field(default=0.0, metadata={"help": "vMF threshold."})
    T: float = field(default=0.5, metadata={"help": "Strength multiplier."})

    def __post_init__(self):
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0.")
