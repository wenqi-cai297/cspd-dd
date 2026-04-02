"""CSPD Stage 2 package.

Stage 2 now means generative-backbone adaptation / canonical-semantic-space
familiarization. This package intentionally does not revive the old Stage 2
render semantics.
"""

from cspd_stage2.data import Stage2PairedDataset, Stage2PairRecord
from cspd_stage2.training import Stage2TrainConfig, run_stage2_training

__all__ = [
    "__version__",
    "Stage2PairRecord",
    "Stage2PairedDataset",
    "Stage2TrainConfig",
    "run_stage2_training",
]

__version__ = "0.1.0"
