"""worldgap — reusable world-model-based domain-gap quantification.

Ground truth for this package's design: docs/TECHNICAL_SPEC.md.
"""

from .analyzer import GapAnalyzer, GapResult
from .config import EncoderConfig, GapConfig, TrainingConfig, WorldModelConfig
from .data.rollout import Rollout

__all__ = [
    "GapAnalyzer",
    "GapResult",
    "GapConfig",
    "EncoderConfig",
    "WorldModelConfig",
    "TrainingConfig",
    "Rollout",
]

__version__ = "0.1.0"
