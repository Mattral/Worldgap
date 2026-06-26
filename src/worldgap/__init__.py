"""worldgap — reusable world-model-based domain-gap quantification.

Ground truth for this package's design: docs/TECHNICAL_SPEC.md.
"""

from .analyzer import GapAnalyzer, GapResult
from .config import EncoderConfig, GapConfig, TrainingConfig, WorldModelConfig
from .data.index import RolloutIndex
from .data.rollout import Rollout
from .report import ReportEntry, generate_report

__all__ = [
    "GapAnalyzer",
    "GapResult",
    "GapConfig",
    "EncoderConfig",
    "WorldModelConfig",
    "TrainingConfig",
    "Rollout",
    "RolloutIndex",
    "ReportEntry",
    "generate_report",
]

__version__ = "0.1.0"
