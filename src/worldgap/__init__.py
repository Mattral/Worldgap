"""worldgap — reusable world-model-based domain-gap quantification.

Ground truth for this package's design: docs/TECHNICAL_SPEC.md.
"""

from .analyzer import GapAnalyzer, GapResult
from .config import EncoderConfig, GapConfig, TrainingConfig, WorldModelConfig
from .data.index import RolloutIndex
from .data.rollout import (
    TEMPORAL_PROVENANCE_KEY,
    TEMPORAL_PROVENANCE_MEANING,
    Rollout,
    split_into_windows,
)
from .report import ReportEntry, generate_report

__all__ = [
    "TEMPORAL_PROVENANCE_KEY",
    "TEMPORAL_PROVENANCE_MEANING",
    "EncoderConfig",
    "GapAnalyzer",
    "GapConfig",
    "GapResult",
    "ReportEntry",
    "Rollout",
    "RolloutIndex",
    "TrainingConfig",
    "WorldModelConfig",
    "generate_report",
    "split_into_windows",
]

__version__ = "0.2.0"
