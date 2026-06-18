"""Tests for config.py's GapConfig.from_yaml, per spec 9.2's
`worldgap train --config configs/v1_default.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worldgap.config import GapConfig

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_from_yaml_loads_real_v1_default_config():
    config = GapConfig.from_yaml(_CONFIGS_DIR / "v1_default.yaml", modality="perception")
    assert config.modality == "perception"
    assert config.state_dim == 258
    assert config.world_model.summary_dim == 64
    assert config.encoder.d_model == 256


def test_from_yaml_loads_real_v2_default_config():
    config = GapConfig.from_yaml(_CONFIGS_DIR / "v2_default.yaml", modality="actuation")
    assert config.modality == "actuation"
    assert config.state_dim == 2
    assert config.world_model.summary_dim == 32
    assert config.encoder.d_model == 64


def test_from_yaml_rejects_modality_in_file(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("modality: perception\nstate_dim: 258\n")
    with pytest.raises(ValueError, match="--modality CLI flag"):
        GapConfig.from_yaml(bad_config, modality="perception")


def test_from_yaml_empty_file_perception_uses_all_defaults(tmp_path):
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("")
    config = GapConfig.from_yaml(empty_config, modality="perception")
    assert config.state_dim == 258
    assert config.world_model.summary_dim == 64


def test_from_yaml_empty_file_actuation_needs_explicit_state_dim(tmp_path):
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("")
    with pytest.raises(ValueError, match="state_dim"):
        GapConfig.from_yaml(empty_config, modality="actuation")
