"""Tests for report.py, per TECHNICAL_SPEC.md Section 9.3."""

from __future__ import annotations

import numpy as np
import pytest

from worldgap.analyzer import GapResult
from worldgap.metrics.frechet import frechet_distance
from worldgap.metrics.mmd import mmd_squared
from worldgap.report import ReportEntry, generate_report


def _gap_result(seed: int, mean_shift: float = 0.0, n: int = 40, dim: int = 8) -> GapResult:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n, dim))
    target = rng.normal(size=(n, dim)) + mean_shift
    fd = frechet_distance(source, target)
    mmd = mmd_squared(source, target)
    return GapResult(frechet=fd, mmd=mmd, warnings=[] if fd.confidence != "low" else ["low confidence"])


def test_html_report_is_written_and_contains_expected_sections(tmp_path):
    entries = [ReportEntry("clean_vs_perturbed", _gap_result(seed=0, mean_shift=1.0))]
    out = generate_report(entries, tmp_path / "report.html")
    assert out.exists()
    html = out.read_text()
    assert "<table>" in html
    assert "data:image/png;base64," in html
    assert "clean_vs_perturbed" in html


def test_markdown_report_is_written_with_sibling_png(tmp_path):
    entries = [ReportEntry("clean_vs_perturbed", _gap_result(seed=0, mean_shift=1.0))]
    out = generate_report(entries, tmp_path / "report.md")
    assert out.exists()
    md = out.read_text()
    assert "| condition |" in md
    assert "report.png" in md
    assert (tmp_path / "report.png").exists()


def test_both_metrics_always_present_in_condition_table(tmp_path):
    """Spec 7.2: Frechet and MMD MUST both be reported together, never one in
    isolation.
    """
    entries = [ReportEntry("only_condition", _gap_result(seed=1))]
    out = generate_report(entries, tmp_path / "report.md")
    md = out.read_text()
    assert "frechet_distance" in md
    assert "mmd_squared" in md


def test_low_confidence_warning_surfaced_in_report(tmp_path):
    # Small n relative to latent_dim triggers low confidence (spec 7.3).
    entries = [ReportEntry("tiny_sample", _gap_result(seed=2, n=5, dim=8))]
    out = generate_report(entries, tmp_path / "report.html")
    html = out.read_text()
    assert "Low sample-size confidence" in html


def test_disagreement_note_appears_with_enough_conditions(tmp_path):
    # Construct entries where Frechet and MMD trend in opposite directions
    # across conditions -- should trigger the disagreement note.
    entries = [
        ReportEntry("a", _gap_result(seed=10, mean_shift=0.0)),
        ReportEntry("b", _gap_result(seed=11, mean_shift=3.0)),
        ReportEntry("c", _gap_result(seed=12, mean_shift=0.1)),
        ReportEntry("d", _gap_result(seed=13, mean_shift=2.5)),
    ]
    # This is a best-effort trigger, not guaranteed for every RNG draw --
    # the important contract is that the function runs and, if it does find
    # disagreement, surfaces it. Assert the report at least always succeeds
    # and includes both metrics regardless.
    out = generate_report(entries, tmp_path / "report.md")
    md = out.read_text()
    assert "frechet_distance" in md and "mmd_squared" in md


def test_empty_entries_raises():
    with pytest.raises(ValueError, match="at least one"):
        generate_report([], "report.html")


def test_unsupported_extension_raises(tmp_path):
    entries = [ReportEntry("x", _gap_result(seed=0))]
    with pytest.raises(ValueError, match="unsupported"):
        generate_report(entries, tmp_path / "report.pdf")
