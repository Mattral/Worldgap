"""Report generation, per TECHNICAL_SPEC.md Section 9.3.

'A generated HTML or Markdown report per `analyze` run, bundling the
GapResult, the condition table, and Frechet/MMD trend plots (matplotlib or
plotly). This is a generated artifact, not a hosted service -- no server
component.'

Also enforces spec 7.2's requirement that Frechet and MMD MUST always be
reported together, never one in isolation, and section 210's requirement that
disagreement between the two metrics MUST be surfaced, not hidden -- Frechet
distance assumes approximately-Gaussian latents, so if the two metrics rank
conditions differently, that disagreement is itself diagnostic of non-Gaussian
latent structure, not noise to average away.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this module never opens a display window

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from .analyzer import GapResult  # noqa: E402

_MIN_CONDITIONS_FOR_DISAGREEMENT_CHECK = 3


@dataclass
class ReportEntry:
    """One (condition, GapResult) pair going into a report. For a single
    `worldgap analyze --source ... --target ...` invocation there's exactly
    one entry; multiple entries (e.g. one per pre-registered validation
    condition) produce a genuine trend plot instead of a single bar.
    """

    condition_label: str
    result: GapResult


def _condition_table_rows(entries: list[ReportEntry]) -> list[dict]:
    rows = []
    for e in entries:
        r = e.result
        rows.append(
            {
                "condition": e.condition_label,
                "n_source": r.n_source,
                "n_target": r.n_target,
                "frechet_distance": r.frechet.distance,
                "mmd_squared": r.mmd.mmd_squared,
                "confidence": r.confidence,
                "warnings": "; ".join(r.warnings) if r.warnings else "",
            }
        )
    return rows


def _disagreement_note(entries: list[ReportEntry]) -> str | None:
    """Per spec 210: flags Frechet/MMD rank disagreement across conditions as
    a diagnostic signal, not something to silently average past. Only
    meaningful with enough points to rank; returns None below that threshold
    rather than reporting a spurious correlation from 1-2 points.
    """
    if len(entries) < _MIN_CONDITIONS_FOR_DISAGREEMENT_CHECK:
        return None
    frechet_vals = [e.result.frechet.distance for e in entries]
    mmd_vals = [e.result.mmd.mmd_squared for e in entries]
    rho, _ = spearmanr(frechet_vals, mmd_vals)
    if rho != rho:  # NaN check without importing numpy just for this
        return "Frechet/MMD rank correlation is undefined (one metric is constant across conditions)."
    if rho < 0.5:
        return (
            f"Frechet distance and MMD rank conditions differently across this run "
            f"(Spearman rho={rho:.2f}) -- per spec Section 7.2, this is a diagnostic "
            "signal for non-Gaussian latent structure, not noise to average past. "
            "Treat single-metric conclusions from this run with caution."
        )
    return None


def _render_trend_plot_png_base64(entries: list[ReportEntry]) -> str:
    labels = [e.condition_label for e in entries]
    frechet_vals = [e.result.frechet.distance for e in entries]
    mmd_vals = [e.result.mmd.mmd_squared for e in entries]

    fig, ax1 = plt.subplots(figsize=(max(5, len(entries) * 1.2), 4))
    x = range(len(entries))
    ax1.plot(x, frechet_vals, marker="o", color="tab:blue", label="Frechet distance")
    ax1.set_ylabel("Frechet distance", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, mmd_vals, marker="s", color="tab:red", label="MMD^2")
    ax2.set_ylabel("MMD^2", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Frechet distance vs. MMD^2 by condition")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_trend_plot_png_bytes(entries: list[ReportEntry]) -> bytes:
    return base64.b64decode(_render_trend_plot_png_base64(entries))


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def _html_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No results.</p>"
    headers = list(rows[0].keys())
    thead = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for row in rows:
        tds = "".join(f"<td>{row[h]}</td>" for h in headers)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def generate_report(
    entries: list[ReportEntry],
    output_path: str | Path,
    title: str = "WorldGap Gap Analysis Report",
) -> Path:
    """Writes an HTML or Markdown report, chosen by `output_path`'s extension.
    `entries` MUST be non-empty. Both Frechet and MMD are always shown
    together per spec 7.2; a rank-disagreement note is added automatically
    when there are enough conditions to check it (see `_disagreement_note`).
    """
    if not entries:
        raise ValueError("generate_report requires at least one ReportEntry")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    rows = _condition_table_rows(entries)
    disagreement = _disagreement_note(entries)
    any_low_confidence = any(e.result.confidence == "low" for e in entries)

    if suffix in (".html", ".htm"):
        img_b64 = _render_trend_plot_png_base64(entries)
        warning_html = ""
        if any_low_confidence:
            warning_html += (
                '<p class="warning"><strong>Low sample-size confidence</strong> flagged on '
                "at least one condition -- see spec Section 7.3 (n &ge; 5 &times; latent_dim "
                "per domain is the rule of thumb).</p>"
            )
        if disagreement:
            warning_html += f'<p class="warning"><strong>Metric disagreement:</strong> {disagreement}</p>'
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #1a1a1a; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: right; }}
th {{ background: #f0f0f0; }}
.warning {{ background: #fff3cd; border: 1px solid #ffe69c; padding: 0.6rem 1rem; border-radius: 4px; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>{title}</h1>
{warning_html}
<h2>Condition table</h2>
{_html_table(rows)}
<h2>Frechet distance vs. MMD by condition</h2>
<img src="data:image/png;base64,{img_b64}" alt="Frechet vs MMD trend plot">
</body>
</html>
"""
        output_path.write_text(html, encoding="utf-8")
        return output_path

    if suffix in (".md", ".markdown"):
        png_path = output_path.with_suffix(".png")
        png_path.write_bytes(_render_trend_plot_png_bytes(entries))
        parts = [f"# {title}", ""]
        if any_low_confidence:
            parts += [
                "> **Low sample-size confidence** flagged on at least one condition -- "
                "see spec Section 7.3 (n >= 5 x latent_dim per domain is the rule of thumb).",
                "",
            ]
        if disagreement:
            parts += [f"> **Metric disagreement:** {disagreement}", ""]
        parts += [
            "## Condition table",
            "",
            _markdown_table(rows),
            "",
            "## Frechet distance vs. MMD by condition",
            "",
            f"![Frechet vs MMD trend plot]({png_path.name})",
            "",
        ]
        output_path.write_text("\n".join(parts), encoding="utf-8")
        return output_path

    raise ValueError(
        f"unsupported report extension {suffix!r} for {output_path} -- use .html/.htm or .md/.markdown"
    )
