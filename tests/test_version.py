"""Guard: the version string lives in two places and they must agree.

`pyproject.toml` sets the wheel/PyPI version; `worldgap.__version__` is what
users see at runtime. Read with a regex rather than `tomllib`, which is
Python 3.11+ while this project supports 3.10.
"""

from __future__ import annotations

import re
from pathlib import Path

import worldgap

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_package_version_matches_pyproject() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "no top-level version = \"...\" line in pyproject.toml"
    assert worldgap.__version__ == match.group(1)
