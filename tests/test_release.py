"""The version is stated consistently wherever it is recorded."""

from __future__ import annotations

import re
from pathlib import Path

import feynkit

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == feynkit.__version__
