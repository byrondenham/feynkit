"""Shared fixtures for the fk command-line tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def work_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each CLI test in its own directory, so a default feynkit.db lands there.

    NO_COLOR keeps Python 3.14's coloured argparse help out of the captured text.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
