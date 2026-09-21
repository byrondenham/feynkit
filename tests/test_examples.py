"""
Smoke test for the example scripts.

Every script in ``examples/`` is run in a subprocess from an empty working
directory and must exit 0 without writing anything into that directory.
The scripts are far slower than the rest of the suite, so the whole module
is marked ``examples`` and skipped unless you ask for it with ``-m examples``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))

pytestmark = pytest.mark.examples


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_runs(script: Path, tmp_path: Path) -> None:
    """The script runs to completion and leaves the working directory alone."""
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout=180,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script.name} exited with {result.returncode}\n"
        f"-- stdout (tail) --\n{result.stdout[-2000:]}\n"
        f"-- stderr (tail) --\n{result.stderr[-2000:]}"
    )
    stray = sorted(path.name for path in tmp_path.iterdir())
    assert not stray, f"{script.name} wrote {stray} into the working directory"
