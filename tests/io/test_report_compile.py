"""Compile tests for the LaTeX analysis report.

The document for the massive bubble, the massless triangle and the massless
box must compile under pdflatex with no errors, no overfull lines and no
undefined references or citations. Each document is written to a temporary
directory and compiled twice, so that the citations resolve on the second
pass. The tests are skipped when pdflatex is not installed. The Landau
analysis of the box is slow without Singular, so the box is also skipped
when Singular is not installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from feynkit import FeynmanIntegral
from feynkit.landau import _singular_binary

requires_pdflatex = pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="pdflatex not installed"
)
requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

# Only these warnings mean a reference is missing; other log lines can
# contain the word "undefined" harmlessly, e.g. in font information.
_UNDEFINED = re.compile(
    r"(Reference|Citation) `[^']*' on page \d+ undefined|There were undefined references"
)


@requires_pdflatex
@pytest.mark.parametrize(  # type: ignore[misc]
    "cnickel",
    [
        "11e|e|:nn",
        "12e|2e|e|:zzz",
        pytest.param("12e|3e|3e|e|:zzzz", marks=requires_singular),
    ],
)
def test_report_compiles_without_errors_or_overfull_boxes(cnickel: str, tmp_path: Path) -> None:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    tex = tmp_path / "report.tex"
    tex.write_text(fi.to_latex())
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    log = (tmp_path / "report.log").read_text(errors="replace")
    lines = log.splitlines()
    assert proc.returncode == 0, log[-3000:]
    assert (tmp_path / "report.pdf").exists()
    assert not [line for line in lines if "Overfull \\hbox" in line]
    assert not [line for line in lines if _UNDEFINED.search(line)]
