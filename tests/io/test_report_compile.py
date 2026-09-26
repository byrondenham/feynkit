"""Compile tests for the LaTeX analysis report.

The document for the massive bubble, the massless triangle and the massless
box must compile under pdflatex with no errors, no overfull lines and no
undefined references or citations, and so must the bubble's document with the
point-count section, with a candidate and without one. Each document is
written to a temporary directory and compiled twice, so that the citations
resolve on the second pass. The tests are skipped when pdflatex is not
installed. The Landau analysis of the box is slow without Singular, so the
box is also skipped when Singular is not installed.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.io.report import SECTION_NAMES, AnalysisReport
from feynkit.io.report_latex import render_latex
from feynkit.landau import _singular_binary
from feynkit.point_count import count_torus_points

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
    _compile(FeynmanIntegral.from_cnickel(cnickel).to_latex(), tmp_path)


@requires_pdflatex
def test_report_with_point_counts_compiles(tmp_path: Path) -> None:
    _compile(FeynmanIntegral.from_cnickel("11e|e|:nn").to_latex(SECTION_NAMES), tmp_path)


@requires_pdflatex
@pytest.mark.parametrize("kind", ["check", "bound", "degree"])  # type: ignore[misc]
def test_report_without_a_candidate_compiles(kind: str, tmp_path: Path) -> None:
    # The reasons write p = 41, [0, N! Vol] = [0, 2] and q^1, which the document sets as maths.
    bubble = FeynmanIntegral.from_cnickel("11e|e|:nn")
    report = AnalysisReport.from_integral(bubble, SECTION_NAMES)
    assert report.torus is not None
    if kind == "check":
        # m_1^2 = 5, m_2^2 = 8 and s = 2, where lambda = -39; the check fails at p = 41.
        point = {key: value for (key, _), value in zip(report.torus.point, (5, 8, 2), strict=True)}
        count = bubble.torus_count(point=point)
    elif kind == "bound":
        count = count_torus_points(
            bubble.symanzik.g,
            bubble.symanzik.lp_parameters,
            scale=bubble.graph.energy_scale,
            volume_bound=2,
        )
    else:
        u, x = sp.symbols("u x")
        count = count_torus_points(u**2 - x, [u], point={x: 11})
    assert count.candidate_master_count is None
    report = dataclasses.replace(report, torus=count)
    _compile(render_latex(report), tmp_path)


def _compile(latex: str, tmp_path: Path) -> None:
    tex = tmp_path / "report.tex"
    tex.write_text(latex)
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
