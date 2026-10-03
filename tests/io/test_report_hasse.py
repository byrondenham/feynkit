"""Tests for the report section on the Hasse diagram of the face lattice."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from feynkit import FeynmanIntegral
from feynkit.io import render_latex, render_text
from feynkit.io.report import DEFAULT_SECTIONS, SECTION_NAMES, AnalysisReport
from feynkit.visualisation.hasse import hasse_tikz


def test_it_follows_the_face_lattice_and_is_not_built_by_default() -> None:
    assert SECTION_NAMES[SECTION_NAMES.index("face_lattice") + 1] == "hasse"
    assert "hasse" not in DEFAULT_SECTIONS
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    assert AnalysisReport.from_integral(fi).hasse is None


def test_the_section_holds_the_diagram_of_the_lattice() -> None:
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    report = AnalysisReport.from_integral(fi, ["hasse"])
    assert report.hasse is not None
    assert report.hasse.codimension == 2
    assert report.hasse.tikz == hasse_tikz(report.hasse.lattice, "codim", 2)
    assert report.hasse.faces == len(re.findall(r"\(f\d+\) at", report.hasse.tikz))


def test_a_large_diagram_is_drawn_to_a_smaller_codimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("feynkit.io.sections.hasse.MAX_FACES", 20)
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    report = AnalysisReport.from_integral(fi, ["hasse"])
    assert report.hasse is not None
    assert report.hasse.codimension == 1
    assert report.hasse.faces == 1 + 6


def test_the_documents_carry_the_section() -> None:
    fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
    report = AnalysisReport.from_integral(fi, ["hasse"])
    text = render_text(report)
    assert "Hasse diagram of the face lattice" in text
    assert "tikzpicture" not in text
    latex = render_latex(report)
    assert "\\label{fig:hasse}" in latex and "\\begin{tikzpicture}" in latex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_the_document_compiles_without_overfull_lines(tmp_path: Path) -> None:
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    report = AnalysisReport.from_integral(fi, ["hasse"])
    source = tmp_path / "report.tex"
    source.write_text(render_latex(report), encoding="utf-8")
    for _ in range(2):
        done = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stdout[-2000:]
    log = (tmp_path / "report.log").read_text(encoding="utf-8", errors="replace")
    assert not re.search(r"Overfull \\[hv]box", log)
