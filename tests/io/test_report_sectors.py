"""Tests for the report section on the sector hierarchy."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from feynkit import FeynmanIntegral
from feynkit.io import render_latex, render_text
from feynkit.io.report import DEFAULT_SECTIONS, SECTION_NAMES, AnalysisReport
from feynkit.landau import _singular_binary
from tests.test_sectors import _on_shell

BOX = FeynmanIntegral.from_cnickel("13e|2e|3e|e|:zzzz", kinematics="massless_on_shell")
requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")


def test_it_follows_the_hasse_diagram_and_is_not_built_by_default() -> None:
    assert SECTION_NAMES[SECTION_NAMES.index("hasse") + 1] == "sectors"
    assert "sectors" not in DEFAULT_SECTIONS
    assert AnalysisReport.from_integral(BOX).sectors is None


def test_the_section_holds_the_hierarchy_and_the_resonance_of_each_face() -> None:
    report = AnalysisReport.from_integral(BOX, ["sectors"])
    section = report.sectors
    assert section is not None and section.hierarchy is not None
    h = section.hierarchy
    assert h.counts_source == "generic" and h.symmetric
    assert h.totals().non_zero == 7 and h.totals().unique == 5
    assert section.resonance is not None
    assert set(section.resonance) == {s.id for s in h.sectors if s.point_indices}
    assert set(section.resonance.values()) <= {"all", "progression", "point", "never"}


def test_a_large_hierarchy_is_left_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("feynkit.io.sections.sectors.MAX_PROPAGATORS", 3)
    section = AnalysisReport.from_integral(BOX, ["sectors"]).sectors
    assert section is not None and section.hierarchy is None
    assert section.propagators == 4 and section.limit == 3
    text = render_text(AnalysisReport.from_integral(BOX, ["sectors"]))
    assert "4 propagators" in text and "left out" in text


def test_the_documents_carry_the_section() -> None:
    report = AnalysisReport.from_integral(BOX, ["sectors"])
    text = render_text(report)
    assert "Sector hierarchy" in text
    assert "N_id" in text
    # The face and its resonance sit side by side, and no claim is made about them.
    assert "Resonance" in text and "agree" not in text
    latex = render_latex(report)
    assert "Sector hierarchy" in latex and "\\begin{longtable}" in latex
    assert "agree" not in latex
    for key in ("lee2013", "duhr2026"):
        assert key in latex


def test_the_summary_has_the_numbers_of_sectors() -> None:
    report = AnalysisReport.from_integral(BOX, ["sectors"])
    rows = dict(report.summary())
    assert rows["Nonzero sectors"] == "7"
    assert rows["Scaleless sectors"] == "8"
    assert rows["Unique sectors"] == "5"


@requires_singular
def test_counts_at_a_point_add_the_columns_and_the_flags() -> None:
    fi = _on_shell("13e|2e|3e|e|:aaaa", ["p1^2", "p2^2", "p3^2", "p4^2"])
    report = AnalysisReport.from_integral(fi, ["sectors"], sectors_counts="critical")
    section = report.sectors
    assert section is not None and section.hierarchy is not None
    assert section.hierarchy.counts_source == "critical"
    text = render_text(report)
    assert "m(T)" in text and "N_T" in text
    top = section.hierarchy.sector(15)
    assert dict(report.summary())["Masters of the top sector"] == str(top.count)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_the_document_compiles_without_overfull_lines(tmp_path: Path) -> None:
    report = AnalysisReport.from_integral(BOX, ["sectors"])
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
