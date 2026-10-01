"""Tests for the report section on degenerate faces.

Oracles: FeynmanIntegral.face_degeneracy at the point the torus counts draw,
whose own tests check it against faces worked by hand and published examples,
and the torus counts of the same report.
"""

from __future__ import annotations

import dataclasses

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit import degeneracy as degeneracy_module
from feynkit.core.exceptions import ValidationError
from feynkit.io import render_latex, render_text
from feynkit.io.report import DEFAULT_SECTIONS, SECTION_NAMES, AnalysisReport
from feynkit.io.sections import degeneracy as section_module
from feynkit.io.sections.degeneracy import decide, torus_point
from feynkit.landau import _singular_binary

pytestmark = [
    pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed"),
    pytest.mark.usefixtures("sympy_seeded"),
]

HEADING = "Degenerate faces"


def _section(text: str, heading: str = HEADING) -> str:
    body = text.split(f"\n{heading}\n{'-' * len(heading)}\n", 1)[1]
    return " ".join(body.split("\n\n\n")[0].split())


def _triangle_with_massless_leg() -> FeynmanIntegral:
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
    p1 = sp.Symbol("p1^2", real=True)
    products = {k: sp.expand(sp.sympify(v).subs(p1, 0)) for k, v in fi.momentum_products.items()}
    return fi.with_(momentum_products=products)


@pytest.fixture(scope="module")  # type: ignore[misc]
def degenerate_report() -> AnalysisReport:
    return AnalysisReport.from_integral(_triangle_with_massless_leg(), ["torus", "degeneracy"])


@pytest.fixture(scope="module")  # type: ignore[misc]
def smooth_report() -> AnalysisReport:
    return AnalysisReport.from_integral(
        FeynmanIntegral.from_cnickel("12e|2e|e|:zzz"), ["torus", "degeneracy"]
    )


class TestBuild:
    def test_it_follows_the_landau_section_and_is_not_built_by_default(self) -> None:
        assert SECTION_NAMES[SECTION_NAMES.index("landau") + 1] == "degeneracy"
        assert "degeneracy" not in DEFAULT_SECTIONS

    def test_the_point_is_that_of_the_torus_counts(self, degenerate_report: AnalysisReport) -> None:
        section, torus = degenerate_report.degeneracy, degenerate_report.torus
        assert section is not None and torus is not None
        assert section.analysis is not None
        assert section.analysis.mode == "point"
        assert section.analysis.point == torus.point
        assert section.seed == torus.seed == 0
        fi = _triangle_with_massless_leg()
        assert section.analysis == fi.face_degeneracy(torus_point(fi, 0))

    def test_identifications_follow_the_degenerate_faces(
        self, degenerate_report: AnalysisReport
    ) -> None:
        section = degenerate_report.degeneracy
        assert section is not None and section.analysis is not None
        (face,) = section.analysis.degenerate_faces
        (found,) = section.identifications
        assert found is not None and found.verified
        points = section.analysis.points
        assert {points[i] for i in found.point_indices} == {points[i] for i in face.point_indices}

    def test_generic_mode_is_an_option(self) -> None:
        fi = _triangle_with_massless_leg()
        report = AnalysisReport.from_integral(fi, ["degeneracy"], degeneracy_mode="generic")
        section = report.degeneracy
        assert section is not None and section.seed is None
        assert section.analysis == fi.face_degeneracy()

    def test_an_unknown_mode_is_refused(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        with pytest.raises(ValidationError, match="degeneracy_mode"):
            AnalysisReport.from_integral(fi, ["degeneracy"], degeneracy_mode="tropical")

    def test_kinematic_constraints_are_refused_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = sp.Symbol("s", real=True)
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematic_constraints=[s - 1])

        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("no section is built")

        monkeypatch.setattr(AnalysisReport, "__init__", refuse)
        with pytest.raises(ValidationError, match="degeneracy section"):
            AnalysisReport.from_integral(fi, ["degeneracy"])

    def test_without_singular_the_faces_are_not_decided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(degeneracy_module, "_singular_binary", lambda: None)
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        report = AnalysisReport.from_integral(fi, ["degeneracy"])
        assert report.degeneracy is not None and report.degeneracy.analysis is None
        assert dict(report.summary())["Degenerate faces"] == "not decided"
        assert "Singular" in _section(render_text(report))
        assert "Singular" in render_latex(report)

    def test_without_singular_nothing_slow_runs_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*args: object, **kwargs: object) -> None:
            raise AssertionError("the Landau analysis ran without Singular")

        monkeypatch.setattr(degeneracy_module, "_singular_binary", lambda: None)
        monkeypatch.setattr(section_module, "landau_analysis", refuse)
        monkeypatch.setattr(section_module, "torus_point", refuse)
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        assert decide(fi, seed=0).analysis is None


class TestRendering:
    def test_a_degenerate_face_is_listed(self, degenerate_report: AnalysisReport) -> None:
        text = _section(render_text(degenerate_report))
        assert "One face is degenerate." in text
        assert "No face is degenerate" not in text
        assert "drawn with seed 0" in text
        assert "FeynmanIntegral.face_degeneracy" in text
        # The counts at seed 0 give no candidate, so nothing is compared.
        assert degenerate_report.torus is not None
        assert degenerate_report.torus.candidate_euler_characteristic is None
        assert "|chi(X)|" not in text
        assert dict(degenerate_report.summary())["Degenerate faces"] == "1"

    def test_a_candidate_of_the_counts_is_set_beside_the_volume(
        self, degenerate_report: AnalysisReport
    ) -> None:
        torus = degenerate_report.torus
        assert torus is not None
        counted = dataclasses.replace(torus, candidate_euler_characteristic=-5)
        report = dataclasses.replace(degenerate_report, torus=counted)
        text = _section(render_text(report))
        assert "The point counts give the candidate |chi(X)| = 5, and N! Vol(P_z) - " in text
        assert "|chi(X)| = 2." in text
        latex = " ".join(render_latex(report).split())
        assert "$N!\\,\\mathrm{Vol}(P_z) - |\\chi(X)| = 2$" in latex
        moved = dataclasses.replace(counted, point=((counted.point[0][0], 99),))
        assert "point counts give" not in _section(
            render_text(dataclasses.replace(degenerate_report, torus=moved))
        )

    def test_no_degenerate_face_gives_the_volume(self, smooth_report: AnalysisReport) -> None:
        section = smooth_report.degeneracy
        assert section is not None and section.analysis is not None
        text = _section(render_text(smooth_report))
        assert "No face is degenerate" in text
        assert f"|chi(X)| = N! Vol(P_z) = {section.analysis.volume}" in text
        assert dict(smooth_report.summary())["Degenerate faces"] == "0"
        latex = render_latex(smooth_report)
        assert "\\section{Degenerate faces}" in latex
        assert "\\cite{gkz1994}" in latex and "\\cite{fmt2024}" in latex

    def test_the_text_is_plain_ascii(
        self, degenerate_report: AnalysisReport, smooth_report: AnalysisReport
    ) -> None:
        for report in (degenerate_report, smooth_report):
            text = render_text(report)
            assert text.isascii()
            assert "$" not in text and "\\" not in text

    def test_an_undecided_face_is_reported(self, degenerate_report: AnalysisReport) -> None:
        section = degenerate_report.degeneracy
        assert section is not None and section.analysis is not None
        faces = list(section.analysis.faces)
        k = next(i for i, f in enumerate(faces) if f.method == "groebner" and not f.degenerate)
        faces[k] = dataclasses.replace(
            faces[k],
            degenerate=None,
            singular_dimension=None,
            tjurina=None,
            reason="Singular ran past timeout=120 s",
        )
        analysis = dataclasses.replace(section.analysis, faces=tuple(faces))
        report = dataclasses.replace(
            degenerate_report, degeneracy=dataclasses.replace(section, analysis=analysis)
        )
        text = _section(render_text(report))
        assert "One face is undecided" in text
        assert dict(report.summary())["Degenerate faces"] == "at least 1"
