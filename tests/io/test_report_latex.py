"""Tests for the LaTeX renderer of the analysis report.

Golden fragments pin the wording of the sentences a reader relies on (the
beta paragraph, the Euler operators, the toric wording) so that changes to
them are deliberate. The integrand templates are checked against the
integrands feynkit computes, which are the reference: the document writes
each representation in terms of U, F and G, and substituting the actual
polynomials must give back the computed integrand.
"""

from __future__ import annotations

import re

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.io.report import AnalysisReport
from feynkit.io.report_latex import CITATIONS, _integrand_templates, render_latex
from feynkit.normal_forms._invariants import hull_vertex_indices, to_integer_points
from feynkit.polytope import polytope_data


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle_report(triangle: FeynmanIntegral) -> AnalysisReport:
    return AnalysisReport.from_integral(triangle)


@pytest.fixture(scope="module")  # type: ignore[misc]
def latex(triangle_report: AnalysisReport) -> str:
    return render_latex(triangle_report)


@pytest.fixture(scope="module")  # type: ignore[misc]
def sunrise() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("111e|e|:nnn")


def test_document_skeleton(latex: str) -> None:
    assert latex.startswith("\\documentclass[11pt,a4paper]{article}")
    assert "\\clearpage" not in latex and "breqn" not in latex and "cleveref" not in latex
    assert "microtype" not in latex
    assert latex.count("\\begin{document}") == 1 and latex.rstrip().endswith("\\end{document}")


def test_preamble_allows_wide_matrices(latex: str) -> None:
    preamble = latex.split("\\begin{document}")[0]
    assert preamble.index("\\setcounter{MaxMatrixCols}{64}") > preamble.index("amsmath")


def test_source_is_ascii(latex: str) -> None:
    assert latex.isascii()


def test_title_is_the_cnickel(latex: str) -> None:
    assert "\\title{Feynman integral \\texttt{12e|2e|e|:zzz}}" in latex
    assert "\\maketitle" in latex and "\\author" not in latex and "\\date{}" in latex


def test_beta_paragraph(latex: str) -> None:
    assert (
        "The parameter vector is $\\beta = (-D/2, -\\nu_1, \\ldots, -\\nu_N)$, the value the "
        "Euler equations require for the Lee-Pomeransky integral~\\cite{delacruz2019,klausen2020}."
    ) in latex


def test_euler_operator_line(latex: str, triangle: FeynmanIntegral) -> None:
    # first row of A is all ones, so the first operator is theta_1 + ... + theta_n - beta_0
    n = len(triangle.gkz.z_variables)
    expected = " + ".join(f"\\theta_{{{j}}}" for j in range(1, n + 1)) + " + \\frac{D}{2}"
    assert expected in latex


def test_euler_operator_line_with_multiplicities(sunrise: FeynmanIntegral) -> None:
    # the sunrise's second row of A is (2, 2, 1, 1, 1, 0, 0, 1, 1, 0) with beta_1 = -nu_1
    latex = render_latex(AnalysisReport.from_integral(sunrise, ["gkz"]))
    expected = (
        "2 \\theta_{1} + 2 \\theta_{2} + \\theta_{3} + \\theta_{4} + \\theta_{5} "
        "+ \\theta_{8} + \\theta_{9} + \\nu_{1}"
    )
    assert expected in latex


def test_toric_wording(latex: str) -> None:
    assert "analogue of integration-by-parts relations" in latex
    assert "Horn" not in latex and "IBP" not in latex
    assert "master integral" not in latex.split("\\section{Newton polytope}")[0]


def test_rank_paragraph_states_the_volume(latex: str, triangle_report: AnalysisReport) -> None:
    assert triangle_report.polytope is not None
    volume = triangle_report.polytope.data.normalized_volume
    assert f"here {volume}~\\cite{{adolphson1994,chestnov2022}}" in latex
    assert "Cohen-Macaulay" in latex and "\\cite{mmw2005}" in latex


def test_one_convergence_inequality_per_facet(latex: str, triangle_report: AnalysisReport) -> None:
    assert triangle_report.polytope is not None
    assert latex.count("&> 0") == len(triangle_report.polytope.data.facets)


def test_polytope_figure_is_referenced(latex: str) -> None:
    assert "\\label{fig:newton-polytope}" in latex
    assert "\\ref{fig:newton-polytope}" in latex


def test_only_cited_references_appear(latex: str) -> None:
    body, bib = latex.split("\\begin{thebibliography}")
    cited = set(re.findall(r"\\cite\{([^}]*)\}", body))
    keys = {k for group in cited for k in group.split(",")}
    listed = set(re.findall(r"\\bibitem\{([^}]*)\}", bib))
    assert keys == listed and keys


def test_citations_are_complete_and_ascii() -> None:
    for key, text in CITATIONS.items():
        assert text.isascii(), key
        assert "xxxxx" not in text, key
    assert "2404.03564" in CITATIONS["delacruz2024"]


def test_every_section_present(latex: str) -> None:
    for heading in (
        "Summary",
        "Graph",
        "Conventions",
        "Symanzik polynomials",
        "Parametric representations",
        "Newton polytope",
        "GKZ system",
        "Symmetries",
        "Landau surfaces",
        "Schwinger-representation system",
    ):
        assert f"\\section{{{heading}}}" in latex
    # article's thebibliography prints its own References heading
    assert "\\begin{thebibliography}" in latex
    assert "\\section*{References}" not in latex


def test_partial_report_omits_sections(triangle: FeynmanIntegral) -> None:
    latex = render_latex(AnalysisReport.from_integral(triangle, ["gkz"]))
    assert "\\section{GKZ system}" in latex and "\\section{Landau surfaces}" not in latex


def test_special_characters_in_title_are_escaped(triangle_report: AnalysisReport) -> None:
    latex = render_latex(triangle_report, title="one_mass & two")
    assert "\\title{one\\_mass \\& two}" in latex


def test_integrand_templates_match_the_computed_integrands(triangle: FeynmanIntegral) -> None:
    u, f, g, nu = sp.symbols("U F G nu")
    templates = _integrand_templates(triangle.loop_count, triangle.dimension)
    symanzik = triangle.symanzik
    edges = triangle.graph.get_internal_edges()
    nu_total = sum(triangle.propagator_exponents[e.idx] for e in edges)

    # The Schwinger result names its parameters alpha_e; U and F use a_e.
    schwinger = triangle.schwinger
    rename = dict(zip(triangle.feynman.parameters, schwinger.parameters, strict=True))
    substituted = templates.schwinger.subs(
        {u: symanzik.u.subs(rename), f: symanzik.f.subs(rename)}, simultaneous=True
    )
    assert sp.simplify(substituted - schwinger.integrand) == 0

    # The delta function of the Feynman representation lives in the measure,
    # not in the integrand, so the template carries U and F only.
    feynman = triangle.feynman
    substituted = templates.feynman.subs(
        {u: symanzik.u, f: symanzik.f, nu: nu_total}, simultaneous=True
    )
    assert sp.simplify(substituted / feynman.integrand) == 1

    lee_pomeransky = triangle.lee_pomeransky
    substituted = templates.lee_pomeransky.subs({g: symanzik.g})
    assert sp.simplify(substituted / lee_pomeransky.integrand) == 1


@pytest.mark.parametrize("cnickel", ["12e|2e|e|:zzz", "111e|e|:nnn"])  # type: ignore[misc]
def test_orbit_numbering_follows_the_vertex_list(cnickel: str) -> None:
    # The symmetry section numbers vertex orbits by the automorphism
    # computation's hull-vertex list and the polytope section lists
    # PolytopeData.vertices; the two must enumerate the same points in the
    # same order for the numbering to agree.
    points = FeynmanIntegral.from_cnickel(cnickel).newton_polytope.points
    hull = tuple(int(i) for i in hull_vertex_indices(to_integer_points(points)))
    assert hull == polytope_data(points).vertex_indices
