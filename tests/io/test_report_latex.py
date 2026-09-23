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

from feynkit import Edge, FeynmanIntegral, Graph
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


@pytest.fixture(scope="module")  # type: ignore[misc]
def sunrise_latex(sunrise: FeynmanIntegral) -> str:
    return render_latex(AnalysisReport.from_integral(sunrise))


def _widest_matrix(latex: str) -> int:
    """The largest number of columns of any bmatrix in the document."""
    widest = 0
    for body in re.findall(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", latex, re.S):
        for row in body.split("\\\\"):
            widest = max(widest, row.count("&") + 1)
    return widest


def test_document_skeleton(latex: str) -> None:
    assert latex.startswith("\\documentclass[11pt,a4paper]{article}")
    assert "\\clearpage" not in latex and "breqn" not in latex and "cleveref" not in latex
    assert "microtype" not in latex
    assert latex.count("\\begin{document}") == 1 and latex.rstrip().endswith("\\end{document}")


def test_matrix_column_limit_fits_the_widest_matrix(latex: str) -> None:
    # amsmath stops at ten columns unless told otherwise; the triangle's
    # widest matrix has six, so the default is kept.
    preamble = latex.split("\\begin{document}")[0]
    assert _widest_matrix(latex) == 6
    assert preamble.index("\\setcounter{MaxMatrixCols}{10}") > preamble.index("amsmath")


def test_matrix_column_limit_grows_with_a_wide_matrix() -> None:
    massive_box = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    latex = render_latex(AnalysisReport.from_integral(massive_box, ["gkz"]))
    assert _widest_matrix(latex) == 14
    assert "\\setcounter{MaxMatrixCols}{14}" in latex


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


@pytest.mark.parametrize("cnickel", ["12e|2e|e|:zzz", "111e|e|:nnn"])  # type: ignore[misc]
def test_integrand_templates_match_the_computed_integrands(cnickel: str) -> None:
    # The sunrise has L = 2, which the triangle's L = 1 would not catch in
    # the Feynman exponents.
    integral = FeynmanIntegral.from_cnickel(cnickel)
    u, f, g, nu = sp.symbols("U F G nu")
    templates = _integrand_templates(integral.loop_count, integral.dimension)
    symanzik = integral.symanzik
    edges = integral.graph.get_internal_edges()
    nu_total = sum(integral.propagator_exponents[e.idx] for e in edges)

    # The Schwinger result names its parameters alpha_e; U and F use a_e.
    schwinger = integral.schwinger
    rename = dict(zip(integral.feynman.parameters, schwinger.parameters, strict=True))
    substituted = templates.schwinger.subs(
        {u: symanzik.u.subs(rename), f: symanzik.f.subs(rename)}, simultaneous=True
    )
    assert sp.simplify(substituted - schwinger.integrand) == 0

    # The delta function of the Feynman representation lives in the measure,
    # not in the integrand, so the template carries U and F only.
    feynman = integral.feynman
    substituted = templates.feynman.subs(
        {u: symanzik.u, f: symanzik.f, nu: nu_total}, simultaneous=True
    )
    assert sp.simplify(substituted / feynman.integrand) == 1

    lee_pomeransky = integral.lee_pomeransky
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


def test_conventions_define_the_integral_and_nu(latex: str) -> None:
    conventions = latex.split("\\section{Conventions}")[1].split("\\section")[0]
    assert (
        "I = e^{L \\epsilon \\gamma_E} \\left(\\mu^2\\right)^{\\nu - L D/2} \\int "
        "\\prod_{r=1}^{L} \\frac{\\mathrm{d}^D k_r}{i \\pi^{D/2}} \\prod_{e} "
        "\\frac{1}{\\left(-q_e^2 + m_e^2\\right)^{\\nu_e}}"
    ) in conventions
    assert "$\\nu = \\sum_e \\nu_e$, here $\\nu = \\nu_{1} + \\nu_{2} + \\nu_{3}$" in conventions
    # nu is defined once, in the conventions
    assert latex.count("\\nu = \\sum_e \\nu_e") == 1


def test_gkz_section_defines_the_euler_mellin_integral(latex: str) -> None:
    assert (
        "I_A(\\beta, z) = \\int_{\\mathbb{R}_{+}^{N}} \\prod_{e} \\mathrm{d}u_e\\, "
        "u_e^{\\nu_e - 1}\\, G(z, u)^{-D/2}"
    ) in latex


def test_symmetry_identity_permutes_the_coefficients_on_the_left(latex: str) -> None:
    # T a_j = a_sigma(j) gives I_A(beta, z_sigma) = I_A(T beta, z), for the
    # integral without its beta-dependent prefactor (de la Cruz 2024).
    assert "$I_A(\\beta, z_\\sigma) = I_A(T\\beta, z)$" in latex
    assert "z_\\sigma = (z_{\\sigma(1)}, \\ldots, z_{\\sigma(6)})" in latex
    assert "I(\\beta, z) = I(T\\beta" not in latex


def test_symmetry_identity_is_self_contained_without_the_gkz_section(
    triangle: FeynmanIntegral,
) -> None:
    latex = render_latex(AnalysisReport.from_integral(triangle, ["symmetries"]))
    assert "\\section{GKZ system}" not in latex
    assert "I_A(\\beta, z) = \\int" in latex
    assert "\\ref{sec:gkz-system}" not in latex


def test_convergence_states_the_euclidean_hypothesis(latex: str) -> None:
    assert (
        "For Euclidean kinematics, where every coefficient of $G$ has positive real part"
    ) in latex


def test_codimension_uses_the_rank_of_a(latex: str) -> None:
    assert "codimension $6 - \\operatorname{rank} A = 6 - 4 = 2$" in latex


def test_every_line_balances_left_and_right(latex: str, sunrise_latex: str) -> None:
    for document in (latex, sunrise_latex):
        for line in document.splitlines():
            assert line.count("\\left") == line.count("\\right"), line


def test_landau_factors_sit_on_their_own_display_lines(sunrise_latex: str) -> None:
    # The sunrise's two-dimensional face discriminant is s times four
    # threshold factors; each gets its own line.
    landau = sunrise_latex.split("\\section{Landau surfaces}")[1].split("\\section")[0]
    two_faces = landau.split("\\item Dimension 2:")[1].split("\\item")[0]
    rows = [
        line.removesuffix(" \\\\")
        for line in two_faces.splitlines()
        if line.startswith("&") and not line.startswith("&\\quad")
    ]
    assert len(rows) == 5
    assert sum(row.startswith("&- m_{1}^{2}") for row in rows) == 4
    assert "&s" in rows


def test_one_loop_factor_lists_are_sorted_and_explained(latex: str) -> None:
    landau = latex.split("\\section{Landau surfaces}")[1].split("\\section")[0]
    first = landau.split("first-type (Cayley) factors are")[1].split("\\end{align*}")[0]
    rows = [line for line in first.splitlines() if line.startswith("&")]
    assert rows == ["&p^{2}_{1} \\\\", "&p^{2}_{2} \\\\", "&p^{2}_{3}"]
    assert "A factor can arise from both a Cayley minor and a Gram minor" in landau


def test_klausen_thesis_names_its_university() -> None:
    assert "PhD thesis, Johannes Gutenberg University Mainz, 2022" in CITATIONS["klausen2023"]


def test_edge_table_uses_indices_and_endpoints(latex: str) -> None:
    assert "$e$ & Vertices & $\\nu_e$ & $m_e$ \\\\" in latex
    assert "1 & 1--2 & $\\nu_{1}$ & $0$ \\\\" in latex


def test_edge_table_of_a_graph_whose_indices_start_at_two() -> None:
    graph = Graph(
        internal_vertices=2,
        external_legs=2,
        edges=[
            Edge(idx=2, v1=1, v2=2, is_internal=True),
            Edge(idx=3, v1=2, v2=1, is_internal=True),
            Edge(idx=4, v1=1, v2=3, is_internal=False),
            Edge(idx=5, v1=2, v2=4, is_internal=False),
        ],
    )
    latex = render_latex(AnalysisReport.from_integral(FeynmanIntegral(graph), []))
    table = latex.split("\\section{Graph}")[1].split("\\section")[0]
    assert "\n2 & 1--2 & $\\nu_{2}$ & $m_{2}$ \\\\" in table
    assert "\n3 & 2--1 & $\\nu_{3}$ & $m_{3}$ \\\\" in table
    assert "{$2$}" in table and "{$3$}" in table  # the figure labels the same indices
    assert "U = a_{2} + a_{3}" in latex


def _landau_rows(latex: str) -> list[str]:
    """The display rows of the Landau section, one factor or continuation each."""
    landau = latex.split("\\section{Landau surfaces}")[1].split("\\section")[0]
    return [line.removesuffix(" \\\\") for line in landau.splitlines() if line.startswith("&")]


@pytest.mark.parametrize(  # type: ignore[misc]
    "cnickel", ["11e|e|:nn", "12e|2e|e|:nzz", "111e|e|:nnn"]
)
def test_landau_factor_lines_carry_no_energy_scale(cnickel: str) -> None:
    # The face discriminants carry powers of mu and numerical factors from
    # the coefficients z_j = c / mu^k; neither is a Landau factor, so the
    # display drops them, and mu never appears on a factor line.
    integral = FeynmanIntegral.from_cnickel(cnickel)
    rows = _landau_rows(render_latex(AnalysisReport.from_integral(integral, ["landau"])))
    assert rows
    for row in rows:
        assert "\\mu" not in row, row
        symbols = re.sub(r"\\[a-zA-Z]+", "", row)
        assert re.search(r"[a-z]", symbols), row


def test_vertex_coefficients_list_their_kinematic_symbol(latex: str) -> None:
    # The triangle's vertex coefficients are -p_i^2 / mu^2; the factor shown
    # is p_i^2.
    two = latex.split("\\item Dimension 0:")[1].split("\\item")[0]
    rows = [line.removesuffix(" \\\\") for line in two.splitlines() if line.startswith("&")]
    assert rows == ["&p^{2}_{1}", "&p^{2}_{2}", "&p^{2}_{3}"]


def test_landau_introduction_reads_plainly(latex: str) -> None:
    assert "Each discriminant that is not identically one factorises as follows" in latex
    assert "other than one factor as follows" not in latex


def test_convergence_rows_take_the_real_part(latex: str) -> None:
    rows = [line for line in latex.splitlines() if "&> 0" in line]
    assert rows
    for row in rows:
        assert row.startswith("\\mathrm{Re}\\left(") and row.rstrip(" \\").endswith("\\right) &> 0")


def test_self_contained_symmetry_identity_uses_one_with_clause(
    triangle: FeynmanIntegral,
) -> None:
    latex = render_latex(AnalysisReport.from_integral(triangle, ["symmetries"]))
    symmetries = latex.split("\\section{Symmetries}")[1].split("\\begin{equation*}")[0]
    identity = symmetries.split("Each gives the identity")[1].split(".")[0]
    assert identity.count(" with ") == 1
    assert "be the Lee-Pomeransky integral without its prefactor" in symmetries
