"""Tests for the plain-text renderer of the analysis report.

The text report states the same facts as the LaTeX one, in the same order and
with the same citations. The parity tests render both from one
AnalysisReport and compare what each shows: the section headings, the summary
rows and the order in which works are first cited. Golden fragments pin the
plain-text wording of the sentences the LaTeX tests pin.
"""

from __future__ import annotations

import re
from itertools import pairwise

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.io.report import AnalysisReport
from feynkit.io.report_latex import CITATIONS, render_latex
from feynkit.io.report_text import _strip_latex, render_text

_KEY = r"[a-z]+\d{4}"


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


@pytest.fixture(scope="module")  # type: ignore[misc]
def triangle_report(triangle: FeynmanIntegral) -> AnalysisReport:
    return AnalysisReport.from_integral(triangle)


@pytest.fixture(scope="module")  # type: ignore[misc]
def text(triangle_report: AnalysisReport) -> str:
    return render_text(triangle_report)


@pytest.fixture(scope="module")  # type: ignore[misc]
def sunrise_text() -> str:
    return render_text(AnalysisReport.from_integral(FeynmanIntegral.from_cnickel("111e|e|:nnn")))


def _flat(text: str) -> str:
    """The text with every run of whitespace, line breaks included, one space."""
    return " ".join(text.split())


def _text_headings(text: str) -> list[str]:
    """Section headings: lines underlined by hyphens to their full length."""
    return [line for line, rule in pairwise(text.splitlines()) if line and rule == "-" * len(line)]


def _latex_headings(latex: str) -> list[str]:
    """Section headings, with the bibliography under the heading article prints for it."""
    headings = re.findall(r"^\\section\{([^}]*)\}$", latex, re.M)
    if "\\begin{thebibliography}" in latex:
        headings.append("References")
    return headings


def _section(text: str, heading: str) -> str:
    """The body of one section of the text report, up to the next heading."""
    body = text.split(f"\n{heading}\n{'-' * len(heading)}\n", 1)[1]
    following = _text_headings(body)
    if following:
        body = body.split(f"\n{following[0]}\n{'-' * len(following[0])}\n", 1)[0]
    return body


def _text_summary(text: str) -> list[tuple[str, str]]:
    rows = _section(text, "Summary").strip().splitlines()
    return [tuple(re.split(r"\s{2,}", row.strip())) for row in rows]  # type: ignore[misc]


def _latex_summary(latex: str) -> list[tuple[str, str]]:
    table = latex.split("\\section{Summary}")[1].split("\\end{tabular}")[0]
    rows = re.findall(r"^(.+) & (.+) \\\\$", table, re.M)
    return [(label.replace("$", ""), value) for label, value in rows]


def _first_citations(text: str) -> list[str]:
    """Cited keys in the order of first citation."""
    keys: list[str] = []
    for group in re.findall(rf"\[({_KEY}(?:, {_KEY})*)\]", _flat(text)):
        for key in group.split(", "):
            if key not in keys:
                keys.append(key)
    return keys


# --- plain ASCII -------------------------------------------------------------


def test_text_is_plain_ascii_without_latex(text: str, sunrise_text: str) -> None:
    for document in (text, sunrise_text):
        assert document.isascii()
        assert "$" not in document
        assert re.search(r"\\[a-zA-Z]", document) is None
        assert "\\" not in document


def test_title_is_the_cnickel_underlined(text: str) -> None:
    lines = text.splitlines()
    assert lines[0] == "Feynman integral 12e|2e|e|:zzz"
    assert lines[1] == "=" * len(lines[0])


def test_custom_title_is_kept_verbatim(triangle_report: AnalysisReport) -> None:
    lines = render_text(triangle_report, title="one_mass & two").splitlines()
    assert lines[:2] == ["one_mass & two", "=============="]


def test_prose_is_wrapped(text: str, sunrise_text: str) -> None:
    # Displays and tables are indented and may run long; prose may not.
    for document in (text, sunrise_text):
        for line in document.splitlines():
            if not line.startswith(" "):
                assert len(line) <= 79, line


def test_wrapping_keeps_inline_maths_on_one_line(text: str) -> None:
    # A line never breaks inside brackets or next to an operator.
    lines = text.splitlines()
    for span in (
        "m_j . x <= b_j",
        "v_3 = (0, 0, 1)",
        "prod_{e not in T_2}",
        "z_sigma = (z_{sigma(1)}, ..., z_{sigma(6)})",
        "beta = (-D/2, -nu_1, ..., -nu_N)",
        "[klausen2023, tellander2023, walther2022]",
        "nu_1 + nu_2 + nu_3 = D",
    ):
        assert any(span in line for line in lines), span


# --- parity with the LaTeX renderer ------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    ("cnickel", "sections"),
    [
        ("12e|2e|e|:zzz", None),
        ("12e|2e|e|:zzz", ["gkz"]),
        ("12e|2e|e|:zzz", ["symmetries"]),
        ("12e|2e|e|:zzz", []),
        ("11e|e|:nn", None),
        ("111e|e|:nnn", None),
    ],
)
def test_text_and_latex_agree(cnickel: str, sections: list[str] | None) -> None:
    report = AnalysisReport.from_integral(FeynmanIntegral.from_cnickel(cnickel), sections)
    text = render_text(report)
    latex = render_latex(report)

    headings = _text_headings(text)
    assert headings == _latex_headings(latex)
    assert headings[0] == "Summary" and headings[-1] == "References"

    assert _text_summary(text) == _latex_summary(latex) == list(report.summary())

    # The same works, first cited in the same order, listed in that order.
    bibliography = re.findall(r"\\bibitem\{([^}]*)\}", latex)
    body, references = text.split("\nReferences\n----------\n")
    assert _first_citations(body) == bibliography
    assert re.findall(rf"^\[({_KEY})\] ", references, re.M) == bibliography


def test_partial_report_omits_sections(triangle: FeynmanIntegral) -> None:
    headings = _text_headings(render_text(AnalysisReport.from_integral(triangle, ["gkz"])))
    assert "GKZ system" in headings and "Landau surfaces" not in headings


# --- references --------------------------------------------------------------


def test_every_citation_strips_to_plain_ascii() -> None:
    for key, entry in CITATIONS.items():
        plain = _strip_latex(entry)
        assert plain.isascii(), key
        for mark in ("\\", "$", "{", "}", "~", "--"):
            assert mark not in plain, (key, mark)
    assert _strip_latex(CITATIONS["fms2019"]).startswith("J. Forsgaard, L.F. Matusevich")
    assert "Birkhaeuser, 1994." in _strip_latex(CITATIONS["gkz1994"])
    assert "from A-hypergeometric systems" in _strip_latex(CITATIONS["jimenez2026"])
    assert "(2019) 497-564" in _strip_latex(CITATIONS["bbkp2017"])


def test_strip_latex_transliterates_accents_and_drops_wrappers() -> None:
    assert _strip_latex('M\\"uller, K\\"onig, Forsg\\aa rd') == "Mueller, Koenig, Forsgaard"
    assert _strip_latex("\\emph{A title} and \\textit{B}~1--2") == "A title and B 1-2"


def test_references_list_each_entry_stripped(text: str) -> None:
    references = _flat(text.split("\nReferences\n----------\n")[1])
    assert (
        "[fms2019] J. Forsgaard, L.F. Matusevich and A. Sobieska, On transformations of "
        "A-hypergeometric functions, Funkcialaj Ekvacioj 62 (2019) 319, arXiv:1703.03036."
    ) in references


# --- content -----------------------------------------------------------------


def test_every_euler_operator_has_its_line(text: str, triangle_report: AnalysisReport) -> None:
    assert triangle_report.gkz is not None
    operators = [line.strip() for line in text.splitlines() if re.match(r"^\s+E_\d+ = ", line)]
    assert len(operators) == triangle_report.gkz.a_matrix.rows
    assert all("theta_" in line for line in operators)
    assert operators[0] == "E_0 = theta_1 + theta_2 + theta_3 + theta_4 + theta_5 + theta_6 + D/2"
    assert operators[1] == "E_1 = theta_1 + theta_2 + theta_4 + nu_1"


def test_toric_generators_are_listed_with_sstr(text: str, triangle_report: AnalysisReport) -> None:
    assert triangle_report.gkz is not None
    lines = {line.strip() for line in text.splitlines()}
    assert triangle_report.gkz.toric_generators
    for generator in triangle_report.gkz.toric_generators:
        assert sp.sstr(generator) in lines


def test_matrices_are_pretty_printed(text: str, triangle_report: AnalysisReport) -> None:
    assert triangle_report.gkz is not None
    rows = sp.pretty(triangle_report.gkz.a_matrix, use_unicode=False).splitlines()
    gkz = _section(text, "GKZ system")
    for row in rows:
        assert row in gkz
    assert f"A = {rows[len(rows) // 2]}" in gkz


def test_one_convergence_row_per_facet(text: str, triangle_report: AnalysisReport) -> None:
    assert triangle_report.polytope is not None
    rows = [line for line in text.splitlines() if re.match(r"^\s+Re\(.*\) > 0$", line)]
    assert len(rows) == len(triangle_report.polytope.data.facets)
    assert rows[0].strip() == "Re(-D/2 + nu_1 + nu_2 + nu_3) > 0"


def test_symanzik_displays(text: str) -> None:
    polynomials = _section(text, "Symanzik polynomials")
    assert "    U = a_1 + a_2 + a_3\n" in polynomials
    assert "    F = (1/mu**2)*(-a_1*a_2*p1^2 - a_1*a_3*p2^2 - a_2*a_3*p3^2)\n" in polynomials
    assert (
        "    G = U + F = u_1 + u_2 + u_3\n"
        "                + (1/mu**2)*(-p1^2*u_1*u_2 - p2^2*u_1*u_3 - p3^2*u_2*u_3)\n"
    ) in polynomials


def test_long_sums_break_at_their_signs(sunrise_text: str) -> None:
    polynomials = _section(sunrise_text, "Symanzik polynomials")
    display = polynomials.split("    G = U + F = ")[1].split("\n\n")[0].splitlines()
    assert len(display) > 2
    for line in display[1:]:
        assert line.startswith(" " * 16 + "+ ") or line.startswith(" " * 16 + "- "), line


def test_landau_factor_lines_carry_no_energy_scale(sunrise_text: str) -> None:
    landau = _section(sunrise_text, "Landau surfaces")
    rows = [line for line in landau.splitlines() if line.startswith("    ")]
    assert rows
    for row in rows:
        assert "mu" not in row, row
    assert "- Dimension 2:" in landau


def test_representations_write_the_integrands_in_u_f_and_g(text: str) -> None:
    representations = _section(text, "Parametric representations")
    assert "Schwinger representation\n~~~~~~~~~~~~~~~~~~~~~~~~" in representations
    assert "        * exp(-F/U)/U**(D/2)\n" in representations
    assert "        * delta(1 - sum_e a_e)\n" in representations
    assert "        * F**(D/2 - nu)*U**(-D + nu)\n" in representations
    assert "        * G**(-D/2)\n" in representations
    assert "        * int_{R_+^3} da_1 da_2 da_3\n" in representations


# --- golden fragments, the plain-text forms of those in test_report_latex ----


def test_beta_paragraph(text: str) -> None:
    assert (
        "The parameter vector is beta = (-D/2, -nu_1, ..., -nu_N), the value the Euler "
        "equations require for the Lee-Pomeransky integral [delacruz2019, klausen2020]."
    ) in _flat(text)


def test_euler_operators_are_built_from_theta(text: str) -> None:
    assert "built from theta_j = z_j d/dz_j, which measures the degree in z_j" in _flat(text)


def test_conventions_define_the_integral_and_nu(text: str) -> None:
    conventions = _section(text, "Conventions")
    assert (
        "    I = exp(L epsilon gamma_E) (mu^2)^(nu - L D/2)\n"
        "        * int prod_{r=1}^{L} d^D k_r/(i pi^(D/2)) prod_e 1/(-q_e^2 + m_e^2)^(nu_e)\n"
    ) in conventions
    assert "nu = sum_e nu_e, here nu = nu_1 + nu_2 + nu_3." in _flat(conventions)
    assert _flat(text).count("nu = sum_e nu_e") == 1


def test_gkz_section_defines_the_euler_mellin_integral(text: str) -> None:
    assert (
        "    I_A(beta, z) = int_{R_+^N} prod_e du_e u_e^(nu_e - 1) G(z, u)^(-D/2),\n"
    ) in _section(text, "GKZ system")


def test_symmetry_identity_and_note(text: str) -> None:
    symmetries = _flat(_section(text, "Symmetries"))
    assert (
        "Each gives the identity I_A(beta, z_sigma) = I_A(T beta, z) for the Euler-Mellin "
        "integral I_A of the GKZ system section, with z_sigma = (z_{sigma(1)}, ..., "
        "z_{sigma(6)})."
    ) in symmetries
    assert (
        "Forsgaard, Matusevich and Sobieska [fms2019] and de la Cruz [delacruz2024] print the "
        "permutation on the other side, but the substitution in the proof of Corollary 4.1 of "
        "[fms2019] gives the form stated here."
    ) in symmetries
    assert "T_1 = [" in _section(text, "Symmetries")
    assert "], sigma_1 = (1, 2, 3, 4, 5, 6)" in _section(text, "Symmetries")
    assert "and 38 further pairs." in symmetries


def test_symmetry_identity_is_self_contained_without_the_gkz_section(
    triangle: FeynmanIntegral,
) -> None:
    text = render_text(AnalysisReport.from_integral(triangle, ["symmetries"]))
    assert "GKZ system" not in _text_headings(text)
    symmetries = _flat(_section(text, "Symmetries"))
    assert "let I_A(beta, z) = int_{R_+^N}" in symmetries
    assert "be the Lee-Pomeransky integral without its prefactor" in symmetries
    assert "GKZ system section" not in symmetries


def test_convergence_states_the_euclidean_hypothesis(text: str) -> None:
    assert (
        "Convergence. For Euclidean kinematics, where every coefficient of G has positive "
        "real part, and for Re D > 0,"
    ) in _flat(text)


def test_codimension_uses_the_rank_of_a(text: str) -> None:
    assert "codimension 6 - rank A = 6 - 4 = 2" in _flat(text)


def test_rank_paragraph_states_the_volume(text: str) -> None:
    flat = _flat(text)
    assert "here 4 [adolphson1994, chestnov2022]" in flat
    assert "Cohen-Macaulay [mmw2005]" in flat
    assert "feynkit computes neither the holonomic rank" in flat


def test_toric_wording(text: str) -> None:
    flat = _flat(text)
    assert "analogue of integration-by-parts relations" in flat
    assert "Horn" not in flat and "IBP" not in flat


def test_landau_factor_lists(text: str) -> None:
    landau = _section(text, "Landau surfaces")
    blocks = landau.strip().split("\n\n")
    lead = next(
        k for k, block in enumerate(blocks) if _flat(block).endswith("(Cayley) factors are")
    )
    assert blocks[lead + 1].splitlines() == ["    p1^2", "    p2^2", "    p3^2"]
    assert "A factor can arise from both a Cayley minor and a Gram minor" in _flat(landau)


def test_edge_table(text: str) -> None:
    graph = _section(text, "Graph")
    assert "    e  Vertices  nu_e  m_e\n" in graph
    assert "    1  1-2       nu_1  0\n" in graph


def test_schwinger_condition(text: str) -> None:
    assert "here nu_1 + nu_2 + nu_3 = D [britto2026]" in _flat(text)
