"""Tests for the plain-text renderer of the analysis report.

The text report states the same facts as the LaTeX one, in the same order and
with the same citations. The parity tests render both from one
AnalysisReport and compare what each shows: the section headings, the summary
rows and the order in which works are first cited. Golden fragments pin the
plain-text wording of the sentences the LaTeX tests pin.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from itertools import pairwise

import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.io._report_shared import NOT_COMPUTED, TORUS_HEADING
from feynkit.io.report import SECTION_NAMES, AnalysisReport
from feynkit.io.report_latex import CITATIONS, render_latex
from feynkit.io.report_text import _str, _strip_latex, render_text
from feynkit.point_count import TorusCount, count_torus_points

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


@pytest.fixture(scope="module")  # type: ignore[misc]
def kite_text() -> str:
    return render_text(
        AnalysisReport.from_integral(FeynmanIntegral.from_cnickel("12e|23|3|e|:zzzzz"))
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def tadpole() -> FeynmanIntegral:
    # A vacuum graph has no Mandelstam invariants to default to.
    return FeynmanIntegral.from_cnickel("0|:n", use_mandelstam=False)


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


# Names the prose uses for mathematical objects, which the LaTeX writes in
# maths mode and the prose comparison therefore leaves out on both sides.
_MATH_WORDS = frozenset(
    {"sum", "prod", "int", "exp", "gamma", "theta", "beta", "sigma", "mu", "lambda", "infinity"}
    | {"epsilon", "delta", "not", "alpha", "Re", "rank", "Aut", "pi", "nu", "Cayley", "dz"}
    | {"Newt", "Vol"}
)

# What the text says differently on purpose, as substitutions on the LaTeX:
# it has no figures, so the sentences pointing at one go, and a
# cross-reference names the section it points at.
_INTENDED = (
    ("Figure~\\ref{fig:feynman_graph} shows the graph. ", ""),
    (
        "The figure labels each propagator with its index $e$, which also names",
        "Its index $e$ also names",
    ),
    ("Figure~\\ref{fig:newton-polytope} draws it.", ""),
    ("Section~\\ref{sec:newton-polytope}", "the Newton polytope section"),
    ("Section~\\ref{sec:gkz-system}", "the GKZ system section"),
)


def _words(prose: str) -> list[str]:
    """The prose words longer than two letters, with maths and hyphens left out."""
    # A bracketed word standing alone, "(Gram)", is prose; "Aut(P)" is not.
    prose = re.sub(r"(?<!\S)\(([A-Za-z-]+)\)", r"\1", prose)
    tokens = [t for t in prose.split() if not re.search(r"[_^\d(){}=<>/*\[\]]", t)]
    words = re.findall(r"[A-Za-z]+", " ".join(tokens))
    return [w for w in words if len(w) > 2 and w not in _MATH_WORDS]


def _latex_prose(latex: str, intended: tuple[tuple[str, str], ...] = _INTENDED) -> str:
    """The running text of the LaTeX document, without displays, maths or markup."""
    body = latex.split("\\begin{document}")[1].split("\\begin{thebibliography}")[0]
    for old, new in intended:
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    environments = r"equation\*|align\*|gather\*|figure|tabular|longtable"
    body = re.sub(rf"\\begin\{{({environments})\}}.*?\\end\{{\1\}}", " ", body, flags=re.S)
    body = re.sub(r"\$[^$]*\$", " ", body)
    body = re.sub(r"~?\\cite\{[^}]*\}", " ", body)
    body = re.sub(r"\\(?:section|subsection|label|begin|end)\{[^}]*\}", " ", body)
    body = re.sub(r"\\paragraph\{([^}]*)\}", r"\1", body)
    body = body.replace("Forsg\\aa rd", "Forsgaard")
    return re.sub(r"\\[a-zA-Z]+", " ", body).replace("~", " ")


def _text_prose(text: str) -> str:
    """The running text of the text report, without headings, displays or citations."""
    lines = text.split("\nReferences\n----------\n")[0].splitlines()
    prose = [
        line
        for line, following in pairwise([*lines, ""])
        if not line.startswith(" ")
        and not re.fullmatch(r"[-=~]+", line)
        and not (following and re.fullmatch(r"[-=~]+", following))
    ]
    return re.sub(rf"\[{_KEY}(?:, {_KEY})*\]", " ", " ".join(prose))


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
        # Powers are written with ^, as the prose writes them.
        assert "**" not in document
        # A caret-named invariant raised to a power is bracketed: (p1^2)^2.
        assert re.search(r"\w\^\w+\^", document) is None


def test_printer_writes_powers_as_the_prose_does() -> None:
    p1, mu = sp.symbols("p1^2 mu")
    u_1, nu_1, g, d = sp.symbols("u_1 nu_1 G D")
    assert _str(p1**3) == "(p1^2)^3"
    assert _str(p1**2 * u_1) == "(p1^2)^2*u_1"
    assert _str(u_1 ** (nu_1 - 1)) == "u_1^(nu_1 - 1)"
    assert _str(g ** (-d / 2)) == "G^(-D/2)"
    assert _str(mu**2) == "mu^2"
    assert _str(sp.gamma(nu_1)) == "Gamma(nu_1)"
    assert _str(sp.exp(sp.Symbol("epsilon") * sp.Symbol("gamma_E"))) == "exp(epsilon*gamma_E)"
    # Quotients stay readable.
    assert _str(-p1 / mu**2) == "-p1^2/mu^2"
    assert _str(1 / p1) == "1/(p1^2)"
    assert _str(u_1 / p1**2) == "u_1/(p1^2)^2"


def test_title_is_the_cnickel_underlined(text: str) -> None:
    lines = text.splitlines()
    assert lines[0] == "Feynman integral 12e|2e|e|:zzz"
    assert lines[1] == "=" * len(lines[0])


def test_custom_title_is_kept_verbatim(triangle_report: AnalysisReport) -> None:
    lines = render_text(triangle_report, title="one_mass & two").splitlines()
    assert lines[:2] == ["one_mass & two", "=============="]


def test_every_line_fits_in_79_columns(text: str, sunrise_text: str, kite_text: str) -> None:
    # Prose is wrapped; displays and tables break where they can.
    for document in (text, sunrise_text, kite_text):
        for line in document.splitlines():
            assert len(line) <= 79, line


def test_a_long_permutation_gets_a_line_of_its_own(kite_text: str) -> None:
    # The kite's 16 columns make "T_k = [...], sigma_k = (...)" too wide.
    lines = _section(kite_text, "Symmetries").splitlines()
    assert "    sigma_1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)" in lines
    assert not any("], sigma_" in line for line in lines)


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


def test_toric_generators_are_listed(text: str, triangle_report: AnalysisReport) -> None:
    assert triangle_report.gkz is not None
    lines = {line.strip() for line in text.splitlines()}
    assert triangle_report.gkz.toric_generators
    for generator in triangle_report.gkz.toric_generators:
        assert _str(generator) in lines
    assert "z_1*z_6 - z_2*z_5" in lines


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
    assert "    F = (1/mu^2)*(-a_1*a_2*p1^2 - a_1*a_3*p2^2 - a_2*a_3*p3^2)\n" in polynomials
    assert (
        "    G = U + F = u_1 + u_2 + u_3\n"
        "                + (1/mu^2)*(-p1^2*u_1*u_2 - p2^2*u_1*u_3 - p3^2*u_2*u_3)\n"
    ) in polynomials
    assert "    1  u_1*u_2   -p1^2/mu^2\n" in polynomials


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
    assert "        * exp(-F/U)/U^(D/2)\n" in representations
    assert "        * delta(1 - sum_e a_e)\n" in representations
    assert "        * F^(D/2 - nu)*U^(-D + nu)\n" in representations
    assert "        * G^(-D/2)\n" in representations
    assert "        * int_{R_+^3} da_1 da_2 da_3\n" in representations
    assert "        * a_1^(nu_1 - 1)*a_2^(nu_2 - 1)*a_3^(nu_3 - 1)\n" in representations
    # A prefactor too long for one line is split into numerator and denominator.
    assert (
        "    I = exp(epsilon*gamma_E)*Gamma(D/2)\n"
        "        / (Gamma(nu_1)*Gamma(nu_2)*Gamma(nu_3)*Gamma(D - nu_1 - nu_2 - nu_3))\n"
        "        * int_{R_+^3} du_1 du_2 du_3\n"
    ) in representations
    assert "    I = exp(epsilon*gamma_E)/(Gamma(nu_1)*Gamma(nu_2)*Gamma(nu_3))\n" in representations


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
        "        * int prod_{r=1}^{L} d^D k_r/(i pi^(D/2))\n"
        "        * prod_e 1/(-q_e^2 + m_e^2)^nu_e\n"
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
    # The Kallen function of the three external masses, each square bracketed.
    kallen = "(p1^2)^2 - 2*p1^2*p2^2 - 2*p1^2*p3^2 + (p2^2)^2 - 2*p2^2*p3^2 + (p3^2)^2"
    assert blocks[lead + 3].splitlines() == ["    p1^2", "    p2^2", "    p3^2", f"    {kallen}"]
    assert f"- Dimension 3:\n    {kallen}\n" in landau
    assert "A factor can arise from both a Cayley minor and a Gram minor" in _flat(landau)


def test_long_displays_break_to_fit(sunrise_text: str) -> None:
    # The sunrise's second Euler operator is too long for one line.
    gkz = _section(sunrise_text, "GKZ system")
    assert (
        "    E_1 = 2 theta_1 + 2 theta_2 + theta_3 + theta_4 + theta_5 + theta_8\n"
        "          + theta_9 + nu_1\n"
    ) in gkz
    # Vectors break after a comma, continuing under their first entry.
    schwinger = _section(sunrise_text, "Schwinger-representation system")
    assert (
        "    beta_Cayley = (-3*D/2 + nu_1 + nu_2 + nu_3, D - nu_1 - nu_2 - nu_3, -nu_1,\n"
        "                   -nu_2).\n"
    ) in schwinger


def test_a_long_coefficient_continues_in_its_column() -> None:
    # The massless box has a z_j with six terms, too wide for one row.
    box = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
    text = render_text(AnalysisReport.from_integral(box, []))
    table = _section(text, "Symanzik polynomials").split("    j  ")[1].split("\n\n")[0]
    rows = table.splitlines()
    wide = next(k for k, row in enumerate(rows) if row.startswith("    3 "))
    assert rows[wide].startswith("    3   u_1*u_4   -p1^2/mu^2 - p2^2/mu^2")
    assert rows[wide + 1].startswith(" " * 18 + "+ ")
    assert all(len(row) <= 79 for row in rows)


def _prose_diff(
    report: AnalysisReport, intended: tuple[tuple[str, str], ...] = _INTENDED
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """The LaTeX prose words and where the text prose departs from them.

    Both renderers' running text, reduced to words: the LaTeX without its
    maths, displays and markup, the text without its displays, headings and
    citations. Only the differences listed in ``intended`` are allowed.
    """
    latex = _words(_latex_prose(render_latex(report), intended))
    text = _words(_text_prose(render_text(report)))
    diff = [
        (tag, " ".join(latex[i1:i2]), " ".join(text[j1:j2]))
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, latex, text, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]
    return latex, diff


@pytest.mark.parametrize("cnickel", ["12e|2e|e|:zzz", "111e|e|:nnn"])  # type: ignore[misc]
def test_prose_matches_the_latex_word_for_word(cnickel: str) -> None:
    report = AnalysisReport.from_integral(FeynmanIntegral.from_cnickel(cnickel))
    latex, diff = _prose_diff(report)
    assert not diff
    assert len(latex) > 700


def test_tadpole_prose_matches_the_latex_word_for_word(tadpole: FeynmanIntegral) -> None:
    # The cross-references to other sections sit in the symmetry section,
    # which the tadpole leaves out.
    intended = tuple(pair for pair in _INTENDED if not pair[0].startswith("Section"))
    latex, diff = _prose_diff(AnalysisReport.from_integral(tadpole), intended)
    assert not diff
    assert len(latex) > 600


# The figure-eight with one massless loop, whose P is a segment in R^2, and the
# massive triangle with a massless self-loop, whose P has dimension 3 in R^4.
@pytest.mark.parametrize("cnickel", ["00|:nz", "012e|2e|e|:znnn"])  # type: ignore[misc]
def test_dependent_rows_prose_matches_the_latex_word_for_word(cnickel: str) -> None:
    # Both reports leave out the symmetries, and with them the cross-references.
    fi = FeynmanIntegral.from_cnickel(cnickel, use_mandelstam=False)
    intended = tuple(pair for pair in _INTENDED if not pair[0].startswith("Section"))
    assert _prose_diff(AnalysisReport.from_integral(fi), intended)[1] == []


def test_skipped_face_prose_matches_the_latex_word_for_word() -> None:
    sunrise = FeynmanIntegral.from_cnickel("111e|e|:nnn")
    report = AnalysisReport.from_integral(sunrise, max_face_points=4)
    assert _prose_diff(report)[1] == []


# --- the tadpole and other small graphs ---------------------------------------


def test_tadpole_renders_in_both_formats(tadpole: FeynmanIntegral) -> None:
    # Its Newton polytope is a segment, below the dimension Aut(P) is computed for.
    latex = tadpole.to_latex()
    text = tadpole.to_text()
    omitted = (
        "The symmetries are not computed for Newton polytopes of dimension below 2, such as "
        "this one."
    )
    assert omitted in " ".join(latex.split("\\section{Symmetries}")[1].split())
    assert _flat(_section(text, "Symmetries")) == omitted
    assert "It has L = 1 loop, N = 1 propagator and 0 external legs." in _flat(text)
    for line in text.splitlines():
        assert len(line) <= 79, line


def test_symmetries_are_omitted_below_full_dimension_in_both_formats() -> None:
    # P has dimension 3 in R^4, and Aut(P) would miss its symmetries.
    fi = FeynmanIntegral.from_cnickel("012e|2e|e|:znnn")
    latex = fi.to_latex(["symmetries"])
    text = fi.to_text(["symmetries"])
    omitted = (
        "The symmetries are not computed for Newton polytopes that are not full-dimensional, "
        "such as this one."
    )
    assert omitted in " ".join(latex.split("\\section{Symmetries}")[1].split())
    assert _flat(_section(text, "Symmetries")) == omitted


@pytest.mark.parametrize("cnickel", ["00|:nz", "0|:z"])  # type: ignore[misc]
def test_no_rank_is_claimed_when_the_rows_of_a_are_dependent(cnickel: str) -> None:
    # A massless loop is scaleless, and P is not full-dimensional.
    fi = FeynmanIntegral.from_cnickel(cnickel, use_mandelstam=False)
    text = _flat(_section(fi.to_text(["polytope"]), "Newton polytope"))
    latex = " ".join(fi.to_latex(["polytope"]).split("\\section{Newton polytope}")[1].split())
    assert (
        "Since P is not full-dimensional, the rows of A are linearly dependent. For generic "
        "beta the Euler equations are then inconsistent, and the GKZ system has no non-zero "
        "solutions: its holonomic rank is 0, not the normalised volume."
    ) in text
    assert (
        "Since $P$ is not full-dimensional, the rows of $A$ are linearly dependent. For "
        "generic $\\beta$ the Euler equations are then inconsistent"
    ) in latex
    assert text.endswith("its holonomic rank is 0, not the normalised volume.")
    for document in (text, latex):
        assert "equals the normalised volume" not in document
        # For generic beta the rank and the Euler characteristic are both 0, so
        # the report does not call them uncomputed.
        assert "feynkit computes neither" not in document
        assert "Euler characteristic" not in document


def test_tadpole_keeps_its_rank_sentence(tadpole: FeynmanIntegral) -> None:
    # Its segment spans R^1, so the rows of A are independent.
    for document in (tadpole.to_text(["polytope"]), tadpole.to_latex(["polytope"])):
        flat = " ".join(document.split())
        assert "the holonomic rank of the GKZ system equals the normalised volume, here 1" in flat
        assert "linearly dependent" not in flat


def test_master_count_is_bounded_by_the_euclidean_volume(tadpole: FeynmanIntegral) -> None:
    # The bound is N! Vol(P), the normalised volume only when the exponent
    # differences span Z^N: for 1 + x^2, |chi| = 2 and the normalised volume is 1.
    text = _flat(tadpole.to_text(["polytope"]))
    latex = _flat(tadpole.to_latex(["polytope"]))
    assert "at most N! times the Euclidean volume of P, with equality for generic" in text
    assert "The bound equals the normalised volume when the exponent differences span Z^N." in text
    assert "at most $N!$ times the Euclidean volume of $P$, with equality for generic" in latex
    assert "the exponent differences span $\\mathbb{Z}^N$." in latex
    for document in (text, latex):
        assert "at most the normalised volume" not in document


def test_vacuum_graphs_have_no_external_momenta(tadpole: FeynmanIntegral) -> None:
    sunrise = Graph(
        internal_vertices=2,
        external_legs=0,
        edges=[Edge(idx=e, v1=1, v2=2, is_internal=True) for e in (1, 2, 3)],
    )
    for fi in (tadpole, FeynmanIntegral(sunrise, use_mandelstam=False)):
        for document in (fi.to_latex([]), fi.to_text([])):
            flat = " ".join(document.split())
            assert "dimensionless. There are no external momenta. The second Symanzik" in flat
            assert "kinematics is written" not in flat


def test_an_empty_factor_list_ends_its_sentence(tadpole: FeynmanIntegral) -> None:
    # A tadpole has a Cayley factor but no Gram factor.
    for document in (tadpole.to_latex(["landau"]), tadpole.to_text(["landau"])):
        assert "and there are no second-type (Gram) factors." in " ".join(document.split())


def test_skipped_faces_are_named() -> None:
    sunrise = FeynmanIntegral.from_cnickel("111e|e|:nnn")
    for document in (
        sunrise.to_latex(["landau"], max_face_points=4),
        sunrise.to_text(["landau"], max_face_points=4),
    ):
        assert (
            "2 faces were skipped as too large to eliminate, and their discriminants are "
            "missing from the list: a face of dimension 2 with 7 points and the whole "
            "polytope, 10 points."
        ) in " ".join(document.split())


def test_edge_table(text: str) -> None:
    graph = _section(text, "Graph")
    assert "    e  Vertices  nu_e  m_e\n" in graph
    assert "    1  1-2       nu_1  0\n" in graph


def test_schwinger_condition(text: str) -> None:
    assert "here nu_1 + nu_2 + nu_3 = D [britto2026]" in _flat(text)


# --- the point counts --------------------------------------------------------

# The Newton polytope section points at the point-count section, by reference in LaTeX and by
# its heading in the text.
_TORUS_INTENDED = (
    *_INTENDED,
    ("Section~\\ref{sec:torus-counts}", f'the section "{TORUS_HEADING}"'),
)


@pytest.fixture(scope="module")  # type: ignore[misc]
def torus_report() -> AnalysisReport:
    # Every section of the massive bubble, whose counts at seed 0 are p - 4.
    return AnalysisReport.from_integral(FeynmanIntegral.from_cnickel("11e|e|:nn"), SECTION_NAMES)


def _without_candidate(report: AnalysisReport) -> AnalysisReport:
    assert report.torus is not None
    torus = dataclasses.replace(
        report.torus,
        candidate_polynomial=None,
        candidate_euler_characteristic=None,
        candidate_master_count=None,
        reason="the count at p = 41 is 37, where the fit predicts 39",
    )
    return dataclasses.replace(report, torus=torus)


def test_point_counts_follow_the_newton_polytope(
    torus_report: AnalysisReport, triangle_report: AnalysisReport
) -> None:
    headings = _text_headings(render_text(torus_report))
    assert headings[headings.index("Newton polytope") + 1] == TORUS_HEADING
    assert headings == _latex_headings(render_latex(torus_report))
    assert TORUS_HEADING not in _text_headings(render_text(triangle_report))


@pytest.mark.parametrize(  # type: ignore[misc]
    "candidate", [True, False], ids=["candidate", "no-candidate"]
)
def test_point_count_report_agrees_with_the_latex(
    torus_report: AnalysisReport, candidate: bool
) -> None:
    report = torus_report if candidate else _without_candidate(torus_report)
    text, latex = render_text(report), render_latex(report)
    assert _text_summary(text) == _latex_summary(latex) == list(report.summary())
    bibliography = re.findall(r"\\bibitem\{([^}]*)\}", latex)
    assert {"bbkp2017", "katz2008", "fmt2024"} <= set(bibliography)
    assert _first_citations(text.split("\nReferences\n----------\n")[0]) == bibliography
    assert _prose_diff(report, _TORUS_INTENDED)[1] == []
    for line in text.splitlines():
        assert len(line) <= 79, line


def test_uncomputed_sentence_points_at_the_candidate(torus_report: AnalysisReport) -> None:
    text = _flat(_section(render_text(torus_report), "Newton polytope"))
    latex = " ".join(render_latex(torus_report).split())
    assert NOT_COMPUTED not in text
    assert (
        "feynkit does not compute the holonomic rank at the physical point, and gives the Euler "
        "characteristic only as a candidate from the point counts of the section "
        f'"{TORUS_HEADING}".'
    ) in text
    assert "point counts of Section~\\ref{sec:torus-counts}." in latex
    assert "\\label{sec:torus-counts}" in latex


def test_uncomputed_sentence_without_a_candidate(torus_report: AnalysisReport) -> None:
    text = render_text(_without_candidate(torus_report))
    polytope = _flat(_section(text, "Newton polytope"))
    assert (
        "feynkit computes neither the holonomic rank at the physical point nor the Euler "
        "characteristic, and the point counts of the section "
        f'"{TORUS_HEADING}" give no candidate.'
    ) in polytope
    counts = _flat(_section(text, TORUS_HEADING))
    assert (
        "The counts give no candidate: the count at p = 41 is 37, where the fit predicts 39."
    ) in counts
    assert "P(q) =" not in counts


def test_uncomputed_sentence_is_unchanged_without_the_section(text: str) -> None:
    assert NOT_COMPUTED in _flat(_section(text, "Newton polytope"))


def test_point_count_section_states_the_counts(torus_report: AnalysisReport) -> None:
    assert torus_report.torus is not None
    counts = _section(render_text(torus_report), TORUS_HEADING)
    for p, n in torus_report.torus.counts:
        assert re.search(rf"^    {p} +{n} +(fit|check)$", counts, re.M)
    flat = _flat(counts)
    assert "drawn with seed 0" in flat
    assert "C = (-1)^N chi(X) = 3." in flat
    assert "finitely many primes is evidence, not a proof" in flat


# How each test that refuses a candidate begins its reason.
_REASONS = {
    "check": "the count at p = 41 is 37,",
    "bound": "the master count 3 given by the fit is not in [0, N! Vol(Newt G)] = [0, 2]",
    "integer": "the polynomial through the fit counts has non-integer coefficients",
    "degree": "the polynomial through the fit counts has a q^1 term",
    "guard": "the Newton polytope has an edge of lattice length 6,",
}


@pytest.fixture(scope="module")  # type: ignore[misc]
def refusals(torus_report: AnalysisReport) -> dict[str, AnalysisReport]:
    """The bubble's report with counts that give no candidate, one for each test that refuses.

    Every count comes from point_count: the bubble at m_1^2 = 5, m_2^2 = 8 and s = 2, where
    lambda = -39 and the check fails at p = 41; the bubble under a volume bound of 2; and the
    small polynomials of the point-count tests for non-integer coefficients, a q^N term and the
    guard on characters of higher order, which no Feynman graph reaches.
    """
    assert torus_report.torus is not None
    bubble = FeynmanIntegral.from_cnickel("11e|e|:nn")
    lambda_39 = {
        key: value for (key, _), value in zip(torus_report.torus.point, (5, 8, 2), strict=True)
    }
    u, v, x = sp.symbols("u v x")
    counts: dict[str, TorusCount] = {
        "check": bubble.torus_count(point=lambda_39),
        "bound": count_torus_points(
            bubble.symanzik.g,
            bubble.symanzik.lp_parameters,
            scale=bubble.graph.energy_scale,
            volume_bound=2,
        ),
        "integer": count_torus_points(1 + x * u**3 + v, [u, v], point={x: 1}),
        "degree": count_torus_points(u**2 - x, [u], point={x: 11}),
        "guard": count_torus_points(v + u**6 + x, [u, v], point={x: 108}),
    }
    for kind, count in counts.items():
        assert count.reason is not None and count.reason.startswith(_REASONS[kind]), kind
    return {kind: dataclasses.replace(torus_report, torus=count) for kind, count in counts.items()}


@pytest.mark.parametrize("kind", list(_REASONS))  # type: ignore[misc]
def test_a_result_without_a_candidate_says_why(
    refusals: dict[str, AnalysisReport], kind: str
) -> None:
    report = refusals[kind]
    assert report.torus is not None
    text, latex = render_text(report), render_latex(report)
    counts = _flat(_section(text, TORUS_HEADING))
    assert f"The counts give no candidate: {report.torus.reason}." in counts
    assert "P(q) =" not in counts
    # Only a fit that passes the tests before the check is checked at further primes.
    checked = (
        "is checked at further primes, at least until every non-trivial one of these characters "
        "has taken both signs"
    )
    if report.torus.verification_primes:
        assert f"{checked}, unless a count disagrees first:" in counts
        assert "The check covers the quadratic characters (d/p) for d in the group" in counts
    else:
        assert checked not in counts
        assert "The check covers" not in counts
        assert "The fit is not checked at further primes." in counts
    # The bound and the guard refuse counts that may well be polynomial.
    assert f'the section "{TORUS_HEADING}" give no candidate.' in _flat(text)
    assert "point counts of Section~\\ref{sec:torus-counts} give no candidate." in _flat(latex)
    for document in (text, latex):
        assert "not polynomial" not in _flat(document)
    assert "\\textasciicircum" not in latex
    assert _text_summary(text) == _latex_summary(latex) == list(report.summary())
    assert _prose_diff(report, _TORUS_INTENDED)[1] == []
    for line in text.splitlines():
        assert len(line) <= 79, line


def test_the_check_names_the_characters_it_covers(torus_report: AnalysisReport) -> None:
    # The check covers a group of quadratic characters, not every character the counts could
    # depend on, and the trivial character never takes both signs.
    counts = _flat(_section(render_text(torus_report), TORUS_HEADING))
    assert (
        "The check covers the quadratic characters (d/p) for d in the group generated by -1 and "
        "the non-zero values at the point of the vertex coefficients of G, the factors of the "
        "face discriminants, written in the squared masses, and the discriminants of G on its "
        "edges. Characters outside that group are not covered;"
    ) in counts
    assert "at least until every non-trivial one of these characters has taken both signs:" in (
        counts
    )
    assert "could depend on" not in counts
    assert "The counts fit the candidate for P P(q) = q - 4 at every prime counted" in counts


def test_the_squared_masses_are_named_only_with_a_mass(
    torus_report: AnalysisReport, triangle: FeynmanIntegral
) -> None:
    # A factor odd in a mass is covered through its norm, written in the squared masses; the
    # massless triangle has no mass to write it in.
    massless = AnalysisReport.from_integral(triangle, SECTION_NAMES)
    for report, massive in ((torus_report, True), (massless, False)):
        assert report.torus is not None and report.torus.verification_primes
        text = _flat(_section(render_text(report), TORUS_HEADING))
        latex = render_latex(report)
        assert ("the face discriminants, written in the squared masses, and the" in text) is massive
        assert ("the face discriminants and the discriminants of G" in text) is not massive
        assert ("written in the squared masses" in latex) is massive
        assert _prose_diff(report, _TORUS_INTENDED)[1] == []


def test_the_primes_left_out_are_stated_up_to_the_largest_prime(
    torus_report: AnalysisReport,
) -> None:
    assert torus_report.torus is not None
    counts = _flat(_section(render_text(torus_report), TORUS_HEADING))
    assert (
        "A prime is left out when it is 2 or divides the numerator or the denominator of a "
        "non-zero value at the point of a coefficient of G, a face discriminant, a factor of one "
        "or the discriminant of G on an edge. Up to 1000 the primes left out are 2, 3 and 5."
    ) in counts
    assert "roots in F_p^* of quadratics" in counts
    assert torus_report.torus.max_prime == 1000


# The forms of the section's account of the point and of what the count leaves out, as
# substitutions in the bubble's point counts, with a sentence each must state.
_P1 = sp.Symbol("p1^2", real=True)
_VARIANTS: dict[str, tuple[dict[str, object], str]] = {
    "given": (
        {"seed": None},
        "The counts are taken at the kinematic point m_1^2 = 3, m_2^2 = 18 and s = 6, given by "
        "the caller.",
    ),
    "singular": (
        {"seed": None, "on_landau_surface": True},
        "given by the caller. A coefficient of G or a face discriminant vanishes at the point, "
        "where C can be smaller than for generic kinematics.",
    ),
    "no-symbols": (
        {"point": ()},
        "G has no kinematic symbols, so the counts need no drawn point.",
    ),
    "on-shell": (
        {"on_shell": ((_P1, 0),)},
        "The counts set p1^2 = 0 in the momentum products first. The counts are taken at the "
        "kinematic point",
    ),
    "on-shell-no-symbols": (
        {"on_shell": ((_P1, 0),), "point": ()},
        "The counts set p1^2 = 0 in the momentum products first. G then has no kinematic "
        "symbols, so the counts need no drawn point.",
    ),
    "skipped": (
        {"skipped_faces": 2},
        "The Landau analysis skipped 2 faces, whose discriminants are left out of the draw, the "
        "choice of excluded primes and the check.",
    ),
    "skipped-unchecked": (
        {
            "seed": None,
            "skipped_faces": 1,
            "verification_primes": (),
            "counts": ((7, 3), (11, 7), (13, 9)),
            "candidate_polynomial": None,
            "candidate_euler_characteristic": None,
            "candidate_master_count": None,
            "reason": "the polynomial through the fit counts has non-integer coefficients",
        },
        "The Landau analysis skipped 1 face, whose discriminant is left out of the test of the "
        "given point and the choice of excluded primes.",
    ),
    "one-prime": ({"excluded_primes": (2,)}, "Up to 1000 only 2 is left out."),
    "max-prime": ({"max_prime": 500}, "Up to 500 the primes left out are 2, 3 and 5."),
}


@pytest.mark.parametrize("variant", list(_VARIANTS))  # type: ignore[misc]
def test_every_form_of_the_point_count_section(torus_report: AnalysisReport, variant: str) -> None:
    assert torus_report.torus is not None
    changes, sentence = _VARIANTS[variant]
    torus = dataclasses.replace(torus_report.torus, **changes)
    report = dataclasses.replace(torus_report, torus=torus)
    assert report.torus is not None
    text, latex = render_text(report), render_latex(report)
    counts = _flat(_section(text, TORUS_HEADING))
    assert sentence in counts
    drawn = bool(report.torus.point) and report.torus.seed is not None
    assert ("drawn with seed 0" in counts) is drawn
    assert ("fmt2024" in re.findall(r"\\bibitem\{([^}]*)\}", latex)) is drawn
    # Without symbols there is no point, and nothing is said to hold at one.
    assert ("at the point" in counts) is bool(report.torus.point)
    assert _prose_diff(report, _TORUS_INTENDED)[1] == []
    for line in text.splitlines():
        assert len(line) <= 79, line
