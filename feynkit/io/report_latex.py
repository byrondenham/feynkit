"""
LaTeX rendering of an analysis report.

:func:`render_latex` turns an :class:`AnalysisReport` into a complete
``article`` document: a summary table, the graph, the conventions, the
Symanzik polynomials, the parametric representations with the convergence
region, the Newton polytope with the rank statement, the candidate Euler
characteristic from finite-field point counts, the GKZ system, the symmetries,
the Landau surfaces, the Schwinger-representation system and the references.
Each section is rendered by its own function and omitted when the report does
not carry it. Citations and the width of the widest matrix are
collected while the sections are rendered, so the bibliography lists exactly
the works the text cites, in the order of first citation, and the preamble
allows exactly as many matrix columns as the document needs. What the text
renderer must state the same way, the section list, the bibliography, the
integrand templates and the Landau factors, comes from ``_report_shared``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction

import sympy as sp

from ..point_count import TorusCount
from ._report_shared import (
    CITATIONS,
    MAX_PAIRS_SHOWN,
    Citations,
    append_signed,
    count_noun,
    count_polynomial,
    face_names,
    in_squared_masses,
    integrand_templates,
    join_words,
    kinematic_class_sentence,
    landau_factors,
    lattice_normality,
    limit_factors,
    limit_sentences,
    not_computed,
    primes_left_out,
    render_sections,
    resonance_forms,
    resonance_paragraphs,
    resonance_rows,
    resonance_windows,
    signed_terms,
    skipped_faces,
    split_g,
    torus_skipped_faces,
)
from .latex import to_latex, to_latex_lines, to_latex_split
from .report import (
    GKZ,
    AnalysisReport,
    Conventions,
    Identity,
    Landau,
    Polytope,
    Representations,
    Resonance,
    Schwinger,
    Symmetries,
)

__all__ = ["CITATIONS", "render_latex"]


# N! Vol(Newt G), the bound on the master count, as the point-count section sets it.
_VOLUME_BOUND = "N!\\,\\mathrm{Vol}(\\mathrm{Newt}\\,G)"

# How the Newton polytope section refers to the point-count section.
_TORUS_REFERENCE = "Section~\\ref{sec:torus-counts}"

# The maths in point_count's reasons for no candidate, which it writes as plain text: a
# power of q, a prime p = 41, the bound [0, N! Vol(Newt G)] = [0, 6], any other interval
# and a negative number.
_REASON_MATHS = re.compile(
    r"q\^\d+|p = \d+|\[0, N! Vol\(Newt G\)\] = \[-?\d+, -?\d+\]|\[-?\d+, -?\d+\]|(?<![\w-])-\d+"
)

# amsmath matrices stop at this many columns unless MaxMatrixCols is raised.
_AMSMATH_MATRIX_COLUMNS = 10

_ESCAPES: dict[int, str] = {
    ord("\\"): "\\textbackslash{}",
    ord("&"): "\\&",
    ord("%"): "\\%",
    ord("$"): "\\$",
    ord("#"): "\\#",
    ord("_"): "\\_",
    ord("{"): "\\{",
    ord("}"): "\\}",
    ord("~"): "\\textasciitilde{}",
    ord("^"): "\\textasciicircum{}",
}

# The Lee-Pomeransky integral without its prefactor, the function the GKZ
# system annihilates.
_EULER_MELLIN = (
    "I_A(\\beta, z) = \\int_{\\mathbb{R}_{+}^{N}} \\prod_{e} \\mathrm{d}u_e\\, "
    "u_e^{\\nu_e - 1}\\, G(z, u)^{-D/2}"
)


# --- small helpers -----------------------------------------------------------


class _Document(Citations):
    """What the preamble and the bibliography need to know about the sections.

    The sections cite through :meth:`cite` and print matrices through
    :meth:`matrix`, so that the bibliography lists the works cited, in the
    order of first citation, and the preamble allows the widest matrix.
    """

    def __init__(self) -> None:
        super().__init__()
        self.widest_matrix = 0

    def cite(self, *keys: str) -> str:
        self.add(*keys)
        return "~\\cite{" + ",".join(keys) + "}"

    def matrix(self, matrix: sp.Matrix) -> str:
        self.widest_matrix = max(self.widest_matrix, matrix.cols)
        rows = [
            " & ".join(to_latex(matrix[r, c]) for c in range(matrix.cols))
            for r in range(matrix.rows)
        ]
        return "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"

    def preamble(self) -> str:
        columns = max(_AMSMATH_MATRIX_COLUMNS, self.widest_matrix)
        return "\n".join(
            [
                "\\documentclass[11pt,a4paper]{article}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage[T1]{fontenc}",
                "\\usepackage{lmodern}",
                "\\usepackage{amsmath,amssymb}",
                f"\\setcounter{{MaxMatrixCols}}{{{columns}}}",
                "\\usepackage[margin=2.5cm]{geometry}",
                "\\usepackage{booktabs,array,longtable}",
                "\\usepackage{tikz}",
                "\\usetikzlibrary{calc}",
                "\\usepackage{tikz-3dplot}",
                "\\usepackage[hidelinks,pdfusetitle]{hyperref}",
                "\\allowdisplaybreaks",
            ]
        )

    def bibliography(self) -> str:
        items = [f"\\bibitem{{{key}}} {CITATIONS[key]}" for key in self.keys]
        return "\\begin{thebibliography}{99}\n" + "\n".join(items) + "\n\\end{thebibliography}"


def _escape(text: str) -> str:
    """Escape the characters that LaTeX treats specially in running text."""
    return text.translate(_ESCAPES)


def _reason(reason: str) -> str:
    """A reason for no candidate, escaped, with its maths set as maths."""
    pieces = []
    start = 0
    for match in _REASON_MATHS.finditer(reason):
        maths = re.sub(r"\^(\d+)", r"^{\1}", match.group())
        maths = maths.replace("N! Vol(Newt G)", _VOLUME_BOUND)
        pieces += [_escape(reason[start : match.start()]), f"${maths}$"]
        start = match.end()
    pieces.append(_escape(reason[start:]))
    return "".join(pieces)


def _math(expr: sp.Expr) -> str:
    return "$" + to_latex(expr) + "$"


def _vector(entries: Sequence[sp.Expr]) -> str:
    return "\\left(" + ", ".join(to_latex(e) for e in entries) + "\\right)"


def _scaled_lines(expr: sp.Expr, scale: sp.Expr) -> list[str]:
    """The lines of ``expr`` written as ``(1/scale)(...)``."""
    lines = to_latex_lines(expr)
    lines[0] = f"\\frac{{1}}{{{to_latex(scale)}}}\\bigl({lines[0]}"
    lines[-1] += "\\bigr)"
    return lines


def _equation(lhs: str, lines: Sequence[str]) -> str:
    """Display ``lhs = ...`` over the given lines, aligned after the equals sign."""
    if len(lines) == 1:
        body = f"{lhs} = {lines[0]}"
    else:
        continued = "".join(f" \\\\\n&\\quad {{}}{line}" for line in lines[1:])
        body = f"\\begin{{split}}\n{lhs} &= {lines[0]}{continued}\n\\end{{split}}"
    return "\\begin{equation*}\n" + body + "\n\\end{equation*}"


# TeX sets an align* whole before breaking it across pages, and the massless hexagon's list of
# over 9,000 lines ran out of pdflatex's memory; 3,967 lines still fitted.
_DISPLAY_LINES = 500


def _factor_lines(factors: Sequence[sp.Expr]) -> str:
    """``align*`` displays with each factor on its own line, long factors broken further.

    A list longer than _DISPLAY_LINES lines is split into several displays,
    between factors where one fits, and a longer factor across displays.
    """
    blocks: list[list[str]] = [[]]
    for factor in factors:
        lines = to_latex_lines(factor, max_length=100)
        rows = ["&" + lines[0]] + ["&\\quad {}" + line for line in lines[1:]]
        if blocks[-1] and len(blocks[-1]) + len(rows) > _DISPLAY_LINES:
            blocks.append([])
        for row in rows:
            if len(blocks[-1]) == _DISPLAY_LINES:
                blocks.append([])
            blocks[-1].append(row)
    return "\n".join(
        "\\begin{align*}\n" + " \\\\\n".join(block) + "\n\\end{align*}" for block in blocks
    )


def _factor_list(kind: str, factors: Sequence[sp.Expr]) -> str:
    """'the <kind> factors are' followed by their display, or a note that there are none."""
    if not factors:
        return f"there are no {kind} factors"
    return f"the {kind} factors are\n" + _factor_lines(factors)


# --- sections ----------------------------------------------------------------


def _summary(report: AnalysisReport) -> str:
    rows = []
    for label, value in report.summary():
        text = re.sub(r"\b([FG])\b", r"$\1$", _escape(label))
        rows.append(f"{text} & {_escape(value)} \\\\")
    return "\n".join(
        [
            "\\begin{center}",
            "\\begin{tabular}{ll}",
            "\\toprule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{center}",
        ]
    )


def _graph(identity: Identity) -> str:
    label = re.search(r"\\label\{([^}]*)\}", identity.graph_tikz)
    shown = f"Figure~\\ref{{{label.group(1)}}} shows the graph. " if label else ""
    rows = [
        f"{e} & {v1}--{v2} & {_math(nu)} & {_math(mass)} \\\\"
        for e, (v1, v2), nu, mass in zip(
            identity.edge_indices,
            identity.edge_endpoints,
            identity.edge_exponents,
            identity.edge_masses,
            strict=True,
        )
    ]
    return "\n".join(
        [
            identity.graph_tikz,
            "",
            f"{shown}It has $L = {identity.loop_count}$ "
            f"{'loop' if identity.loop_count == 1 else 'loops'}, "
            f"$N = {identity.propagators}$ "
            f"{'propagator' if identity.propagators == 1 else 'propagators'} and "
            f"{count_noun(identity.external_legs, 'external leg')}.",
            "\\begin{center}",
            "\\begin{tabular}{cccc}",
            "\\toprule",
            "$e$ & Vertices & $\\nu_e$ & $m_e$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{center}",
            "Propagator $e$ joins the vertices listed and carries the exponent $\\nu_e$ and the "
            "mass $m_e$ shown in the table; massless propagators have $m_e = 0$. The figure "
            "labels each propagator with its index $e$, which also names its parameters $a_e$ "
            "and $u_e$.",
        ]
    )


def _kinematics(conventions: Conventions) -> str:
    invariants = conventions.invariants
    if invariants is None:
        return "the dot products $p_i \\cdot p_j$"
    symbols = list(dict.fromkeys(invariants.symbols))
    listed = join_words([_math(s) for s in symbols])
    noun = "invariant" if len(symbols) == 1 else "invariants"
    if invariants.n_external == 2:
        return f"the {noun} {listed}: $s = p^2$ for $p = p_1 = -p_2$, so that $p_1 \\cdot p_2 = -s$"
    clause = "the external masses $p_i^2$"
    if invariants.mandelstam:
        clause += " and the planar invariants $s_{i \\ldots j-1} = (p_i + \\cdots + p_{j-1})^2$"
    return f"the {noun} {listed}: {clause}"


def _conventions(report: AnalysisReport, doc: _Document) -> str:
    conventions = report.conventions
    dimension = conventions.dimension
    is_d = isinstance(dimension, sp.Symbol) and dimension.name == "D"
    in_d = "$D$" if is_d else f"$D = {to_latex(dimension)}$"
    nu_total = sp.Add(*report.identity.edge_exponents)
    if report.identity.external_legs:
        momenta = (
            "External momenta are incoming, and the kinematics is written in "
            f"{_kinematics(conventions)}."
        )
    else:
        momenta = "There are no external momenta."
    kinematic_class = kinematic_class_sentence(
        conventions,
        report.identity.external_legs,
        report.identity.edge_masses,
        name=lambda text: f"\\texttt{{{_escape(text)}}}",
        math=lambda text: f"${text}$",
        printer=_math,
    )
    return "\n".join(
        [
            "The integral is",
            "\\begin{equation*}",
            "I = e^{L \\epsilon \\gamma_E} \\left(\\mu^2\\right)^{\\nu - L D/2} \\int "
            "\\prod_{r=1}^{L} \\frac{\\mathrm{d}^D k_r}{i \\pi^{D/2}} \\prod_{e} "
            "\\frac{1}{\\left(-q_e^2 + m_e^2\\right)^{\\nu_e}}",
            "\\end{equation*}",
            f"in {in_d} dimensions{doc.cite('weinzierl2022')}, with loop momenta $k_r$, the "
            "momentum $q_e$ of propagator $e$ fixed by momentum conservation at the vertices, "
            "the metric signature $(+,-,\\ldots,-)$ and $\\nu = \\sum_e \\nu_e$, here "
            f"$\\nu = {to_latex(nu_total)}$. Dimensional regularisation sets "
            "$D = D_0 - 2\\epsilon$ for an even integer $D_0$ chosen when the result is "
            "expanded. The energy scale $\\mu$ makes $I$ and the Symanzik polynomials "
            f"dimensionless. {momenta} {kinematic_class} The second Symanzik polynomial is "
            "$F = -\\sum_{T_2} s_{T_2} \\prod_{e \\notin T_2} a_e + U \\sum_e m_e^2 a_e$, "
            "divided by $\\mu^2$, the sum running over spanning two-forests $T_2$ and "
            "$s_{T_2}$ being the square of the momentum flowing from one tree of $T_2$ to the "
            f"other{doc.cite('weinzierl2022')}.",
        ]
    )


def _polynomials(report: AnalysisReport, doc: _Document) -> str:
    polynomials = report.polynomials
    power = polynomials.f_scale_power
    scale = report.conventions.energy_scale
    if power:
        f_lines = _scaled_lines(polynomials.f_numerator, scale**power)
        u_part, f_part, g_power = split_g(polynomials.g, scale)
        g_f_lines = _scaled_lines(f_part, scale**g_power)
        g_lines = [*to_latex_lines(u_part), "+ " + g_f_lines[0], *g_f_lines[1:]]
    else:
        f_lines = to_latex_lines(polynomials.f_numerator)
        g_lines = to_latex_lines(polynomials.g)
    rows = [
        f"{entry.index} & {_math(entry.monomial)} & {_math(entry.coefficient)} \\\\"
        for entry in polynomials.z_table
    ]
    rank = polynomials.monomials_g - polynomials.codimension
    return "\n".join(
        [
            "In the Schwinger parameters $a_e$ the first Symanzik polynomial, the sum over "
            "spanning trees $T$ of $\\prod_{e \\notin T} a_e$, is",
            _equation("U", to_latex_lines(polynomials.u)),
            "and the second is",
            _equation("F", f_lines),
            "In the Lee-Pomeransky parameters $u_e$ their sum is",
            _equation("G = U + F", g_lines),
            f"$U$ has degree $L = {polynomials.degree_u}$ and $F$ degree "
            f"$L+1 = {polynomials.degree_f}$; $F$ has "
            f"{count_noun(polynomials.monomials_f, 'monomial')} and $G$ has "
            f"{polynomials.monomials_g}. The coefficients of $G$ are the GKZ variables $z_j$, "
            "here at their physical values:",
            "\\begin{longtable}{rll}",
            "\\toprule",
            "$j$ & Monomial & $z_j$ \\\\",
            "\\midrule",
            "\\endhead",
            "\\bottomrule",
            "\\endlastfoot",
            *rows,
            "\\end{longtable}",
            f"The configuration has codimension ${polynomials.monomials_g} - "
            f"\\operatorname{{rank}} A = {polynomials.monomials_g} - {rank} = "
            f"{polynomials.codimension}$, with $A$ the matrix whose columns are the exponent "
            "vectors of the monomials of $G$ below a row of ones, against "
            f"{count_noun(polynomials.independent_invariants, 'independent kinematic invariant')} "
            f"in the coefficients{doc.cite('klausen2023')}.",
        ]
    )


def _representation(
    prefactor: sp.Expr,
    parameters: Sequence[sp.Symbol],
    measure: sp.Expr,
    constraint: str,
    integrand: sp.Expr,
) -> str:
    differentials = "\\, ".join(f"\\mathrm{{d}}{to_latex(p)}" for p in parameters)
    factors = [f for f in (constraint, "" if measure == 1 else to_latex(measure)) if f]
    factors.append(to_latex(integrand, fold_short_frac=True))
    return "\n".join(
        [
            "\\begin{equation*}",
            "\\begin{split}",
            f"I &= {to_latex(prefactor)} \\\\",
            f"&\\quad \\times \\int_{{\\mathbb{{R}}_{{+}}^{{{len(parameters)}}}}} "
            f"{differentials}\\; " + "\\, ".join(factors),
            "\\end{split}",
            "\\end{equation*}",
        ]
    )


def _representations(
    report: AnalysisReport, representations: Representations, doc: _Document
) -> str:
    schwinger = representations.schwinger
    feynman = representations.feynman
    lee_pomeransky = representations.lee_pomeransky
    templates = integrand_templates(report.identity.loop_count, report.conventions.dimension)
    # The Schwinger result names its parameters alpha_e; the document uses a_e
    # throughout, the names of the Feynman result and of U and F.
    rename = dict(zip(schwinger.parameters, feynman.parameters, strict=True))

    parts = [
        "\\subsection{Schwinger representation}",
        "With the Schwinger parameters $a_e$ integrated over $[0, \\infty)$"
        f"{doc.cite('weinzierl2022')},",
        _representation(
            schwinger.prefactor,
            feynman.parameters,
            schwinger.measure.subs(rename),
            "",
            templates.schwinger,
        ),
        "\\subsection{Feynman representation}",
        "With the same parameters restricted to the simplex $\\sum_e a_e = 1$"
        f"{doc.cite('weinzierl2022')},",
        _representation(
            feynman.prefactor,
            feynman.parameters,
            feynman.measure,
            "\\delta\\Bigl(1 - \\sum_{e} a_{e}\\Bigr)",
            templates.feynman,
        ),
        "\\subsection{Lee-Pomeransky representation}",
        "With the Lee-Pomeransky parameters $u_e$ integrated over $[0, \\infty)$"
        f"{doc.cite('leepomeransky2013')},",
        _representation(
            lee_pomeransky.prefactor,
            lee_pomeransky.parameters,
            lee_pomeransky.measure,
            "",
            templates.lee_pomeransky,
        ),
        "\\paragraph{Convergence.}",
    ]
    euclidean = "For Euclidean kinematics, where every coefficient of $G$ has positive real part"
    if representations.convergence is None:
        parts.append(
            f"The Newton polytope $P$ of $G$ is not full-dimensional. {euclidean}, the "
            "Lee-Pomeransky integral therefore converges absolutely for no $D$ and $\\nu_e$"
            f"{doc.cite('klausen2023')}."
        )
        return "\n".join(parts)
    lines = " \\\\\n".join(
        f"\\mathrm{{Re}}\\left({to_latex(c)}\\right) &> 0" for c in representations.convergence
    )
    parts += [
        f"{euclidean}, and for $\\mathrm{{Re}}\\,D > 0$, the Lee-Pomeransky integral converges "
        "absolutely when the real parts of $(\\nu_1, \\ldots, \\nu_N)$, divided by "
        "$\\mathrm{Re}(D/2)$, lie in the interior of the Newton polytope $P$ of $G$, that is "
        "when for every facet $m_j \\cdot x \\le b_j$ of $P$ the real part of "
        f"$b_j D/2 - \\sum_e m_{{je}} \\nu_e$ is positive{doc.cite('klausen2023')}:",
        "\\begin{align*}",
        lines,
        "\\end{align*}",
        "For such kinematics it converges absolutely for no $D$ and $\\nu_e$ when $P$ is not "
        "full-dimensional.",
    ]
    return "\n".join(parts)


def _polytope(report: AnalysisReport, polytope: Polytope, doc: _Document) -> str:
    data = polytope.data
    counts = [
        count_noun(n, singular, plural)
        for n, (singular, plural) in zip(
            data.f_vector[:-1], face_names(data.dimension), strict=True
        )
    ]
    shape = f", with {join_words(counts)}" if counts else ""
    vertices = join_words(
        [
            f"$v_{{{k}}} = ({', '.join(str(x) for x in vertex)})$"
            for k, vertex in enumerate(data.vertices, start=1)
        ]
    )
    volume = data.normalized_volume
    parts = [
        "\\label{sec:newton-polytope}",
        "The Newton polytope $P$ of $G$ is the convex hull of the exponent vectors of its "
        f"monomials. It has dimension {data.dimension} in "
        f"$\\mathbb{{R}}^{{{data.ambient_dimension}}}$ and normalised volume {volume}{shape}. "
        f"Its vertices are {vertices}.",
    ]
    if polytope.figure is not None:
        parts += [
            "Figure~\\ref{fig:newton-polytope} draws it.",
            "\\begin{figure}[htbp]",
            "\\centering",
            polytope.figure,
            "\\caption{The Newton polytope $P$ of $G$.}",
            "\\label{fig:newton-polytope}",
            "\\end{figure}",
        ]

    def lattice() -> list[str]:
        # Called where the paragraph goes, so that its works are cited in reading order.
        return [] if polytope.invariants is None else ["", _lattice(polytope, doc)]

    if not data.is_full_dimensional:
        parts += [
            "",
            "Since $P$ is not full-dimensional, the rows of $A$ are linearly dependent. For "
            "generic $\\beta$ the Euler equations are then inconsistent, and the GKZ system has "
            "no non-zero solutions: its holonomic rank is 0, not the normalised volume.",
            *lattice(),
            "",
            _scaleless(report, doc),
        ]
        return "\n".join(parts)
    parts += [
        "",
        "For generic coefficients and non-resonant $\\beta$ the holonomic rank of the GKZ "
        "system equals the normalised volume, here "
        f"{volume}{doc.cite('adolphson1994', 'chestnov2022')}. The rank equals the volume for "
        "every $\\beta$ exactly when the toric ring $\\mathbb{C}[\\mathbb{N}A]$ is "
        f"Cohen-Macaulay{doc.cite('mmw2005')}. This holds when the graph is one-particle "
        "irreducible and one-vertex irreducible, its external momenta are generic enough that "
        "no monomial of $G$ cancels, and its propagators are all massive, all massless, or "
        "such that every vertex reaches an external leg along massive propagators alone; the "
        "configuration is then normal and so Cohen-Macaulay"
        f"{doc.cite('klausen2023', 'tellander2023', 'walther2022')}. The number of master "
        "integrals is, up to sign, the Euler characteristic of the complement of "
        "$\\{G = 0\\}$ in the torus, at most $N!$ times the Euclidean volume of $P$, with "
        "equality for generic coefficients; graph-polynomial coefficients are rarely generic"
        f"{doc.cite('bbkp2017')}. The bound equals the normalised volume when the exponent "
        f"differences span $\\mathbb{{Z}}^N$. {not_computed(report, _TORUS_REFERENCE)}",
        *lattice(),
    ]
    return "\n".join(parts)


def _lattice(polytope: Polytope, doc: _Document) -> str:
    """The lattice invariants of $P$ and what they certify about $\\mathbb{C}[\\mathbb{N}A]$."""
    found = polytope.invariants
    assert found is not None
    counts = f"{found.lattice_points} lattice points ({found.interior_points} interior)"
    if found.h_star is None:
        shape = f"{counts} and"
    else:
        # A long h* vector may break after any comma.
        h_star = ",\\allowbreak ".join(str(h) for h in found.h_star)
        shape = f"{counts}, $h^*$ vector $({h_star})$ and"
    parts = [
        f"In the lattice generated by its points, $P$ has {shape} lattice width "
        f"{found.lattice_width}{doc.cite('beckrobins2015')}."
    ]
    index = found.gorenstein_index
    if index == 1:
        parts.append(
            "It is reflexive: its one interior lattice point lies at lattice distance 1 from "
            f"every facet{doc.cite('batyrev1994')}."
        )
    elif index is not None:
        parts.append(
            f"It is Gorenstein of index {index}: ${index}P$ has a lattice point at lattice "
            "distance 1 from every facet."
        )
    else:
        parts.append(
            "It is not Gorenstein: no dilate of $P$ has a lattice point at lattice distance 1 "
            "from every facet."
        )
    parts.append(lattice_normality(polytope, doc.cite, latex=True))
    return " ".join(parts)


def _rational(value: Fraction) -> str:
    return to_latex(sp.Rational(value.numerator, value.denominator))


def _torus(torus: TorusCount, doc: _Document) -> str:
    intro = (
        "Let $X$ be the complement of $V = \\{G = 0\\}$ in the torus $(\\mathbb{C}^*)^N$, with "
        "$\\mu = 1$. The number of master integrals, with subsectors included, symmetries unused "
        f"and $D$ symbolic, is $C = (-1)^N \\chi(X)$ by Corollary~37 of{doc.cite('bbkp2017')}. "
        f"By Theorem~44 of the same paper, $C$ is at most ${_VOLUME_BOUND}$, where "
        "$\\mathrm{Newt}\\,G$ is the Newton polytope of $G$ and $\\mathrm{Vol}$ its Euclidean "
        "volume. "
        "If the number of points of $V$ in $(\\mathbb{F}_q^*)^N$ is a polynomial $P(q)$ for "
        "every finite field $\\mathbb{F}_q$ whose characteristic avoids a finite set, then "
        "$\\chi(V) = P(1)$ by Theorem~6.1.2(3) of Katz's appendix to the paper of Hausel and "
        f"Rodriguez-Villegas{doc.cite('katz2008')}, and $\\chi(X) = -P(1)$, since the Euler "
        "characteristic is additive and vanishes on the torus."
    )
    origin = []
    if torus.on_shell:
        settings = join_words([f"${to_latex(key)} = {to_latex(v)}$" for key, v in torus.on_shell])
        origin.append(f"The counts set {settings} in the momentum products first.")
    if not torus.point:
        then = " then" if torus.on_shell else ""
        origin.append(f"$G${then} has no kinematic symbols, so the counts need no drawn point.")
    else:
        where = join_words([f"${to_latex(key)} = {_rational(v)}$" for key, v in torus.point])
        if torus.seed is None:
            origin.append(
                f"The counts are taken at the kinematic point {where}, given by the caller."
            )
        else:
            origin.append(
                f"The counts are taken at the kinematic point {where}, drawn with seed "
                f"{torus.seed} so that every coefficient of $G$ and every face discriminant is "
                "non-zero. The value of $|\\chi(X)|$ is the same on an open dense set of "
                f"kinematics{doc.cite('fmt2024')}, and the draw avoids the Landau surfaces found, "
                "on which it can be smaller."
            )
    if torus.on_landau_surface:
        origin.append(
            "A coefficient of $G$ or a face discriminant vanishes at the point, where $C$ can be "
            "smaller than for generic kinematics."
        )
    skipped = torus_skipped_faces(torus)
    if skipped is not None:
        origin.append(skipped)
    at = " at the point" if torus.point else ""
    n = len(torus.variables)
    if torus.verification_primes:
        # A fit that fails no test before the check; the check stops at a disagreement.
        stops = "" if torus.candidate_polynomial is not None else ", unless a count disagrees first"
        fit = (
            "The check covers the quadratic characters $(d/p)$ for $d$ in the group generated by "
            f"$-1$ and the non-zero values{at} of the vertex coefficients of $G$, the factors of "
            f"the face discriminants{in_squared_masses(torus)} and the discriminants of $G$ "
            "on its edges. Characters outside that group are not covered; they can come from the "
            "constant factors of the discriminants of faces of dimension 2 or more, or from the "
            f"discriminants of skipped faces. The polynomial of degree at most {n} through the "
            f"counts at the first {n + 1} primes not left out is checked at further primes, at "
            "least until every non-trivial one of these characters has taken both signs"
            f"{stops}:"
        )
    else:
        fit = (
            f"The counts at the first {n + 1} primes not left out determine a polynomial of "
            f"degree at most {n}:"
        )
    fit_primes = set(torus.fit_primes)
    rows = [f"{p} & {c} & {'fit' if p in fit_primes else 'check'} \\\\" for p, c in torus.counts]
    parts = [
        "\\label{sec:torus-counts}",
        intro,
        "",
        " ".join(origin),
        "",
        f"$G$ is solved for ${to_latex(torus.eliminated)}$, in which it has degree at most 2, so "
        "each count is a sum of numbers of roots in $\\mathbb{F}_p^*$ of quadratics. A prime is "
        "left out when it is 2 or divides the numerator or the denominator of a non-zero value"
        f"{at} of a coefficient of $G$, a face discriminant, a factor of one or the discriminant "
        f"of $G$ on an edge. {primes_left_out(torus)}",
        "",
        fit,
        "\\begin{center}",
        "\\begin{tabular}{rrl}",
        "\\toprule",
        "$p$ & $\\#V(\\mathbb{F}_p)$ & Use \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{center}",
    ]
    if torus.candidate_polynomial is None:
        refusal = f"The counts give no candidate: {_reason(torus.reason or '')}."
        if not torus.verification_primes:
            refusal += " The fit is not checked at further primes."
        parts.append(refusal)
    else:
        parts += [
            "The counts fit the candidate for $P$",
            _equation("P(q)", to_latex_lines(count_polynomial(torus.candidate_polynomial))),
            "at every prime counted, which gives the candidates "
            f"$\\chi(X) = -P(1) = {torus.candidate_euler_characteristic}$ and "
            f"$C = (-1)^N \\chi(X) = {torus.candidate_master_count}$.",
        ]
    parts += [
        "",
        "A fit on finitely many primes is evidence, not a proof: Katz's theorem needs the count "
        "to be polynomial for every finite field of all but finitely many characteristics"
        f"{doc.cite('katz2008')}. The rule that excludes primes is heuristic, and a bad prime it "
        "misses would, provided some check prime is good, make the fit or its check fail rather "
        "than give a wrong candidate.",
    ]
    return "\n".join(parts)


def _gkz(gkz: GKZ, doc: _Document) -> str:
    operators = []
    for r, (row, beta_r) in enumerate(gkz.euler_rows):
        terms = signed_terms([(a, f"\\theta_{{{j}}}") for j, a in enumerate(row, start=1)])
        operators.append(f"E_{{{r}}} &= {append_signed(terms, -beta_r, to_latex)}")
    parts = [
        "\\label{sec:gkz-system}",
        "Writing $G = \\sum_j z_j u^{\\alpha_j}$ and treating the coefficients $z_j$ as "
        "independent variables gives the Euler-Mellin integral",
        "\\begin{equation*}",
        _EULER_MELLIN + ",",
        "\\end{equation*}",
        "the Lee-Pomeransky integral without its prefactor, which the GKZ system annihilates. "
        "The system has the matrix",
        "\\begin{equation*}",
        "A = " + doc.matrix(gkz.a_matrix),
        "\\end{equation*}",
        "whose column $j$ is $(1, \\alpha_j)$. The parameter vector is "
        "$\\beta = (-D/2, -\\nu_1, \\ldots, -\\nu_N)$, the value the Euler equations require "
        f"for the Lee-Pomeransky integral{doc.cite('delacruz2019', 'klausen2020')}. Here",
        "\\begin{equation*}",
        "\\beta = " + _vector(gkz.beta) + ".",
        "\\end{equation*}",
        "The Euler operators are built from $\\theta_j = z_j \\partial_{z_j}$, which measures "
        "the degree in $z_j$. The operators $E_r = \\sum_j A_{rj}\\theta_j - \\beta_r$ "
        "annihilate $I_A$: row $0$ states that $I_A$ is homogeneous of degree $-D/2$ in the "
        "$z_j$, and row $i$ that it has degree $-\\nu_i$ under "
        "$z_j \\mapsto \\lambda^{A_{ij}} z_j$, which a rescaling of $u_i$ absorbs:",
        "\\begin{align*}",
        " \\\\\n".join(operators),
        "\\end{align*}",
    ]
    generators = gkz.toric_generators
    if not generators:
        parts.append("The toric ideal is trivial.")
        return "\n".join(parts)
    parts += [
        f"The toric ideal of $A$ is generated by {count_noun(len(generators), 'binomial')}. Each "
        "binomial $z^{k} - z^{l}$, with $Ak = Al$, gives the operator "
        "$\\partial^{k} - \\partial^{l}$; these annihilators of $I_A$ form an analogue of "
        f"integration-by-parts relations for it{doc.cite('chestnov2022')}, and the "
        "specialisation to physical coefficients is a separate step. The generators are",
        "\\begin{gather*}",
        " \\\\\n".join(to_latex_split(g) for g in generators),
        "\\end{gather*}",
    ]
    return "\n".join(parts)


def _scaleless(report: AnalysisReport, doc: _Document) -> str:
    """Whether the integral is scaleless by Lee's criterion, for a $P$ that is not full-dimensional."""
    cite = doc.cite("lee2013")
    if report.polynomials.scaleless:
        return (
            "The origin does not lie in the affine hull of $P$, so the integral is scaleless"
            f"{cite}: for an equation $h_0 + h \\cdot x = 0$ of the affine hull with "
            "$h_0 \\neq 0$, substituting $\\lambda^{h_e} u_e$ for $u_e$ multiplies the "
            "Lee-Pomeransky integral by $\\lambda^{h_0 D/2 + \\sum_e h_e \\nu_e}$, and "
            "dimensional regularisation sets it to zero."
        )
    return (
        "The origin lies in the affine hull of $P$, so the integral is not scaleless by "
        f"Lee's criterion{cite}: for every equation $h \\cdot x = 0$ of the affine hull, "
        "substituting $\\lambda^{h_e} u_e$ for $u_e$ multiplies the Lee-Pomeransky integral by "
        "$\\lambda^{\\sum_e h_e \\nu_e}$, which does not involve $D$, and dimensional "
        "regularisation does not regulate it."
    )


def _longtable(spec: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A longtable with booktabs rules; header None for a table without one."""
    head = ["\\toprule", " & ".join(header) + " \\\\", "\\midrule"] if header else ["\\toprule"]
    return "\n".join(
        [
            f"\\begin{{longtable}}{{{spec}}}",
            *head,
            "\\endhead",
            "\\bottomrule",
            "\\endlastfoot",
            *(" & ".join(row) + " \\\\" for row in rows),
            "\\end{longtable}",
        ]
    )


def _resonance(report: AnalysisReport, section: Resonance, doc: _Document) -> str:
    paragraphs, classes, window = resonance_paragraphs(section, doc.cite, latex=True)
    parts = list(paragraphs)
    if not section.facets:
        return "\n\n".join([*parts, "$P$ is a point and has no facets."])
    forms = [
        (f"$F_{{{k}}}$", f"${to_latex(lhs)} \\le {b}$", _math(form))
        for k, lhs, b, form in resonance_forms(report, section)
    ]
    rows = [
        (f"$F_{{{k}}}$", *row) for k, row in enumerate(resonance_rows(section, latex=True), start=1)
    ]
    parts += [
        _longtable("lll", ("Facet", "Inequality", "$l_F(\\beta)$"), forms),
        classes,
        _longtable(
            "lllll",
            ("Facet", "Resonant for", "At $0$", "Admissible at", "Reducible"),
            rows,
        ),
    ]
    windows = resonance_windows(section, latex=True)
    if windows:
        parts += [window, _longtable("ll", (), [(f"$F_{{{k}}}$", v) for k, v in windows])]
    return "\n\n".join(parts)


def _symmetries(report: AnalysisReport, symmetries: Symmetries, doc: _Document) -> str:
    orbits = symmetries.vertex_orbits
    if report.polytope is not None:
        # A long orbit may break after any of its commas.
        listed = join_words(
            [
                "$\\{" + ",\\allowbreak ".join(f"v_{{{i + 1}}}" for i in orbit) + "\\}$"
                for orbit in orbits
            ]
        )
        action = (
            f"acts on the vertices listed in Section~\\ref{{sec:newton-polytope}} with "
            f"{'the orbit' if len(orbits) == 1 else 'orbits'} {listed}"
        )
    else:
        sizes = join_words([str(len(orbit)) for orbit in orbits])
        action = (
            f"acts on the vertices of $P$ with {count_noun(len(orbits), 'orbit')} of "
            f"{'size' if len(orbits) == 1 else 'sizes'} {sizes}"
        )
    if report.gkz is not None:
        definition = ""
        integral = " for the Euler-Mellin integral $I_A$ of Section~\\ref{sec:gkz-system}"
    else:
        definition = (
            f"Write $G = \\sum_j z_j u^{{\\alpha_j}}$ and let ${_EULER_MELLIN}$ be the "
            "Lee-Pomeransky integral without its prefactor, where "
            "$\\beta = (-D/2, -\\nu_1, \\ldots, -\\nu_N)$. "
        )
        integral = ""
    preserving = symmetries.coefficient_preserving
    pairs = symmetries.symmetry_pairs
    group = (
        "unimodular affine maps taking $P$ to itself"
        if symmetries.full_dimensional
        else "affine maps of the affine hull of $P$ that preserve its integer points and take "
        "$P$ to itself"
    )
    parts = [
        f"The group $\\mathrm{{Aut}}(P)$ of {group} has "
        f"order {symmetries.automorphism_order} and {action}. The graph has "
        f"{count_noun(len(symmetries.graph_automorphisms), 'automorphism')}. Of the polytope "
        f"automorphisms, {preserving} {'preserves' if preserving == 1 else 'preserve'} the "
        "coefficients of $G$.",
    ]
    if not pairs:
        parts.append("The configuration has no symmetry pairs.")
        return "\n".join(parts)
    if not symmetries.full_dimensional:
        parts.append(
            f"{definition}The configuration has {count_noun(len(pairs), 'symmetry pair')} "
            "$(T, \\sigma)$: integer matrices $T$ and permutations $\\sigma$ of the columns of "
            "$A$ such that $T$ takes column $j$ of $A$ to column $\\sigma(j)$. Since $P$ is not "
            "full-dimensional, the identities $I_A(\\beta, z_\\sigma) = I_A(T\\beta, z)$"
            f"{integral} that they give hold only trivially: the integral converges absolutely "
            "for no $D$ and $\\nu_e$, $T\\beta$ depends on how $T$ is extended off the affine "
            "hull of $P$, and for generic $D$ and $\\nu_e$ the GKZ system has no non-zero "
            "solutions."
        )
        return "\n".join(parts)
    shown = pairs[:MAX_PAIRS_SHOWN]
    n_columns = report.polynomials.monomials_g
    parts.append(
        f"{definition}The configuration has {count_noun(len(pairs), 'symmetry pair')} "
        "$(T, \\sigma)$: "
        "integer matrices $T$ and permutations $\\sigma$ of the columns of $A$ such that $T$ "
        "takes column $j$ of $A$ to column $\\sigma(j)$. Each gives the identity "
        f"$I_A(\\beta, z_\\sigma) = I_A(T\\beta, z)${integral}, with "
        f"$z_\\sigma = (z_{{\\sigma(1)}}, \\ldots, z_{{\\sigma({n_columns})}})$. "
        f"Forsg\\aa rd, Matusevich and Sobieska{doc.cite('fms2019')} and de la Cruz"
        f"{doc.cite('delacruz2024')} print the permutation on the other side, but the "
        f"substitution in the proof of Corollary~4.1 of{doc.cite('fms2019')} gives the form "
        "stated here. For $I$ itself the prefactors at $\\beta$ and $T\\beta$ enter as well. "
        + ("They are" if len(pairs) == len(shown) else f"The first {len(shown)} are")
        + ", with $\\sigma$ listed as $(\\sigma(1), \\sigma(2), \\ldots)$:"
    )
    for k, pair in enumerate(shown, start=1):
        sigma = ", ".join(str(j + 1) for j in pair.column_permutation)
        parts += [
            "\\begin{equation*}",
            f"T_{{{k}}} = {doc.matrix(pair.homogenized_map)}, \\qquad \\sigma_{{{k}}} = ({sigma})",
            "\\end{equation*}",
        ]
    if len(pairs) > len(shown):
        rest = len(pairs) - len(shown)
        parts.append(f"and {rest} further {'pair' if rest == 1 else 'pairs'}.")
    return "\n".join(parts)


def _landau(landau: Landau, scale: sp.Symbol, doc: _Document) -> str:
    intro = (
        "The reduced principal $A$-determinant of $G$, each irreducible kinematic factor taken "
        "once, is the product of the discriminants of $G$ restricted to the faces of $P$"
        f"{doc.cite('gkz1994', 'dhpt2023', 'fmt2023')}."
    )
    factors = landau_factors(landau, scale)
    parts: list[str] = []
    if factors.by_dimension:
        parts += [
            intro + " Each discriminant that is not identically one factorises as follows, by "
            "face dimension, with one factor per line; numerical factors and powers of the "
            "energy scale $\\mu$, which only normalise the coefficients $z_j$, are left out:",
            "\\begin{itemize}",
        ]
        for dimension, listed in factors.by_dimension:
            parts += [f"\\item Dimension {dimension}:", _factor_lines(listed)]
        parts.append("\\end{itemize}")
    else:
        parts.append(intro + " No face discriminant has a kinematic factor.")
    if factors.closed_form:
        text = (
            "By comparison with the closed form from the modified Cayley matrix"
            f"{doc.cite('dhpt2023')}, {_factor_list('first-type (Cayley)', factors.first_type)}"
            f"\nand {_factor_list('second-type (Gram)', factors.second_type)}"
        )
        if not factors.second_type:
            text += "."
        shared = factors.in_both
        if shared:
            text += (
                "\nA factor can arise from both a Cayley minor and a Gram minor, which is why "
                f"{count_noun(shared, 'factor')} {'appears' if shared == 1 else 'appear'} in both "
                "lists."
            )
        if factors.bridge_poles:
            lead = (
                "The bridge pole, which the faces give as well, is"
                if len(factors.bridge_poles) == 1
                else "The bridge poles, which the faces give as well, are"
            )
            text += (
                "\nA propagator on no loop is a bridge. The closed form is that of the graph's "
                "cycle, with the legs of each tree attached to the cycle moved to the vertex "
                "where the tree meets it, and each bridge $b$ adds the pole $m_b^2 = q_b^2$ of "
                f"its propagator, $q_b$ being the momentum through it. {lead}\n"
                + _factor_lines(factors.bridge_poles)
            )
        parts.append(text)
    parts.append(
        "The factors are candidate codimension-one singular loci on all sheets of the "
        "integral: a point on one of them may or may not be singular on the physical sheet, "
        f"and the list is not guaranteed complete{doc.cite('fmt2023')}. The coefficients are "
        "specialised to physical kinematics before each face discriminant is computed, so "
        "beyond one loop this is the principal Landau determinant rather than the principal "
        "$A$-determinant of the generic polynomial; multiplicities are dropped."
    )
    skipped = skipped_faces(landau)
    if skipped is not None:
        parts.append(skipped)
    parts += _limits(landau, doc)
    return "\n".join(parts)


def _limits(landau: Landau, doc: _Document) -> list[str]:
    """The paragraphs on the limit surfaces and candidates, none without a parent family."""
    limits = limit_factors(landau)
    if limits is None:
        return []
    maths = {
        "cite": doc.cite("fmt2024"),
        "function": "$\\sum_e \\nu_e \\log u_e - (D/2) \\log G$",
        "complement": "the complement of $\\{G = 0\\}$ in the torus",
        "chi": "$|\\chi|$",
    }
    sentences = limit_sentences(limits)
    parts = ["", sentences.intro.format(**maths)]
    if sentences.surfaces is not None:
        parts += [
            sentences.surfaces.format(**maths),
            _factor_lines([record.surface for record in limits.surfaces]),
            sentences.counts or "",
        ]
    if sentences.candidates is not None:
        parts += [
            sentences.candidates,
            _factor_lines([record.surface for record in limits.candidates]),
            _escape(sentences.reasons or ""),
        ]
    if sentences.closing is not None:
        parts.append(sentences.closing)
    return parts


def _schwinger(report: AnalysisReport, schwinger: Schwinger, doc: _Document) -> str:
    system = schwinger.system
    f_block = schwinger.f_block
    loops = report.identity.loop_count
    condition = sp.Eq(
        sp.Add(*report.identity.edge_exponents),
        (loops + 1) * report.conventions.dimension / 2,
        evaluate=False,
    )
    check = "verified" if schwinger.columns_match else "not verified"
    return "\n".join(
        [
            "The Schwinger representation gives a second GKZ system on the Cayley "
            "configuration of $(\\tilde U, \\tilde F)$, the Symanzik polynomials with "
            "$a_N = 1$ and $a_e = u_e$ otherwise"
            f"{doc.cite('jimenez2026', 'klausen2023')}:",
            "\\begin{gather*}",
            "A_{\\text{Cayley}} = " + doc.matrix(system.a_matrix) + ", \\\\",
            "\\beta_{\\text{Cayley}} = " + _vector(system.beta_parameters) + ".",
            "\\end{gather*}",
            "It is unimodularly equivalent to the Lee-Pomeransky configuration, the matrix "
            "$A_{\\text{LP}}$ and parameter vector "
            "$\\beta_{\\text{LP}} = (-D/2, -\\nu_1, \\ldots, -\\nu_N)$ of the GKZ system of $G$, "
            "through",
            "\\begin{equation*}",
            "T = " + doc.matrix(schwinger.lp_to_cayley) + ",",
            "\\end{equation*}",
            "which maps the parameter vectors as $\\beta_{\\text{Cayley}} = T "
            "\\beta_{\\text{LP}}$ and the columns of $A_{\\text{LP}}$ to those of "
            "$A_{\\text{Cayley}}$ up to order; the column correspondence is "
            f"{check} for this integral. Restricting to the $\\tilde F$ block gives the system",
            "\\begin{gather*}",
            "A_F = " + doc.matrix(f_block.a_matrix) + ", \\\\",
            "\\beta_F = " + _vector(f_block.beta_parameters) + ";",
            "\\end{gather*}",
            "its solutions solve the full system when $\\beta_{\\text{Cayley}}$ lies in the span "
            "of the face's columns, which for the $\\tilde F$ block means $\\nu = (L+1)D/2$, "
            f"here ${to_latex(condition)}${doc.cite('britto2026')}. Away from that value the "
            "relation between the two systems is not established.",
        ]
    )


# --- the document ------------------------------------------------------------


def render_latex(report: AnalysisReport, *, title: str | None = None) -> str:
    """Render an analysis report as a complete LaTeX ``article``.

    Parameters
    ----------
    report
        The report to render; sections it does not carry are omitted.
    title
        Document title, escaped for LaTeX; by default "Feynman integral"
        followed by the CNickel string.

    Returns
    -------
    str
        The document source, ASCII only, compilable with pdflatex.
    """
    doc = _Document()
    if title is None:
        heading = f"Feynman integral \\texttt{{{_escape(report.identity.cnickel)}}}"
    else:
        heading = _escape(title)

    sections = render_sections(
        report,
        summary=lambda: _summary(report),
        graph=lambda: _graph(report.identity),
        conventions=lambda: _conventions(report, doc),
        polynomials=lambda: _polynomials(report, doc),
        representations=lambda section: _representations(report, section, doc),
        polytope=lambda section: _polytope(report, section, doc),
        torus=lambda section: _torus(section, doc),
        gkz=lambda section: _gkz(section, doc),
        resonance=lambda section: _resonance(report, section, doc),
        symmetries=lambda section: _symmetries(report, section, doc),
        landau=lambda section: _landau(section, report.conventions.energy_scale, doc),
        schwinger=lambda section: _schwinger(report, section, doc),
    )

    document = [
        doc.preamble(),
        "",
        f"\\title{{{heading}}}",
        "\\date{}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\n\n".join(f"\\section{{{name}}}\n{body}" for name, body in sections),
        "",
        doc.bibliography(),
        "",
        "\\end{document}",
        "",
    ]
    return "\n".join(document)
