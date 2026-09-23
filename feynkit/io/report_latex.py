"""
LaTeX rendering of an analysis report.

:func:`render_latex` turns an :class:`AnalysisReport` into a complete
``article`` document: a summary table, the graph, the conventions, the
Symanzik polynomials, the parametric representations with the convergence
region, the Newton polytope with the rank statement, the GKZ system, the
symmetries, the Landau surfaces, the Schwinger-representation system and the
references. Each section is rendered by its own function and omitted when the
report does not carry it. Citations and the width of the widest matrix are
collected while the sections are rendered, so the bibliography lists exactly
the works the text cites, in the order of first citation, and the preamble
allows exactly as many matrix columns as the document needs. What the text
renderer must state the same way, the section list, the bibliography, the
integrand templates and the Landau factors, comes from ``_report_shared``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import sympy as sp

from ._report_shared import (
    CITATIONS,
    MAX_PAIRS_SHOWN,
    Citations,
    append_signed,
    count_noun,
    face_names,
    integrand_templates,
    join_words,
    landau_factors,
    render_sections,
    signed_terms,
    split_g,
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
    Schwinger,
    Symmetries,
)

__all__ = ["CITATIONS", "render_latex"]


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


def _factor_lines(factors: Sequence[sp.Expr]) -> str:
    """An ``align*`` with each factor on its own line, long factors broken further."""
    rows = []
    for factor in factors:
        lines = to_latex_lines(factor, max_length=100)
        rows.append("&" + lines[0])
        rows.extend("&\\quad {}" + line for line in lines[1:])
    return "\\begin{align*}\n" + " \\\\\n".join(rows) + "\n\\end{align*}"


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
            f"$N = {identity.propagators}$ propagators and "
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
            "dimensionless. External momenta are incoming, and the kinematics is written in "
            f"{_kinematics(conventions)}. The second Symanzik polynomial is "
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


def _polytope(polytope: Polytope, doc: _Document) -> str:
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
        "$\\{G = 0\\}$ in the torus, at most the normalised volume and equal to it for "
        "generic coefficients; graph-polynomial coefficients are rarely generic"
        f"{doc.cite('bbkp2017')}. feynkit computes neither the holonomic rank at the physical "
        "point nor the Euler characteristic, and produces no series solutions, Pfaffian "
        "system or restriction to physical kinematics.",
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


def _symmetries(report: AnalysisReport, symmetries: Symmetries, doc: _Document) -> str:
    orbits = symmetries.vertex_orbits
    if report.polytope is not None:
        listed = join_words(
            ["$\\{" + ", ".join(f"v_{{{i + 1}}}" for i in orbit) + "\\}$" for orbit in orbits]
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
    parts = [
        "The group $\\mathrm{Aut}(P)$ of unimodular affine maps taking $P$ to itself has "
        f"order {symmetries.automorphism_order} and {action}. The graph has "
        f"{count_noun(len(symmetries.graph_automorphisms), 'automorphism')}. Of the polytope "
        f"automorphisms, {preserving} {'preserves' if preserving == 1 else 'preserve'} the "
        "coefficients of $G$.",
    ]
    if not pairs:
        parts.append("The configuration has no symmetry pairs.")
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
        shared = factors.in_both
        if shared:
            text += (
                "\nA factor can arise from both a Cayley minor and a Gram minor, which is why "
                f"{count_noun(shared, 'factor')} {'appears' if shared == 1 else 'appear'} in both "
                "lists."
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
    skipped = len(landau.analysis.skipped_faces)
    if skipped:
        parts.append(
            f"{count_noun(skipped, 'face')} {'was' if skipped == 1 else 'were'} skipped as too "
            "large to eliminate, and "
            f"{'its discriminant is' if skipped == 1 else 'their discriminants are'} missing "
            "from the list."
        )
    return "\n".join(parts)


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
        polytope=lambda section: _polytope(section, doc),
        gkz=lambda section: _gkz(section, doc),
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
