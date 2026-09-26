"""
Plain-text rendering of an analysis report.

:func:`render_text` states the facts of :func:`~feynkit.io.report_latex.render_latex`
in the same order, with the same sentences and citations, as plain ASCII:
maths is written out in plain names (``beta = (-D/2, -nu_1, ..., -nu_N)``),
computed expressions are printed like ``sympy.sstr`` but with ``^`` for
powers, as the prose writes them, matrices by ``sympy.pretty`` in ASCII, and
citations as ``[key]``, listed at the end with their entries stripped of
LaTeX. The text has no figures, so the sentences
that point at a figure are left out and a cross-reference names the section
it refers to. Prose is wrapped at 79 columns without breaking inline maths;
displays are indented and broken to fit the same width wherever a sum, a
product, a quotient or a vector allows it.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence

import sympy as sp
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter

from ._report_shared import (
    CITATIONS,
    MAX_PAIRS_SHOWN,
    NOT_COMPUTED,
    Citations,
    append_signed,
    count_noun,
    face_names,
    integrand_templates,
    join_words,
    landau_factors,
    render_sections,
    signed_terms,
    skipped_faces,
    split_g,
    symmetries_omitted,
)
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

__all__ = ["render_text"]


_WIDTH = 79
_INDENT = "    "

# Operators that stand as words in the prose. The wrapper keeps them on the
# line of their operands, joining them with a no-break space that textwrap
# does not split at.
_OPERATORS = frozenset({"=", "+", "-", ".", "<=", "->"})
_GLUE = "\xa0"

# Accent macros used in the bibliography, as ASCII.
_ACCENTS = {
    '\\"a': "ae",
    '\\"o': "oe",
    '\\"u': "ue",
    "\\aa ": "aa",
}

# The Lee-Pomeransky integral without its prefactor, the function the GKZ
# system annihilates.
_EULER_MELLIN = "I_A(beta, z) = int_{R_+^N} prod_e du_e u_e^(nu_e - 1) G(z, u)^(-D/2)"


# --- small helpers -----------------------------------------------------------


class _TextPrinter(StrPrinter):
    """``sympy.sstr`` in the notation of the report's prose.

    Powers are written with ``^``, and an exponent other than a symbol or a
    non-negative integer is bracketed: ``u_1^(nu_1 - 1)``, ``G^(-D/2)``. A
    symbol whose name contains ``^``, such as the invariant ``p1^2``, is
    bracketed as the base of a power, so that its square reads ``(p1^2)^2``
    rather than ``p1^2^2``. The gamma function is written ``Gamma``, apart
    from the constant ``gamma_E``. Products keep ``*``.
    """

    def _print_Pow(self, expr: sp.Pow, rational: bool = False) -> str:
        if expr.exp is sp.S.Half and not rational:
            return f"sqrt({self._print(expr.base)})"
        if expr.is_commutative and -expr.exp is sp.S.Half and not rational:
            return f"1/sqrt({self._print(expr.base)})"
        if expr.is_commutative and expr.exp is sp.S.NegativeOne:
            return f"1/{self._base(expr.base)}"
        return f"{self._base(expr.base)}^{self._exponent(expr.exp)}"

    def _base(self, base: sp.Expr) -> str:
        if _caret_named(base):
            return f"({self._print(base)})"
        return str(self.parenthesize(base, PRECEDENCE["Pow"], strict=False))

    def _exponent(self, exponent: sp.Expr) -> str:
        if (isinstance(exponent, sp.Symbol) and not _caret_named(exponent)) or (
            exponent.is_Integer and exponent >= 0
        ):
            return str(self._print(exponent))
        return f"({self._print(exponent)})"

    def _print_gamma(self, expr: sp.Expr) -> str:
        return f"Gamma({self._print(expr.args[0])})"


def _caret_named(expr: sp.Expr) -> bool:
    return isinstance(expr, sp.Symbol) and "^" in expr.name


_str = _TextPrinter().doprint


class _Document(Citations):
    """The works the sections cite, for the references at the end."""

    def cite(self, *keys: str) -> str:
        self.add(*keys)
        return " [" + ", ".join(keys) + "]"

    def references(self) -> str:
        return "\n".join(
            textwrap.fill(
                f"[{key}] {_strip_latex(CITATIONS[key])}",
                width=_WIDTH,
                subsequent_indent=_INDENT,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for key in self.keys
        )


def _strip_latex(text: str) -> str:
    """A bibliography entry in plain ASCII.

    Accent macros are transliterated (``\\"a`` to ae, ``\\aa`` to aa), the
    ``\\emph``, ``\\textit`` and ``\\mathcal`` wrappers and dollar signs are
    dropped, ``~`` becomes a space and ``--`` a hyphen.
    """
    for macro, letters in _ACCENTS.items():
        text = text.replace(macro, letters)
    # Innermost wrappers first, so that one nested in another comes off too.
    unwrapped = None
    while unwrapped != text:
        unwrapped = text
        text = re.sub(r"\\(?:emph|textit|mathcal)\{([^{}]*)\}", r"\1", text)
    return text.replace("$", "").replace("~", " ").replace("--", "-")


def _heading(text: str, rule: str) -> str:
    return f"{text}\n{rule * len(text)}"


def _paragraph(text: str) -> str:
    """Wrap prose at 79 columns without breaking inline maths.

    A line never breaks inside brackets or next to an operator standing as a
    word of its own, so ``m_j . x <= b_j`` and ``(nu_1, ..., nu_N)`` each
    stay on one line.
    """
    units: list[str] = []
    depth = 0
    joined = False
    for word in text.split(" "):
        if units and (joined or depth > 0 or word in _OPERATORS):
            units[-1] += _GLUE + word
        else:
            units.append(word)
        depth = max(0, depth + sum(map(word.count, "([{")) - sum(map(word.count, ")]}")))
        joined = word in _OPERATORS
    wrapped = textwrap.fill(
        " ".join(units), width=_WIDTH, break_long_words=False, break_on_hyphens=False
    )
    return wrapped.replace(_GLUE, " ")


def _blocks(*blocks: str) -> str:
    """Paragraphs and displays, separated by blank lines."""
    return "\n\n".join(blocks)


def _break(text: str, width: int, separators: tuple[str, ...] = (" + ", " - ")) -> list[str]:
    """Break printed maths into lines of at most ``width`` characters where it can.

    A line breaks only before one of ``separators`` standing outside every
    pair of brackets, the top-level signs of a sum by default, and the pieces
    are packed greedily. A piece longer than ``width`` keeps a line of its
    own. Every line after the first starts with its separator.
    """
    if len(text) <= width:
        return [text]
    pieces: list[str] = []
    depth = 0
    start = 0
    for i, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif depth == 0 and i > start and text.startswith(separators, i):
            pieces.append(text[start:i])
            start = i
    pieces.append(text[start:])

    lines = [pieces[0]]
    for piece in pieces[1:]:
        if len(lines[-1]) + len(piece) > width:
            lines.append(piece.strip())
        else:
            lines[-1] += piece
    return lines


def _lines(expr: sp.Expr, width: int) -> list[str]:
    """``expr`` over lines of at most ``width``: a sum at its signs, then a long
    term at its top-level products and quotients."""
    return [
        piece for line in _break(_str(expr), width) for piece in _break(line, width, ("*", "/"))
    ]


def _quotient_lines(expr: sp.Expr, width: int) -> list[str]:
    """``expr`` over lines, a quotient too long for one line split at its bar."""
    text = _str(expr)
    if len(text) <= width:
        return [text]
    numerator, denominator = sp.fraction(expr)
    if denominator == 1:
        return _lines(expr, width)
    below = _lines(denominator, width - 4)
    below = [f"/ ({below[0]}", *(f"   {line}" for line in below[1:])]
    below[-1] += ")"
    return [*_lines(numerator, width), *below]


def _scaled_lines(expr: sp.Expr, scale: sp.Expr, width: int) -> list[str]:
    """The lines of ``expr`` written as ``(1/scale)*(...)``."""
    prefix = f"(1/{_str(scale)})*("
    lines = _lines(expr, width - len(prefix) - 1)
    lines[0] = prefix + lines[0]
    lines[-1] += ")"
    return lines


def _room(lhs: str) -> int:
    """The width left for the right-hand side of a display of ``lhs = ...``."""
    return _WIDTH - len(_INDENT) - len(lhs) - 3


def _equation(lhs: str, lines: Sequence[str]) -> str:
    """Display ``lhs = ...`` over the given lines, continued below the first term."""
    continued = " " * (len(_INDENT) + len(lhs) + 3)
    return "\n".join([f"{_INDENT}{lhs} = {lines[0]}", *(continued + line for line in lines[1:])])


def _expressions(expressions: Sequence[sp.Expr]) -> str:
    """Display each expression on its own line, long ones continued further in."""
    rows = []
    for expr in expressions:
        text = _str(expr)
        whole = len(_INDENT) + len(text) <= _WIDTH
        lines = [text] if whole else _lines(expr, _WIDTH - 2 * len(_INDENT))
        rows.append(_INDENT + lines[0])
        rows.extend(_INDENT * 2 + line for line in lines[1:])
    return "\n".join(rows)


def _matrix(lhs: str, matrix: sp.Matrix, after: str = "") -> str:
    """Display ``lhs = matrix``, with ``lhs`` and ``after`` on the middle row."""
    rows = sp.pretty(matrix, use_unicode=False, wrap_line=False).splitlines()
    middle = len(rows) // 2
    head = f"{lhs} = "
    lines = [
        _INDENT + (head if k == middle else " " * len(head)) + row + (after if k == middle else "")
        for k, row in enumerate(rows)
    ]
    return "\n".join(line.rstrip() for line in lines)


def _vector(lhs: str, entries: Sequence[sp.Expr], end: str) -> str:
    """Display ``lhs = (e_1, e_2, ...)`` then ``end``, broken after commas to fit."""
    width = _room(lhs) - len(end)
    printed = [_str(e) for e in entries]
    lines = ["(" + printed[0]]
    for entry in printed[1:]:
        if len(lines[-1]) + len(entry) + 3 > width:
            lines[-1] += ","
            lines.append(" " + entry)
        else:
            lines[-1] += ", " + entry
    lines[-1] += ")" + end
    return _equation(lhs, lines)


def _table(header: Sequence[str] | None, rows: Sequence[Sequence[str]]) -> str:
    """An indented table with columns padded to their widest cell.

    The last column takes what is left of the line; an entry too long for it
    is broken at its top-level signs and continued below in the same column.
    """
    cells = ([list(header)] if header else []) + [list(row) for row in rows]
    widths = [max(len(row[c]) for row in cells) for c in range(len(cells[0]) - 1)]
    lead = len(_INDENT) + sum(width + 2 for width in widths)
    lasts = [_break(row[-1], _WIDTH - lead) for row in cells]
    widths.append(max(len(line) for last in lasts for line in last))

    def line(row: Sequence[str]) -> str:
        padded = "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        return (_INDENT + padded).rstrip()

    lines = []
    for k, (row, last) in enumerate(zip(cells, lasts, strict=True)):
        lines.append(line([*row[:-1], last[0]]))
        lines.extend(" " * lead + more for more in last[1:])
        if header and k == 0:
            lines.append(line(["-" * width for width in widths]))
    return "\n".join(lines)


def _factor_list(kind: str, factors: Sequence[sp.Expr]) -> tuple[str, str | None]:
    """'the <kind> factors are' and their display, or a note that there are none."""
    if not factors:
        return f"there are no {kind} factors", None
    return f"the {kind} factors are", _expressions(factors)


# --- sections ----------------------------------------------------------------


def _summary(report: AnalysisReport) -> str:
    return _table(None, report.summary())


def _graph(identity: Identity) -> str:
    rows = [
        (str(e), f"{v1}-{v2}", _str(nu), _str(mass))
        for e, (v1, v2), nu, mass in zip(
            identity.edge_indices,
            identity.edge_endpoints,
            identity.edge_exponents,
            identity.edge_masses,
            strict=True,
        )
    ]
    return _blocks(
        _paragraph(
            f"It has L = {identity.loop_count} "
            f"{'loop' if identity.loop_count == 1 else 'loops'}, "
            f"N = {identity.propagators} "
            f"{'propagator' if identity.propagators == 1 else 'propagators'} and "
            f"{count_noun(identity.external_legs, 'external leg')}."
        ),
        _table(("e", "Vertices", "nu_e", "m_e"), rows),
        _paragraph(
            "Propagator e joins the vertices listed and carries the exponent nu_e and the "
            "mass m_e shown in the table; massless propagators have m_e = 0. Its index e also "
            "names its parameters a_e and u_e."
        ),
    )


def _kinematics(conventions: Conventions) -> str:
    invariants = conventions.invariants
    if invariants is None:
        return "the dot products p_i . p_j"
    symbols = list(dict.fromkeys(invariants.symbols))
    listed = join_words([_str(s) for s in symbols])
    noun = "invariant" if len(symbols) == 1 else "invariants"
    if invariants.n_external == 2:
        return f"the {noun} {listed}: s = p^2 for p = p_1 = -p_2, so that p_1 . p_2 = -s"
    clause = "the external masses p_i^2"
    if invariants.mandelstam:
        clause += " and the planar invariants s_{i...j-1} = (p_i + ... + p_{j-1})^2"
    return f"the {noun} {listed}: {clause}"


def _conventions(report: AnalysisReport, doc: _Document) -> str:
    conventions = report.conventions
    dimension = conventions.dimension
    is_d = isinstance(dimension, sp.Symbol) and dimension.name == "D"
    in_d = "D" if is_d else f"D = {_str(dimension)}"
    nu_total = sp.Add(*report.identity.edge_exponents)
    if report.identity.external_legs:
        momenta = (
            "External momenta are incoming, and the kinematics is written in "
            f"{_kinematics(conventions)}."
        )
    else:
        momenta = "There are no external momenta."
    return _blocks(
        "The integral is",
        f"{_INDENT}I = exp(L epsilon gamma_E) (mu^2)^(nu - L D/2)\n"
        f"{_INDENT}    * int prod_{{r=1}}^{{L}} d^D k_r/(i pi^(D/2))\n"
        f"{_INDENT}    * prod_e 1/(-q_e^2 + m_e^2)^nu_e",
        _paragraph(
            f"in {in_d} dimensions{doc.cite('weinzierl2022')}, with loop momenta k_r, the "
            "momentum q_e of propagator e fixed by momentum conservation at the vertices, the "
            "metric signature (+,-,...,-) and nu = sum_e nu_e, here "
            f"nu = {_str(nu_total)}. Dimensional regularisation sets D = D_0 - 2 epsilon "
            "for an even integer D_0 chosen when the result is expanded. The energy scale mu "
            f"makes I and the Symanzik polynomials dimensionless. {momenta} The second "
            "Symanzik polynomial is "
            "F = -sum_{T_2} s_{T_2} prod_{e not in T_2} a_e + U sum_e m_e^2 a_e, divided by "
            "mu^2, the sum running over spanning two-forests T_2 and s_{T_2} being the square "
            "of the momentum flowing from one tree of T_2 to the "
            f"other{doc.cite('weinzierl2022')}."
        ),
    )


def _polynomials(report: AnalysisReport, doc: _Document) -> str:
    polynomials = report.polynomials
    power = polynomials.f_scale_power
    scale = report.conventions.energy_scale
    g_room = _room("G = U + F")
    if power:
        f_lines = _scaled_lines(polynomials.f_numerator, scale**power, _room("F"))
        u_part, f_part, g_power = split_g(polynomials.g, scale)
        g_f_lines = _scaled_lines(f_part, scale**g_power, g_room - 2)
        g_lines = [*_lines(u_part, g_room), "+ " + g_f_lines[0], *g_f_lines[1:]]
    else:
        f_lines = _lines(polynomials.f_numerator, _room("F"))
        g_lines = _lines(polynomials.g, g_room)
    rows = [
        (str(entry.index), _str(entry.monomial), _str(entry.coefficient))
        for entry in polynomials.z_table
    ]
    rank = polynomials.monomials_g - polynomials.codimension
    return _blocks(
        _paragraph(
            "In the Schwinger parameters a_e the first Symanzik polynomial, the sum over "
            "spanning trees T of prod_{e not in T} a_e, is"
        ),
        _equation("U", _lines(polynomials.u, _room("U"))),
        "and the second is",
        _equation("F", f_lines),
        "In the Lee-Pomeransky parameters u_e their sum is",
        _equation("G = U + F", g_lines),
        _paragraph(
            f"U has degree L = {polynomials.degree_u} and F degree "
            f"L+1 = {polynomials.degree_f}; F has "
            f"{count_noun(polynomials.monomials_f, 'monomial')} and G has "
            f"{polynomials.monomials_g}. The coefficients of G are the GKZ variables z_j, here "
            "at their physical values:"
        ),
        _table(("j", "Monomial", "z_j"), rows),
        _paragraph(
            f"The configuration has codimension {polynomials.monomials_g} - rank A = "
            f"{polynomials.monomials_g} - {rank} = {polynomials.codimension}, with A the "
            "matrix whose columns are the exponent vectors of the monomials of G below a row "
            "of ones, against "
            f"{count_noun(polynomials.independent_invariants, 'independent kinematic invariant')}"
            f" in the coefficients{doc.cite('klausen2023')}."
        ),
    )


def _representation(
    prefactor: sp.Expr,
    parameters: Sequence[sp.Symbol],
    measure: sp.Expr,
    constraint: str,
    integrand: sp.Expr,
) -> str:
    """Display ``I`` as its prefactor times the integral, one factor per line."""
    width = _WIDTH - 2 * len(_INDENT)
    differentials = " ".join(f"d{_str(p)}" for p in parameters)
    head = _quotient_lines(prefactor, width)
    lines = [
        f"I = {head[0]}",
        *(f"    {line}" for line in head[1:]),
        f"    * int_{{R_+^{len(parameters)}}} {differentials}",
    ]
    if constraint:
        lines.append(f"    * {constraint}")
    for factor in ([] if measure == 1 else [measure]) + [integrand]:
        first, *rest = _lines(factor, width - 2)
        lines.append(f"    * {first}")
        lines.extend(f"      {line}" for line in rest)
    return "\n".join(_INDENT + line for line in lines)


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

    blocks = [
        _heading("Schwinger representation", "~"),
        _paragraph(
            "With the Schwinger parameters a_e integrated over [0, infinity)"
            f"{doc.cite('weinzierl2022')},"
        ),
        _representation(
            schwinger.prefactor,
            feynman.parameters,
            schwinger.measure.subs(rename),
            "",
            templates.schwinger,
        ),
        _heading("Feynman representation", "~"),
        _paragraph(
            "With the same parameters restricted to the simplex sum_e a_e = 1"
            f"{doc.cite('weinzierl2022')},"
        ),
        _representation(
            feynman.prefactor,
            feynman.parameters,
            feynman.measure,
            "delta(1 - sum_e a_e)",
            templates.feynman,
        ),
        _heading("Lee-Pomeransky representation", "~"),
        _paragraph(
            "With the Lee-Pomeransky parameters u_e integrated over [0, infinity)"
            f"{doc.cite('leepomeransky2013')},"
        ),
        _representation(
            lee_pomeransky.prefactor,
            lee_pomeransky.parameters,
            lee_pomeransky.measure,
            "",
            templates.lee_pomeransky,
        ),
    ]
    euclidean = "For Euclidean kinematics, where every coefficient of G has positive real part"
    if representations.convergence is None:
        blocks.append(
            _paragraph(
                f"Convergence. The Newton polytope P of G is not full-dimensional. {euclidean}, "
                "the Lee-Pomeransky integral therefore converges absolutely for no D and nu_e"
                f"{doc.cite('klausen2023')}."
            )
        )
        return _blocks(*blocks)
    blocks += [
        _paragraph(
            f"Convergence. {euclidean}, and for Re D > 0, the Lee-Pomeransky integral "
            "converges absolutely when the real parts of (nu_1, ..., nu_N), divided by "
            "Re(D/2), lie in the interior of the Newton polytope P of G, that is when for "
            "every facet m_j . x <= b_j of P the real part of b_j D/2 - sum_e m_{je} nu_e is "
            f"positive{doc.cite('klausen2023')}:"
        ),
        "\n".join(f"{_INDENT}Re({_str(c)}) > 0" for c in representations.convergence),
        _paragraph(
            "For such kinematics it converges absolutely for no D and nu_e when P is not "
            "full-dimensional."
        ),
    ]
    return _blocks(*blocks)


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
            f"v_{k} = ({', '.join(str(x) for x in vertex)})"
            for k, vertex in enumerate(data.vertices, start=1)
        ]
    )
    volume = data.normalized_volume
    description = _paragraph(
        "The Newton polytope P of G is the convex hull of the exponent vectors of its "
        f"monomials. It has dimension {data.dimension} in R^{data.ambient_dimension} and "
        f"normalised volume {volume}{shape}. Its vertices are {vertices}."
    )
    if not data.is_full_dimensional:
        return _blocks(
            description,
            _paragraph(
                "Since P is not full-dimensional, the rows of A are linearly dependent. For "
                "generic beta the Euler equations are then inconsistent, and the GKZ system has "
                "no non-zero solutions: its holonomic rank is 0, not the normalised volume."
            ),
        )
    return _blocks(
        description,
        _paragraph(
            "For generic coefficients and non-resonant beta the holonomic rank of the GKZ "
            "system equals the normalised volume, here "
            f"{volume}{doc.cite('adolphson1994', 'chestnov2022')}. The rank equals the volume "
            "for every beta exactly when the toric ring C[NA] is "
            f"Cohen-Macaulay{doc.cite('mmw2005')}. This holds when the graph is one-particle "
            "irreducible and one-vertex irreducible, its external momenta are generic enough "
            "that no monomial of G cancels, and its propagators are all massive, all massless, "
            "or such that every vertex reaches an external leg along massive propagators "
            "alone; the configuration is then normal and so Cohen-Macaulay"
            f"{doc.cite('klausen2023', 'tellander2023', 'walther2022')}. The number of master "
            "integrals is, up to sign, the Euler characteristic of the complement of "
            "{G = 0} in the torus, at most N! times the Euclidean volume of P, with equality "
            "for generic coefficients; graph-polynomial coefficients are rarely generic"
            f"{doc.cite('bbkp2017')}. The bound equals the normalised volume when the exponent "
            f"differences span Z^N. {NOT_COMPUTED}"
        ),
    )


def _gkz(gkz: GKZ, doc: _Document) -> str:
    operators = []
    for r, (row, beta_r) in enumerate(gkz.euler_rows):
        terms = signed_terms([(a, f"theta_{j}") for j, a in enumerate(row, start=1)])
        operator = append_signed(terms, -beta_r, _str)
        operators.append(_equation(f"E_{r}", _break(operator, _room(f"E_{r}"))))
    blocks = [
        _paragraph(
            "Writing G = sum_j z_j u^alpha_j and treating the coefficients z_j as independent "
            "variables gives the Euler-Mellin integral"
        ),
        f"{_INDENT}{_EULER_MELLIN},",
        _paragraph(
            "the Lee-Pomeransky integral without its prefactor, which the GKZ system "
            "annihilates. The system has the matrix"
        ),
        _matrix("A", gkz.a_matrix),
        _paragraph(
            "whose column j is (1, alpha_j). The parameter vector is "
            "beta = (-D/2, -nu_1, ..., -nu_N), the value the Euler equations require for the "
            f"Lee-Pomeransky integral{doc.cite('delacruz2019', 'klausen2020')}. Here"
        ),
        _vector("beta", gkz.beta, "."),
        _paragraph(
            "The Euler operators are built from theta_j = z_j d/dz_j, which measures the "
            "degree in z_j. The operators E_r = sum_j A_{rj} theta_j - beta_r annihilate I_A: "
            "row 0 states that I_A is homogeneous of degree -D/2 in the z_j, and row i that "
            "it has degree -nu_i under z_j -> lambda^(A_{ij}) z_j, which a rescaling of u_i "
            "absorbs:"
        ),
        "\n".join(operators),
    ]
    generators = gkz.toric_generators
    if not generators:
        blocks.append("The toric ideal is trivial.")
        return _blocks(*blocks)
    blocks += [
        _paragraph(
            f"The toric ideal of A is generated by {count_noun(len(generators), 'binomial')}. "
            "Each binomial z^k - z^l, with A k = A l, gives the operator "
            "(d/dz)^k - (d/dz)^l; these annihilators of I_A form an analogue of "
            f"integration-by-parts relations for it{doc.cite('chestnov2022')}, and the "
            "specialisation to physical coefficients is a separate step. The generators are"
        ),
        _expressions(generators),
    ]
    return _blocks(*blocks)


def _symmetries(report: AnalysisReport, symmetries: Symmetries | None, doc: _Document) -> str:
    if symmetries is None:
        return _paragraph(symmetries_omitted(report))
    orbits = symmetries.vertex_orbits
    if report.polytope is not None:
        listed = join_words(
            ["{" + ", ".join(f"v_{i + 1}" for i in orbit) + "}" for orbit in orbits]
        )
        action = (
            "acts on the vertices listed in the Newton polytope section with "
            f"{'the orbit' if len(orbits) == 1 else 'orbits'} {listed}"
        )
    else:
        sizes = join_words([str(len(orbit)) for orbit in orbits])
        action = (
            f"acts on the vertices of P with {count_noun(len(orbits), 'orbit')} of "
            f"{'size' if len(orbits) == 1 else 'sizes'} {sizes}"
        )
    if report.gkz is not None:
        definition = ""
        integral = " for the Euler-Mellin integral I_A of the GKZ system section"
    else:
        definition = (
            f"Write G = sum_j z_j u^alpha_j and let {_EULER_MELLIN} be the Lee-Pomeransky "
            "integral without its prefactor, where beta = (-D/2, -nu_1, ..., -nu_N). "
        )
        integral = ""
    preserving = symmetries.coefficient_preserving
    pairs = symmetries.symmetry_pairs
    blocks = [
        _paragraph(
            "The group Aut(P) of unimodular affine maps taking P to itself has order "
            f"{symmetries.automorphism_order} and {action}. The graph has "
            f"{count_noun(len(symmetries.graph_automorphisms), 'automorphism')}. Of the "
            f"polytope automorphisms, {preserving} "
            f"{'preserves' if preserving == 1 else 'preserve'} the coefficients of G."
        )
    ]
    if not pairs:
        blocks.append("The configuration has no symmetry pairs.")
        return _blocks(*blocks)
    shown = pairs[:MAX_PAIRS_SHOWN]
    n_columns = report.polynomials.monomials_g
    blocks.append(
        _paragraph(
            f"{definition}The configuration has {count_noun(len(pairs), 'symmetry pair')} "
            "(T, sigma): integer matrices T and permutations sigma of the columns of A such "
            "that T takes column j of A to column sigma(j). Each gives the identity "
            f"I_A(beta, z_sigma) = I_A(T beta, z){integral}, with "
            f"z_sigma = (z_{{sigma(1)}}, ..., z_{{sigma({n_columns})}}). Forsgaard, Matusevich "
            f"and Sobieska{doc.cite('fms2019')} and de la Cruz{doc.cite('delacruz2024')} print "
            "the permutation on the other side, but the substitution in the proof of "
            f"Corollary 4.1 of{doc.cite('fms2019')} gives the form stated here. For I itself "
            "the prefactors at beta and T beta enter as well. "
            + ("They are" if len(pairs) == len(shown) else f"The first {len(shown)} are")
            + ", with sigma listed as (sigma(1), sigma(2), ...):"
        )
    )
    for k, pair in enumerate(shown, start=1):
        sigma = ", ".join(str(j + 1) for j in pair.column_permutation)
        display = _matrix(f"T_{k}", pair.homogenized_map, f", sigma_{k} = ({sigma})")
        if max(len(line) for line in display.splitlines()) > _WIDTH:
            # sigma_k moves to a line of its own, broken after commas to fit.
            images = [sp.Integer(j + 1) for j in pair.column_permutation]
            display = "\n".join(
                [
                    _matrix(f"T_{k}", pair.homogenized_map, ","),
                    _vector(f"sigma_{k}", images, ""),
                ]
            )
        blocks.append(display)
    if len(pairs) > len(shown):
        rest = len(pairs) - len(shown)
        blocks.append(f"and {rest} further {'pair' if rest == 1 else 'pairs'}.")
    return _blocks(*blocks)


def _landau(landau: Landau, scale: sp.Symbol, doc: _Document) -> str:
    intro = (
        "The reduced principal A-determinant of G, each irreducible kinematic factor taken "
        "once, is the product of the discriminants of G restricted to the faces of P"
        f"{doc.cite('gkz1994', 'dhpt2023', 'fmt2023')}."
    )
    factors = landau_factors(landau, scale)
    blocks = []
    if factors.by_dimension:
        blocks += [
            _paragraph(
                intro + " Each discriminant that is not identically one factorises as follows, "
                "by face dimension, with one factor per line; numerical factors and powers of "
                "the energy scale mu, which only normalise the coefficients z_j, are left out:"
            ),
            "\n".join(
                f"- Dimension {dimension}:\n{_expressions(listed)}"
                for dimension, listed in factors.by_dimension
            ),
        ]
    else:
        blocks.append(_paragraph(intro + " No face discriminant has a kinematic factor."))
    if factors.closed_form:
        first, first_display = _factor_list("first-type (Cayley)", factors.first_type)
        second, second_display = _factor_list("second-type (Gram)", factors.second_type)
        sentence = (
            "By comparison with the closed form from the modified Cayley matrix"
            f"{doc.cite('dhpt2023')}, {first}"
        )
        if first_display is not None:
            blocks += [_paragraph(sentence), first_display]
            sentence = f"and {second}"
        else:
            sentence += f" and {second}"
        blocks.append(_paragraph(sentence if second_display is not None else sentence + "."))
        if second_display is not None:
            blocks.append(second_display)
        shared = factors.in_both
        if shared:
            blocks.append(
                _paragraph(
                    "A factor can arise from both a Cayley minor and a Gram minor, which is why "
                    f"{count_noun(shared, 'factor')} {'appears' if shared == 1 else 'appear'} "
                    "in both lists."
                )
            )
    blocks.append(
        _paragraph(
            "The factors are candidate codimension-one singular loci on all sheets of the "
            "integral: a point on one of them may or may not be singular on the physical "
            f"sheet, and the list is not guaranteed complete{doc.cite('fmt2023')}. The "
            "coefficients are specialised to physical kinematics before each face "
            "discriminant is computed, so beyond one loop this is the principal Landau "
            "determinant rather than the principal A-determinant of the generic polynomial; "
            "multiplicities are dropped."
        )
    )
    skipped = skipped_faces(landau)
    if skipped is not None:
        blocks.append(_paragraph(skipped))
    return _blocks(*blocks)


def _schwinger(report: AnalysisReport, schwinger: Schwinger, doc: _Document) -> str:
    system = schwinger.system
    f_block = schwinger.f_block
    loops = report.identity.loop_count
    nu_total = sp.Add(*report.identity.edge_exponents)
    value = (loops + 1) * report.conventions.dimension / 2
    check = "verified" if schwinger.columns_match else "not verified"
    return _blocks(
        _paragraph(
            "The Schwinger representation gives a second GKZ system on the Cayley "
            "configuration of (U~, F~), the Symanzik polynomials with a_N = 1 and a_e = u_e "
            f"otherwise{doc.cite('jimenez2026', 'klausen2023')}:"
        ),
        _matrix("A_Cayley", system.a_matrix),
        _vector("beta_Cayley", system.beta_parameters, "."),
        _paragraph(
            "It is unimodularly equivalent to the Lee-Pomeransky configuration, the matrix "
            "A_LP and parameter vector beta_LP = (-D/2, -nu_1, ..., -nu_N) of the GKZ system "
            "of G, through"
        ),
        _matrix("T", schwinger.lp_to_cayley),
        _paragraph(
            "which maps the parameter vectors as beta_Cayley = T beta_LP and the columns of "
            "A_LP to those of A_Cayley up to order; the column correspondence is "
            f"{check} for this integral. Restricting to the F~ block gives the system"
        ),
        _matrix("A_F", f_block.a_matrix),
        _vector("beta_F", f_block.beta_parameters, ";"),
        _paragraph(
            "its solutions solve the full system when beta_Cayley lies in the span of the "
            "face's columns, which for the F~ block means nu = (L+1)D/2, here "
            f"{_str(nu_total)} = {_str(value)}{doc.cite('britto2026')}. Away from that "
            "value the relation between the two systems is not established."
        ),
    )


# --- the document ------------------------------------------------------------


def render_text(report: AnalysisReport, *, title: str | None = None) -> str:
    """Render an analysis report as plain text.

    Parameters
    ----------
    report
        The report to render; sections it does not carry are omitted.
    title
        Title line, used as given; by default "Feynman integral" followed by
        the CNickel string.

    Returns
    -------
    str
        The report, ASCII only: the title underlined with ``=``, each section
        heading with ``-``, and the references last.
    """
    doc = _Document()
    heading = f"Feynman integral {report.identity.cnickel}" if title is None else title
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
    blocks = [
        _heading(heading, "="),
        *(f"{_heading(name, '-')}\n\n{body}" for name, body in sections),
        f"{_heading('References', '-')}\n\n{doc.references()}",
    ]
    return "\n\n".join(blocks) + "\n"
