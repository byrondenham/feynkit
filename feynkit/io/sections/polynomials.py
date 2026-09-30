"""
The Symanzik polynomials U, F and G and the counts derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from ...systems.monomial import extract_monomial_support
from .._latex_kit import LatexDocument, _latex_equation, _math
from .._report_shared import count_noun, split_g
from .._text_kit import (
    TextDocument,
    _blocks,
    _lines,
    _paragraph,
    _room,
    _str,
    _table,
    _text_equation,
)
from ..latex import factor_energy_scale, to_latex, to_latex_lines
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class ZEntry:
    """One monomial of G and the GKZ variable standing for its coefficient.

    Attributes
    ----------
    index
        The index j of z_j, counting from one, in the column order of the
        GKZ A-matrix.
    exponent
        The exponent vector in the Lee-Pomeransky parameters.
    monomial
        The monomial itself, the product of u_e**a_e.
    coefficient
        The physical value of z_j, the coefficient the monomial carries in
        G.
    """

    index: int
    exponent: tuple[int, ...]
    monomial: sp.Expr
    coefficient: sp.Expr


@dataclass(frozen=True)
class Polynomials:
    """The Symanzik polynomials and the counts derived from them.

    Attributes
    ----------
    u, g
        U in the Schwinger parameters and G = U + F in the Lee-Pomeransky
        parameters.
    f_numerator, f_scale_power
        F written as ``f_numerator / mu**f_scale_power`` with the numerator
        free of mu; the power is 2 for an integral with kinematics and 0
        when mu does not occur.
    degree_u, degree_f
        The total degrees L and L + 1.
    monomials_f, monomials_g
        The number of monomials of F and of G.
    z_table
        One entry per monomial of G, in GKZ column order.
    codimension
        ``monomials_g - rank A``, the codimension of the A-configuration
        (Klausen 2023, Table A.4); it is ``monomials_g - N - 1`` for N internal
        edges when the Newton polytope is full-dimensional.
    independent_invariants
        The number of distinct kinematic symbols, masses and invariants, in
        the coefficients of G; mu is not counted.
    scaleless
        Whether the integral is scaleless by Lee's criterion, as
        FeynmanIntegral.is_scaleless says.
    """

    u: sp.Expr
    f_numerator: sp.Expr
    f_scale_power: int
    g: sp.Expr
    degree_u: int
    degree_f: int
    monomials_f: int
    monomials_g: int
    z_table: tuple[ZEntry, ...]
    codimension: int
    independent_invariants: int
    scaleless: bool


def _total_degree(expr: sp.Expr, parameters: list[sp.Symbol]) -> int:
    """The total degree of expr in the parameters, 0 for the zero polynomial.

    SymPy gives the zero polynomial in one variable degree 0 with its own
    ground types but -inf with python-flint's, which it uses whenever
    python-flint is installed.
    """
    if expr == 0:
        return 0
    return int(sp.Poly(expr, *parameters).total_degree())


def build(ctx: BuildContext) -> Polynomials:
    fi = ctx.integral
    symanzik = fi.symanzik
    parameters = list(symanzik.lp_parameters)
    scale = fi.graph.energy_scale
    numerator, power = factor_energy_scale(symanzik.f, scale)

    physical = {tuple(alpha): coefficient for alpha, coefficient in fi.newton_polytope.support}
    z_table = tuple(
        ZEntry(
            index=j,
            exponent=tuple(int(k) for k in alpha),
            monomial=sp.Mul(*[v ** int(k) for v, k in zip(parameters, alpha, strict=True)]),
            coefficient=physical.get(tuple(alpha), sp.Integer(0)),
        )
        for j, (alpha, _) in enumerate(fi.gkz.support, start=1)
    )
    invariants = {s for entry in z_table for s in entry.coefficient.free_symbols} - {scale}

    f_expanded = sp.expand(symanzik.f)
    monomials_f = (
        0
        if f_expanded == 0
        else len(extract_monomial_support(f_expanded, list(symanzik.schwinger_parameters)))
    )
    return Polynomials(
        u=symanzik.u,
        f_numerator=numerator,
        f_scale_power=power,
        g=symanzik.g,
        degree_u=_total_degree(symanzik.u_lp, parameters),
        degree_f=_total_degree(symanzik.f_lp, parameters),
        monomials_f=monomials_f,
        monomials_g=len(z_table),
        z_table=z_table,
        codimension=len(z_table) - int(fi.gkz.a_matrix.rank()),
        independent_invariants=len(invariants),
        scaleless=fi.is_scaleless,
    )


def _text_scaled_lines(expr: sp.Expr, scale: sp.Expr, width: int) -> list[str]:
    """The lines of ``expr`` written as ``(1/scale)*(...)``."""
    prefix = f"(1/{_str(scale)})*("
    lines = _lines(expr, width - len(prefix) - 1)
    lines[0] = prefix + lines[0]
    lines[-1] += ")"
    return lines


def text(report: AnalysisReport, polynomials: Polynomials, doc: TextDocument) -> str:
    power = polynomials.f_scale_power
    scale = report.conventions.energy_scale
    g_room = _room("G = U + F")
    if power:
        f_lines = _text_scaled_lines(polynomials.f_numerator, scale**power, _room("F"))
        u_part, f_part, g_power = split_g(polynomials.g, scale)
        g_f_lines = _text_scaled_lines(f_part, scale**g_power, g_room - 2)
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
        _text_equation("U", _lines(polynomials.u, _room("U"))),
        "and the second is",
        _text_equation("F", f_lines),
        "In the Lee-Pomeransky parameters u_e their sum is",
        _text_equation("G = U + F", g_lines),
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


def _latex_scaled_lines(expr: sp.Expr, scale: sp.Expr) -> list[str]:
    """The lines of ``expr`` written as ``(1/scale)(...)``."""
    lines = to_latex_lines(expr)
    lines[0] = f"\\frac{{1}}{{{to_latex(scale)}}}\\bigl({lines[0]}"
    lines[-1] += "\\bigr)"
    return lines


def latex(report: AnalysisReport, polynomials: Polynomials, doc: LatexDocument) -> str:
    power = polynomials.f_scale_power
    scale = report.conventions.energy_scale
    if power:
        f_lines = _latex_scaled_lines(polynomials.f_numerator, scale**power)
        u_part, f_part, g_power = split_g(polynomials.g, scale)
        g_f_lines = _latex_scaled_lines(f_part, scale**g_power)
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
            _latex_equation("U", to_latex_lines(polynomials.u)),
            "and the second is",
            _latex_equation("F", f_lines),
            "In the Lee-Pomeransky parameters $u_e$ their sum is",
            _latex_equation("G = U + F", g_lines),
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


def summary(polynomials: Polynomials) -> list[tuple[str, str]]:
    return [
        ("Monomials of F", str(polynomials.monomials_f)),
        ("Monomials of G", str(polynomials.monomials_g)),
        ("Independent invariants", str(polynomials.independent_invariants)),
        ("Codimension", str(polynomials.codimension)),
        ("Scaleless", "yes" if polynomials.scaleless else "no"),
    ]


SECTION = Section(
    name="polynomials",
    heading="Symanzik polynomials",
    always=True,
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(30, summary),),
)
