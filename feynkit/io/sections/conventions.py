"""
The conventions: dimension, energy scale, invariants and kinematic class.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from ...kinematics.mandelstam import KinematicInvariants, standard_invariants
from .._latex_kit import LatexDocument, _escape, _math
from .._report_shared import join_words
from .._text_kit import _INDENT, TextDocument, _blocks, _paragraph, _str
from ..latex import to_latex
from ._base import Section, SummaryPart

if TYPE_CHECKING:
    from ..report import AnalysisReport, BuildContext


@dataclass(frozen=True)
class Conventions:
    """The dimension, the energy scale and the kinematics.

    Attributes
    ----------
    dimension
        The spacetime dimension D.
    energy_scale
        The scale mu that keeps the Symanzik polynomials dimensionless.
    invariants
        The standard planar invariants, or None when the integral has fewer
        than two external legs or does not use Mandelstam variables.
    momentum_products
        The products p_i . p_j as ((i, j), value) pairs, sorted by key.
    kinematic_axes, kinematic_class
        The internal and external kinematic axes and the kinematic class, as
        FeynmanIntegral.kinematic_axes and kinematic_class give them.
    """

    dimension: sp.Expr
    energy_scale: sp.Symbol
    invariants: KinematicInvariants | None
    momentum_products: tuple[tuple[tuple[int, int], sp.Expr], ...]
    kinematic_axes: tuple[str, str]
    kinematic_class: str


def build(ctx: BuildContext) -> Conventions:
    fi = ctx.integral
    n_legs = fi.graph.external_legs
    invariants = standard_invariants(n_legs) if fi.use_mandelstam and n_legs >= 2 else None
    return Conventions(
        dimension=fi.dimension,
        energy_scale=fi.graph.energy_scale,
        invariants=invariants,
        momentum_products=tuple(sorted(fi.momentum_products.items())),
        kinematic_axes=fi.kinematic_axes,
        kinematic_class=fi.kinematic_class,
    )


# What each kinematic axis says, for a class that is "other". Maths is written as LaTeX and
# passed through the renderer's ``math``.
_INTERNAL_AXES = {
    "zero": "every propagator is massless",
    "equal": "every propagator has the same mass",
    "generic": "the masses are generic",
    "other": "the masses are neither all zero, all equal nor distinct symbols",
}


_EXTERNAL_AXES = {
    "off_shell": "the legs are off shell",
    "on_shell": "every leg is on shell",
    "equal": "every leg has the same {p2}",
    "other": "the invariants are neither free, on shell nor of equal mass",
}


def kinematic_class_sentence(
    conventions: Conventions,
    external_legs: int,
    masses: Sequence[sp.Expr],
    *,
    name: Callable[[str], str],
    math: Callable[[str], str],
    printer: Callable[[sp.Expr], str],
) -> str:
    """The sentence of the conventions naming the kinematic class and what it imposes.

    ``name`` sets the class as code, ``math`` a LaTeX fragment as the
    renderer writes maths and ``printer`` an expression. With fewer than two
    legs there are no invariants, and the sentence says nothing of the legs.
    """
    kinematic_class = conventions.kinematic_class
    internal, external = conventions.kinematic_axes
    legs = external_legs >= 2
    free = ", and no relation among the invariants is imposed" if legs else ""
    if kinematic_class == "generic":
        what = f"the nonzero masses {math('m_e')} are distinct symbols{free}"
    elif kinematic_class == "massless_off_shell":
        what = f"every propagator is massless{free}"
    elif kinematic_class == "equal_masses":
        what = f"every propagator has the mass {printer(masses[0])}{free}"
    elif kinematic_class == "massless_on_shell":
        left = sorted(
            {x for _, value in conventions.momentum_products for x in value.free_symbols},
            key=sp.default_sort_key,
        )
        dot, zero = math("p_i \\cdot p_j"), math("p_i^2 = 0")
        rest = (
            f"leaving {join_words([printer(x) for x in left])}"
            if left
            else f"so every product {dot} vanishes"
        )
        what = f"every propagator is massless and every leg on shell, {zero}, {rest}"
    else:
        what = _INTERNAL_AXES.get(internal, _INTERNAL_AXES["other"])
        if legs or external != "off_shell":
            phrase = _EXTERNAL_AXES.get(external, _EXTERNAL_AXES["other"])
            what += " and " + phrase.format(p2=math("p_i^2"))
    return f"The kinematic class is {name(kinematic_class)}: {what}."


def _text_kinematics(conventions: Conventions) -> str:
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


def text(report: AnalysisReport, conventions: Conventions, doc: TextDocument) -> str:
    dimension = conventions.dimension
    is_d = isinstance(dimension, sp.Symbol) and dimension.name == "D"
    in_d = "D" if is_d else f"D = {_str(dimension)}"
    nu_total = sp.Add(*report.identity.edge_exponents)
    if report.identity.external_legs:
        momenta = (
            "External momenta are incoming, and the kinematics is written in "
            f"{_text_kinematics(conventions)}."
        )
    else:
        momenta = "There are no external momenta."
    kinematic_class = kinematic_class_sentence(
        conventions,
        report.identity.external_legs,
        report.identity.edge_masses,
        name=str,
        math=lambda text: text.replace(" \\cdot ", " . "),
        printer=_str,
    )
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
            f"makes I and the Symanzik polynomials dimensionless. {momenta} {kinematic_class} "
            "The second Symanzik polynomial is "
            "F = -sum_{T_2} s_{T_2} prod_{e not in T_2} a_e + U sum_e m_e^2 a_e, divided by "
            "mu^2, the sum running over spanning two-forests T_2 and s_{T_2} being the square "
            "of the momentum flowing from one tree of T_2 to the "
            f"other{doc.cite('weinzierl2022')}."
        ),
    )


def _latex_kinematics(conventions: Conventions) -> str:
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


def latex(report: AnalysisReport, conventions: Conventions, doc: LatexDocument) -> str:
    dimension = conventions.dimension
    is_d = isinstance(dimension, sp.Symbol) and dimension.name == "D"
    in_d = "$D$" if is_d else f"$D = {to_latex(dimension)}$"
    nu_total = sp.Add(*report.identity.edge_exponents)
    if report.identity.external_legs:
        momenta = (
            "External momenta are incoming, and the kinematics is written in "
            f"{_latex_kinematics(conventions)}."
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


def summary(conventions: Conventions) -> list[tuple[str, str]]:
    return [("Kinematic class", conventions.kinematic_class)]


SECTION = Section(
    name="conventions",
    heading="Conventions",
    always=True,
    build=build,
    text=text,
    latex=latex,
    summary=(SummaryPart(20, summary),),
)
