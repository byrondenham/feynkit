"""
Kinematic classes of a Feynman integral.

The class is read from two axes. The internal axis describes the masses m_e
of the propagators, the external axis the momentum products p_i . p_j of the
legs. Their values follow the principal Landau determinant database of
Fevola, Mizera and Telen (arXiv:2311.16219): internal zero, equal and generic
are its internal names, and external on_shell, equal and off_shell its
external zero, equal and generic. Four pairs of axes are named classes; every
other pair, among them the massive box with on-shell legs, is "other".

The axes are derived from the integral as it stands and never raise, so a
substitution made with ``with_`` changes the class with it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from math import lcm
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import sympy as sp

from .. import _exact
from ..core.constants import MASS_ASSUMPTIONS
from ..core.exceptions import ValidationError
from ..core.graph import Graph

if TYPE_CHECKING:
    from ..integral import FeynmanIntegral

__all__ = [
    "CLASS_OF_AXES",
    "IMPOSABLE_CLASSES",
    "KINEMATIC_CLASSES",
    "ExternalAxis",
    "InternalAxis",
    "KinematicClass",
    "impose_kinematics",
    "kinematic_axes",
    "kinematic_class",
]

InternalAxis = Literal["zero", "equal", "generic", "other"]
ExternalAxis = Literal["off_shell", "on_shell", "equal", "other"]
KinematicClass = Literal[
    "generic", "massless_off_shell", "massless_on_shell", "equal_masses", "other"
]

KINEMATIC_CLASSES: tuple[KinematicClass, ...] = (
    "generic",
    "massless_off_shell",
    "massless_on_shell",
    "equal_masses",
    "other",
)

# The classes with_kinematics can impose: all but "other".
IMPOSABLE_CLASSES: tuple[KinematicClass, ...] = KINEMATIC_CLASSES[:4]

# The named cells of the table; every other pair of axes is "other".
CLASS_OF_AXES: Mapping[tuple[InternalAxis, ExternalAxis], KinematicClass] = MappingProxyType(
    {
        ("zero", "off_shell"): "massless_off_shell",
        ("zero", "on_shell"): "massless_on_shell",
        ("equal", "off_shell"): "equal_masses",
        ("generic", "off_shell"): "generic",
    }
)


def _expression(value: object) -> sp.Expr | None:
    """value as a SymPy expression, or None when it is not one."""
    try:
        expression = sp.sympify(value)
    except (sp.SympifyError, TypeError, ValueError):
        return None
    return expression if isinstance(expression, sp.Expr) else None


def _masses(integral: FeynmanIntegral) -> list[sp.Expr | None]:
    """The mass of each internal edge, expanded as in G, in internal-edge order."""
    masses = [_expression(e.get_mass()) for e in integral.graph.get_internal_edges()]
    return [None if m is None else sp.expand(m) for m in masses]


def _internal_axis(integral: FeynmanIntegral) -> InternalAxis:
    masses = _masses(integral)
    # Exact zero only: Float(0.0) is a number, and so "other".
    if all(m is not None and m == 0 for m in masses):
        return "zero"
    if len(masses) >= 2 and isinstance(masses[0], sp.Symbol) and len(set(masses)) == 1:
        return "equal"
    nonzero = [m for m in masses if not (m is not None and m == 0)]
    if all(isinstance(m, sp.Symbol) for m in nonzero) and len(set(nonzero)) == len(nonzero):
        return "generic"
    return "other"


def _products(integral: FeynmanIntegral) -> list[sp.Expr | None]:
    """P_ij for i < j in lexicographic order, read under (i, j) or (j, i), 0 when missing."""
    given = integral.momentum_products
    n = integral.graph.external_legs
    values: list[sp.Expr | None] = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            value = given.get((i, j))
            if value is None:
                value = given.get((j, i))
            expression = _expression(0 if value is None else value)
            values.append(None if expression is None else sp.expand(expression))
    return values


def _squares(products: list[sp.Expr], n: int) -> list[sp.Expr]:
    """p_i^2 = -sum_{j != i} p_i . p_j for each leg, by momentum conservation."""
    pairs = [(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    return [
        sp.expand(-sp.Add(*(v for (a, b), v in zip(pairs, products, strict=True) if i in (a, b))))
        for i in range(1, n + 1)
    ]


def _linear_row(value: sp.Expr, symbols: list[sp.Symbol]) -> list[int] | None:
    """The coefficients of value in symbols, scaled to integers, when value is affine in them
    with rational coefficients; otherwise None."""
    present = sorted(value.free_symbols, key=sp.default_sort_key)
    if not present:
        return [0] * len(symbols) if value.is_Rational else None
    try:
        if not value.is_polynomial(*present):
            return None
        poly = sp.Poly(value, *present)
    except (sp.PolynomialError, sp.GeneratorsError, TypeError, ValueError):
        return None
    if poly.total_degree() > 1 or not all(c.is_Rational for c in poly.coeffs()):
        return None
    coefficients = [
        sp.Rational(poly.coeff_monomial(x)) if x in present else sp.Integer(0) for x in symbols
    ]
    scale = lcm(*(int(c.q) for c in coefficients))
    return [int(c * scale) for c in coefficients]


def _external_axis(integral: FeynmanIntegral) -> ExternalAxis:
    if integral.kinematic_constraints:
        return "other"
    n = integral.graph.external_legs
    if n < 2:
        return "off_shell"
    parsed = _products(integral)
    if any(v is None for v in parsed):
        return "other"
    products = [v for v in parsed if v is not None]
    masses = _masses(integral)
    banned = {integral.graph.energy_scale}.union(*(m.free_symbols for m in masses if m is not None))
    symbols = sorted(set().union(*(v.free_symbols for v in products)), key=sp.default_sort_key)
    if banned & set(symbols):
        return "other"
    rows = [_linear_row(v, symbols) for v in products]
    if any(row is None for row in rows):
        return "other"
    k = len(products)
    r = _exact.rank([row for row in rows if row is not None]) if symbols else 0
    if r == k:
        return "off_shell"
    squares = _squares(products, n)
    if all(q == 0 for q in squares) and r == max(k - n, 0):
        return "on_shell"
    if n >= 3 and squares[0] != 0 and all(q == squares[0] for q in squares) and r == k - n + 1:
        return "equal"
    return "other"


def kinematic_axes(integral: FeynmanIntegral) -> tuple[InternalAxis, ExternalAxis]:
    """
    The internal and external kinematic axes of an integral.

    The internal axis reads the masses m_e = ``get_mass()`` of the N internal
    edges; the first rule that applies decides. ``zero``: every m_e is exactly
    0. ``equal``: N >= 2 and every m_e is the same symbol. ``generic``: the
    nonzero m_e are distinct symbols, massless lines allowed. ``other``: a
    number or an expression, Float(0.0) included, or a symbol shared by some
    lines but not all.

    The external axis reads the n legs and the products P_ij, i < j, each
    under (i, j) or (j, i), a missing product counting as 0. With
    k = n(n-1)/2, p_i^2 = -sum_{j != i} P_ij and r the rank of the Jacobian
    of the P_ij in their free symbols: ``other`` with kinematic constraints;
    ``off_shell`` for n < 2; ``other`` unless every P_ij is of degree at most
    1 with rational coefficients and contains neither a mass symbol nor the
    energy scale; ``off_shell`` if r = k; ``on_shell`` if every p_i^2 = 0 and
    r = max(k - n, 0); ``equal`` if n >= 3, every p_i^2 is the same nonzero
    expression and r = k - n + 1; ``other`` otherwise.

    It never raises.
    """
    return _internal_axis(integral), _external_axis(integral)


def kinematic_class(integral: FeynmanIntegral) -> KinematicClass:
    """The class of the axes of :func:`kinematic_axes`: a named cell of
    :data:`CLASS_OF_AXES`, or ``"other"``."""
    return CLASS_OF_AXES.get(kinematic_axes(integral), "other")


def _edge_list(indices: list[int]) -> str:
    return ", ".join(str(i) for i in indices)


def _on_shell(integral: FeynmanIntegral) -> FeynmanIntegral:
    """Every p_i^2 set to 0: each must be 0 or a symbol, which is set to 0 in every product."""
    n = integral.graph.external_legs
    if n < 2:
        raise ValidationError(
            "cannot impose massless_on_shell: the integral has fewer than two external legs, "
            "so there is nothing to put on shell"
        )
    massive = [
        e.idx
        for e, m in zip(integral.graph.get_internal_edges(), _masses(integral), strict=True)
        if m is None or m.is_zero is not True
    ]
    if massive:
        noun = "propagator" if len(massive) == 1 else "propagators"
        verb = "is" if len(massive) == 1 else "are"
        raise ValidationError(
            f"cannot impose massless_on_shell: {noun} {_edge_list(massive)} {verb} massive, and "
            "a class does not change which propagators are massless"
        )
    parsed = _products(integral)
    if any(v is None for v in parsed):
        raise ValidationError(
            "cannot impose massless_on_shell: a momentum product is not a SymPy expression"
        )
    squares = _squares([v for v in parsed if v is not None], n)
    for i, square in enumerate(squares, start=1):
        if square != 0 and not isinstance(square, sp.Symbol):
            raise ValidationError(
                f"cannot impose massless_on_shell: p_{i}^2 = {square} is not a symbol to set to "
                "0; build the integral with use_mandelstam=True, whose p_i^2 are symbols"
            )
    zeros = {square: sp.Integer(0) for square in squares if square != 0}
    return integral.with_(
        momentum_products={
            key: sp.expand(sp.sympify(value).subs(zeros))
            for key, value in integral.momentum_products.items()
        }
    )


def _equal_masses(integral: FeynmanIntegral) -> FeynmanIntegral:
    """Every internal edge given the mass m_a of the CNickel code a."""
    graph = integral.graph
    if len(graph.get_internal_edges()) < 2:
        raise ValidationError(
            "cannot impose equal_masses: a single propagator cannot have equal masses"
        )
    massless = [
        e.idx
        for e, m in zip(graph.get_internal_edges(), _masses(integral), strict=True)
        if m is not None and m.is_zero is True
    ]
    if massless:
        noun = "propagator" if len(massless) == 1 else "propagators"
        verb = "is" if len(massless) == 1 else "are"
        raise ValidationError(
            f"cannot impose equal_masses: {noun} {_edge_list(massless)} {verb} massless, and a "
            "class does not change which propagators are massless"
        )
    m_a = sp.Symbol("m_a", **MASS_ASSUMPTIONS)
    edges = [dataclasses.replace(e, mass=m_a) if e.is_internal else e for e in graph.edges]
    return integral.with_(
        graph=Graph(
            graph.internal_vertices, graph.external_legs, edges, energy_scale=graph.energy_scale
        )
    )


def impose_kinematics(integral: FeynmanIntegral, kinematic_class: str) -> FeynmanIntegral:
    """
    The integral with a kinematic class imposed by substitution.

    Returns ``integral`` itself when it already has the class. Otherwise:
    ``massless_on_shell`` needs every propagator massless and at least two
    legs, and sets each p_i^2, which must be 0 or a symbol as with
    ``use_mandelstam=True``, to 0 in every momentum product; ``equal_masses``
    needs every propagator massive and gives each the mass m_a of the CNickel
    code a; ``generic`` and ``massless_off_shell`` make no substitution. A
    class never changes which propagators are massless.

    Raises
    ------
    ValidationError
        If ``kinematic_class`` is not one of :data:`IMPOSABLE_CLASSES`, if the
        substitution cannot be made, or if the result is not of the class.
    """
    if kinematic_class not in IMPOSABLE_CLASSES:
        raise ValidationError(
            f"cannot impose the kinematic class {kinematic_class!r}; choose from "
            f"{', '.join(IMPOSABLE_CLASSES)}"
        )
    internal, external = kinematic_axes(integral)
    found = CLASS_OF_AXES.get((internal, external), "other")
    if found == kinematic_class:
        return integral
    if kinematic_class in ("generic", "massless_off_shell"):
        raise ValidationError(
            f"cannot impose {kinematic_class}: it makes no substitution, and the integral has "
            f"internal axis {internal} and external axis {external}, of class {found}"
        )
    result = (
        _on_shell(integral) if kinematic_class == "massless_on_shell" else _equal_masses(integral)
    )
    internal, external = kinematic_axes(result)
    found = CLASS_OF_AXES.get((internal, external), "other")
    if found != kinematic_class:
        raise ValidationError(
            f"cannot impose {kinematic_class}: the result has internal axis {internal} and "
            f"external axis {external}, of class {found}"
        )
    return result
