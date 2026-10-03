"""
Graphs of the faces of the Newton polytope of G = U + F.

For a face F of the Newton polytope P of G, in the Lee-Pomeransky parameters
u_e, the restriction G|_F keeps the terms of G whose exponents lie on F. It
is the initial form of G for any weight w in the relative interior of the
normal cone of F: the terms of least w-degree (Fevola, Mizera and Telen,
arXiv:2311.16219, p. 28). For the weight w_gamma that is 1 on the edges of a
connected subgraph gamma and 0 elsewhere, G|_F = U_gamma G_{Gamma/gamma}
when every mass is non-zero (their Eqs. 3.13 and 3.15, pp. 27-28), and the
face of the terms free of u_e is G of Gamma/e (Britto, Grimm and Hoefnagels,
arXiv:2606.09978, Eq. 21, p. 11, for a one-particle-irreducible graph).

:func:`identify_faces` names every face in this way. It takes the weight
w = -sum m over the outward normals m of the facets containing F, which lies
in the relative interior of the normal cone, and reads a flag of subgraphs off
it: with t_1 > ... > t_k the distinct values of w_e and sigma_j the edges with
w_e >= t_j, the minors H_j = sigma_j / sigma_(j-1) keep the edges of weight
t_j, contract those above and delete those below, legs moving with their
vertices. The initial form of U is the product of the U(H_j). With j* the
last level whose F(H_j) is not zero, the prediction for G|_F is that product
with U(H_j*) replaced by G(H_j*) when t_j* = 0, by F(H_j*) when t_j* < 0,
and kept when t_j* > 0. The prediction is compared with G|_F exactly, and
a face is identified only when the two are equal. Otherwise it is a
support product when the exponents of G|_F are exactly the sums of one
exponent of each factor, the product of their supports in disjoint
variables, and unidentified when they are not; both polynomials are
recorded.

The faces of a Newton polytope that is not full-dimensional, as that of
every scaleless integral is, are not identified: the normal of a facet
relative to the affine hull is fixed only modulo the equations of the hull,
and so is the flag.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import sympy as sp

from .core.exceptions import ValidationError
from .polynomials.minors import Terms, _add, _expression, _times, minor_terms
from .polytope import Facet, PolytopeData, polytope_data

if TYPE_CHECKING:
    from .integral import FeynmanIntegral

__all__ = ["FACE_KINDS", "FaceIdentification", "FaceKind", "FlagLevel", "identify_faces"]

FaceKind = Literal[
    "whole",
    "contraction",
    "product_uv",
    "product_ir",
    "u_layer",
    "f_layer",
    "support_product",
    "unidentified",
]

# The kinds in the order the report lists them.
FACE_KINDS: tuple[FaceKind, ...] = (
    "whole",
    "contraction",
    "product_uv",
    "product_ir",
    "u_layer",
    "f_layer",
    "support_product",
    "unidentified",
)

UNIDENTIFIED = "G|_F differs from the prediction of its flag"
SUPPORT_ONLY = (
    "G|_F has the exponents of the prediction of its flag, the sums of those of its factors, "
    "but not its coefficients"
)
NOT_FULL_DIMENSIONAL = "the Newton polytope is not full-dimensional"


@dataclass(frozen=True)
class FlagLevel:
    """One level of the flag of a face: the minor H_j = sigma_j / sigma_(j-1).

    Attributes
    ----------
    edges
        The edges of weight t_j, those of H_j, by index.
    contracted
        sigma_(j-1), the edges of greater weight, contracted.
    deleted
        The edges of smaller weight, deleted.
    weight
        t_j.
    kind
        Which polynomial of H_j the prediction takes: "U", "F" or "G".
    loops
        The loop number of H_j.
    f_vanishes
        Whether F(H_j) is the zero polynomial. For a minor with loops this implies that
        H_j is scaleless by Lee's criterion, but not conversely: a massless tadpole on a
        massive line has F different from 0 and is scaleless.
    """

    edges: tuple[int, ...]
    contracted: tuple[int, ...]
    deleted: tuple[int, ...]
    weight: int
    kind: Literal["U", "F", "G"]
    loops: int
    f_vanishes: bool = False

    @property
    def trivial(self) -> bool:
        """Whether the factor is 1: U of a minor without loops."""
        return self.kind == "U" and self.loops == 0

    def minor(self) -> str:
        """H_j as a string: Gamma or the subgraph sigma_j, then /sigma_(j-1) if not empty."""
        top = "Gamma" if not self.deleted else _edge_set(self.edges + self.contracted)
        return f"{top}/{_edge_set(self.contracted)}" if self.contracted else top

    def name(self) -> str:
        """The factor as a string, such as U({1,2}) or G(Gamma/{1,2})."""
        return f"{self.kind}({self.minor()})"


@dataclass(frozen=True)
class FaceIdentification:
    """The graph of one face F of the Newton polytope of G.

    Attributes
    ----------
    point_indices
        The points of the Newton polytope on F, as indices into
        ``FeynmanIntegral.newton_polytope.points``.
    dimension, codimension
        The dimension of F and its codimension in P.
    facet
        The facet, when F is one, else None.
    weight
        w, the sum of the inward normals of the facets containing F, in
        internal-edge order; None when P is not full-dimensional.
    levels
        The flag, from the greatest weight down; empty when P is not
        full-dimensional.
    kind
        "whole" for P itself, "contraction" for G(Gamma/S), "product_uv" for
        a product whose G factor is on a quotient Gamma/S, "product_ir" for
        one whose G factor is on a minor with edges deleted, "u_layer" for a
        product of U's, "f_layer" for one with an F factor. A face whose
        G|_F is not the prediction is a "support_product" when its exponents
        are exactly the sums of one exponent of each factor, the product of
        their supports, and "unidentified" otherwise.
    contracted
        S, for a contraction.
    verified
        Whether G|_F equals the prediction exactly.
    polynomial
        G|_F.
    prediction
        The product of the level polynomials; None when P is not
        full-dimensional.
    reason
        Why F is not identified; None when it is.
    support_verified
        Whether the exponents of G|_F are exactly the product of the supports
        of the factors, the sums of one exponent of each; true when
        ``verified`` is.
    """

    point_indices: tuple[int, ...]
    dimension: int
    codimension: int
    facet: Facet | None
    weight: tuple[int, ...] | None
    levels: tuple[FlagLevel, ...]
    kind: FaceKind
    contracted: tuple[int, ...] | None
    verified: bool
    polynomial: sp.Expr
    prediction: sp.Expr | None
    reason: str | None = None
    support_verified: bool = False

    def name(self) -> str:
        """The prediction as a product of minors, such as U({3,4}) G(Gamma/{3,4}).

        Factors equal to 1 are left out; "1" when all are, "none" without a
        flag.
        """
        if not self.levels:
            return "none"
        return " ".join(level.name() for level in self.levels if not level.trivial) or "1"


def _edge_set(edges: Sequence[int]) -> str:
    return "{" + ",".join(map(str, sorted(edges))) + "}"


def check_codimension(max_codimension: object) -> None:
    """Raise ValidationError unless max_codimension is None or a non-negative integer."""
    if max_codimension is not None and (
        isinstance(max_codimension, bool)
        or not isinstance(max_codimension, int)
        or max_codimension < 0
    ):
        raise ValidationError(
            f"max_codimension must be None or a non-negative integer, not {max_codimension!r}"
        )


def identify_faces(
    fi: FeynmanIntegral,
    *,
    max_codimension: int | None = 2,
    data: PolytopeData | None = None,
) -> tuple[FaceIdentification, ...]:
    """
    Identify the faces of the Newton polytope of G with products of minor polynomials.

    Parameters
    ----------
    fi
        The integral.
    max_codimension
        Faces of codimension up to this, P itself counting as 0 and its
        facets as 1; None for every face, vertices included.
    data
        ``polytope_data(fi.newton_polytope.points)``, if already computed.

    Returns
    -------
    tuple of FaceIdentification
        One per face, by codimension and then by point indices, so that the
        facets come in the order of ``data.facets``.

    Raises
    ------
    ValidationError
        If max_codimension is not None or a non-negative integer, or data is
        not the polytope data of the Newton polytope of fi.
    """
    check_codimension(max_codimension)
    support = fi.newton_polytope.support
    points = [tuple(int(x) for x in alpha) for alpha, _ in support]
    if data is None:
        data = polytope_data(points)
    elif data.points != tuple(points):
        raise ValidationError("data is not the polytope data of the Newton polytope of fi")

    faces = sorted(
        ((data.dimension - d, idx) for d, idx in data.faces),
        key=lambda face: (face[0], face[1]),
    )
    if max_codimension is not None:
        faces = [face for face in faces if face[0] <= max_codimension]
    variables = list(fi.symanzik.lp_parameters)
    coefficients = [sp.sympify(c) for _, c in support]

    def restriction(idx: tuple[int, ...]) -> Terms:
        return {points[i]: coefficients[i] for i in idx}

    if not data.is_full_dimensional:
        return tuple(
            FaceIdentification(
                point_indices=idx,
                dimension=data.dimension - codim,
                codimension=codim,
                facet=None,
                weight=None,
                levels=(),
                kind="unidentified",
                contracted=None,
                verified=False,
                polynomial=_expression(restriction(idx), variables),
                prediction=None,
                reason=NOT_FULL_DIMENSIONAL,
            )
            for codim, idx in faces
        )

    flags = _Flags(fi)
    facets = [(frozenset(f.point_indices), f) for f in data.facets]
    found = []
    for codim, idx in faces:
        members = frozenset(idx)
        weight = [0] * len(variables)
        facet = None
        for points_on, f in facets:
            if members <= points_on:
                weight = [w - m for w, m in zip(weight, f.normal, strict=True)]
                if members == points_on:
                    facet = f
        levels, factors = flags.predict(weight)
        predicted = _product(factors)
        actual = restriction(idx)
        verified = _equal(predicted, actual)
        support_verified = verified or set(actual) == _sums([set(f) for f in factors])
        if verified:
            kind = _classify(levels)
        else:
            kind = "support_product" if support_verified else "unidentified"
        g_level = next((level for level in levels if level.kind == "G"), None)
        found.append(
            FaceIdentification(
                point_indices=idx,
                dimension=data.dimension - codim,
                codimension=codim,
                facet=facet,
                weight=tuple(weight),
                levels=levels,
                kind=kind,
                contracted=(
                    g_level.contracted if kind == "contraction" and g_level is not None else None
                ),
                verified=verified,
                polynomial=_expression(actual, variables),
                prediction=_expression(predicted, variables),
                reason=(None if verified else SUPPORT_ONLY if support_verified else UNIDENTIFIED),
                support_verified=support_verified,
            )
        )
    return tuple(found)


class _Flags:
    """The flag of a weight and its prediction, with the minors cached."""

    def __init__(self, fi: FeynmanIntegral) -> None:
        self.graph = fi.graph
        self.products: Mapping[tuple[int, int], sp.Expr] = fi.momentum_products
        self.order = tuple(e.idx for e in fi.graph.get_internal_edges())
        self.minors: dict[tuple[frozenset[int], frozenset[int]], tuple[int, Terms, Terms]] = {}

    def minor(self, edges: frozenset[int], above: frozenset[int]) -> tuple[int, Terms, Terms]:
        key = (edges, above)
        if key not in self.minors:
            below = frozenset(self.order) - edges - above
            graph, u, f = minor_terms(self.graph, self.products, above, below, self.order)
            self.minors[key] = (graph.get_loop_count(), u, f)
        return self.minors[key]

    def predict(self, weight: Sequence[int]) -> tuple[tuple[FlagLevel, ...], list[Terms]]:
        """The flag of the weight and the polynomial each level contributes to the prediction."""
        values = sorted(set(weight), reverse=True)
        above: frozenset[int] = frozenset()
        steps = []
        for t in values:
            edges = frozenset(e for e, w in zip(self.order, weight, strict=True) if w == t)
            loops, u, f = self.minor(edges, above)
            steps.append((t, edges, above, loops, u, f))
            above = above | edges
        with_f = [j for j, step in enumerate(steps) if step[5]]
        star = with_f[-1] if with_f else None

        levels = []
        factors = []
        everything = frozenset(self.order)
        for j, (t, edges, contracted, loops, u, f) in enumerate(steps):
            kind: Literal["U", "F", "G"] = "U"
            factor = u
            if j == star and t == 0:
                kind, factor = "G", _add(u, f)
            elif j == star and t < 0:
                kind, factor = "F", f
            factors.append(factor)
            levels.append(
                FlagLevel(
                    edges=tuple(sorted(edges)),
                    contracted=tuple(sorted(contracted)),
                    deleted=tuple(sorted(everything - edges - contracted)),
                    weight=t,
                    kind=kind,
                    loops=loops,
                    f_vanishes=not f,
                )
            )
        return tuple(levels), factors


def _product(factors: Sequence[Terms]) -> Terms:
    """The product of the level polynomials."""
    product: Terms = {(0,) * len(next(iter(factors[0]))): sp.Integer(1)}
    for factor in factors:
        product = _times(product, factor)
    return product


def _sums(supports: Sequence[set[tuple[int, ...]]]) -> set[tuple[int, ...]]:
    """The sums of one exponent from each support: their Minkowski sum as point sets."""
    total = {(0,) * len(next(iter(supports[0])))}
    for support in supports:
        total = {tuple(a + b for a, b in zip(x, y, strict=True)) for x in total for y in support}
    return total


def _equal(p: Terms, q: Terms) -> bool:
    """Whether two polynomials given by their terms are equal, coefficients compared exactly."""
    return p.keys() == q.keys() and all(sp.expand(p[k] - q[k]) == 0 for k in p)


def _classify(levels: Sequence[FlagLevel]) -> FaceKind:
    """The kind of a verified face from its flag."""
    kinds = [level.kind for level in levels]
    if "G" in kinds:
        g = levels[kinds.index("G")]
        rest = [level for level in levels if level is not g and not level.trivial]
        if g.deleted:
            return "product_ir"
        if rest:
            return "product_uv"
        return "contraction" if g.contracted else "whole"
    return "f_layer" if "F" in kinds else "u_layer"
