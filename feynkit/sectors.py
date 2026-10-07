"""
The sector hierarchy of a Feynman integral: which sectors vanish, and how many master
integrals each one carries.

A sector of a graph with propagators E is a subset T of E, the propagators with a positive
index, and S = E \\ T is the set of contracted propagators (R. N. Lee, arXiv:1310.1145,
section 2; Lee14 below). It is numbered by S. Weinzierl's N_id = sum_j 2^(j-1) Theta(n_j - 1/2)
(Feynman Integrals, Springer 2022, Eq. 6.16): the first propagator is the least significant
bit, as in :meth:`feynkit.family.IntegralFamily.sector_id`.

Each sector is one of three kinds.

``"cycle"``
    S contains a cycle. Some loop momentum then appears in no denominator and G_T = 0.
``"scaleless"``
    S is a forest and the origin does not lie in the affine hull of the support of G_T. This is
    Lee14's criterion (Eq. 16 and the text round it), which
    :attr:`FeynmanIntegral.is_scaleless` states for the whole graph.
``"non_zero"``
    every other sector.

The polynomial of a sector is G_T, which is G with the monomials that involve a contracted
parameter removed; it is the Lee-Pomeransky polynomial of the graph with S contracted, term
by term. So every sector is read off the support of G, at the symbolic kinematics of the
integral, and no graph is contracted and no polynomial rebuilt. Zero sectors form a down-set:
a subsector of a zero sector is zero.

Counts
------
For a sector T, with X_T the complement of {G_T = 0} in the torus (C^*)^T,

``t(T) = (-1)^|T| chi(X_T)``
    the number of master integrals of T with its subsectors (Bitoun, Bogner, Klausen and Panzer,
    arXiv:1712.09215, BBKP19 below, Cor. 37);
``m(T) = sum_{T' in T} (-1)^|T \\ T'| t(T')``
    the number of T alone, by inclusion and exclusion over its subsectors (BBKP19, Rem. 60).
    It can be negative, and BBKP19 warn that then it is not a dimension. It is reported as it
    is, and :attr:`Sector.negative_sector_count` flags it;
``t_gen(T) = |T|! Vol(F_S)``
    the count for generic coefficients on the same support (BBKP19, Thm. 44), an upper bound
    for t(T). F_S is the face of the Newton polytope of G on which the contracted parameters
    vanish, and Vol its Euclidean volume in R^T when it has dimension |T|; the count is 0
    otherwise.

``t_gen`` needs no kinematics. ``t`` and ``m`` are counts at one rational kinematic point, the
same for every sector, which each G_T takes from G.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

import sympy as sp

from . import _exact
from .core.exceptions import ComputationError, ValidationError
from .point_count import count_torus_points, critical_point_count
from .polytope import polytope_data

if TYPE_CHECKING:
    from .face_lattice import DecoratedFace, DecoratedFaceLattice
    from .integral import FeynmanIntegral

__all__ = [
    "FixedPointClass",
    "Sector",
    "SectorHierarchy",
    "SectorKind",
    "SectorTotals",
    "sector_hierarchy",
]

SectorKind = Literal["non_zero", "scaleless", "cycle"]
CountsSource = Literal["generic", "torus", "critical"]

_COUNTS = ("generic", "torus", "critical")


@dataclass(frozen=True)
class FixedPointClass:
    """The permutations of one cycle type in a sector's group, and chi of their fixed set.

    Attributes
    ----------
    cycle_type
        The lengths of the cycles of the permutation, in decreasing order.
    size
        How many permutations of the group have this cycle type.
    euler_characteristic
        chi(X_sigma) for one of them, the same for all of them, where X_sigma is the set of
        points of C^T outside {G_T = 0} that sigma fixes.
    """

    cycle_type: tuple[int, ...]
    size: int
    euler_characteristic: int


@dataclass(frozen=True)
class Sector:
    """One sector T of an integral, with its face, counts and symmetry data.

    Attributes
    ----------
    id
        The sector identity N_id = sum 2^j over the positions j of the propagators of T in the
        order of the integral's internal edges: the first propagator is the least significant
        bit.
    propagators
        The indices of the internal edges of T, increasing.
    contracted
        The indices of the other internal edges, S.
    kind
        ``"cycle"``, ``"scaleless"`` or ``"non_zero"``.
    point_indices
        The points of the Newton polytope of G that lie on the face F_S, as indices into
        ``FeynmanIntegral.newton_polytope.points``: those with every contracted exponent
        zero. Empty for a sector of kind ``"cycle"``.
    dimension
        The dimension of F_S, or -1 when there is none.
    generic_count
        t_gen(T), or None when the counts were not asked for.
    count
        t(T) at the hierarchy's point, or None when no count at a point was asked for.
    sector_count
        m(T), signed, or None as ``count``.
    orbit
        The least id in the orbit of the sector under the symmetries, or None before they
        are computed.
    stabiliser_order
        The order of the group of permutations of T that fix G_T, or None.
    fixed_points
        The classes of that group by cycle type, with chi of their fixed sets.
    symmetric_count, signed_symmetric_count
        The count of the sector with its symmetries, in the absolute and in the signed form,
        or None.
    signs_consistent
        Whether the two forms agree on this sector, or None.
    """

    id: int
    propagators: tuple[int, ...]
    contracted: tuple[int, ...]
    kind: SectorKind
    point_indices: tuple[int, ...]
    dimension: int
    generic_count: int | None = None
    count: int | None = None
    sector_count: int | None = None
    orbit: int | None = None
    stabiliser_order: int | None = None
    fixed_points: tuple[FixedPointClass, ...] = ()
    symmetric_count: Fraction | None = None
    signed_symmetric_count: Fraction | None = None
    signs_consistent: bool | None = None

    @property
    def negative_sector_count(self) -> bool:
        """Whether m(T) is negative, so that it is not a number of master integrals."""
        return self.sector_count is not None and self.sector_count < 0


@dataclass(frozen=True)
class SectorTotals:
    """The sums over a hierarchy.

    Attributes
    ----------
    non_zero, scaleless, cycle
        How many sectors have each kind; the empty sector counts as a cycle when the graph has
        a loop.
    generic
        t_gen of the top sector, or None.
    count
        t(E), the count of the top sector with its subsectors, or None.
    sector_count
        The sum of m(T) over every sector, which is t(E), or None.
    negative
        The propagators of each sector with m(T) < 0, in the order of the ids.
    """

    non_zero: int
    scaleless: int
    cycle: int
    generic: int | None
    count: int | None
    sector_count: int | None
    negative: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SectorHierarchy:
    """Every sector of an integral, by id.

    Attributes
    ----------
    edges
        The indices of the internal edges, in order; bit j of an id stands for ``edges[j]``.
    sectors
        All 2^N sectors, ``sectors[i].id == i``.
    point
        The kinematic point of the counts at a point, as (quantity, value) pairs in the form of
        :attr:`~feynkit.point_count.TorusCount.point`; None for generic counts.
    counts_source
        ``"generic"``, ``"torus"``, ``"critical"`` or None.
    """

    edges: tuple[int, ...]
    sectors: tuple[Sector, ...]
    point: tuple[tuple[sp.Expr, Fraction], ...] | None
    counts_source: CountsSource | None

    def sector(self, id_or_edges: int | Iterable[int]) -> Sector:
        """The sector with this identity, or with exactly these propagators.

        Raises
        ------
        ValidationError
            If the identity is outside [0, 2^N), or an index is not an internal edge or is
            repeated.
        """
        if isinstance(id_or_edges, int):
            if not 0 <= id_or_edges < len(self.sectors):
                raise ValidationError(
                    f"a sector identity lies in [0, {len(self.sectors)}); got {id_or_edges}"
                )
            return self.sectors[id_or_edges]
        chosen = list(id_or_edges)
        if len(set(chosen)) != len(chosen):
            raise ValidationError(f"the propagators {chosen} repeat an edge")
        unknown = [e for e in chosen if e not in self.edges]
        if unknown:
            raise ValidationError(
                f"{unknown} are not internal edges; the edges are {list(self.edges)}"
            )
        return self.sectors[sum(1 << self.edges.index(e) for e in chosen)]

    def non_zero(self) -> tuple[Sector, ...]:
        """The non-zero sectors, in the order of the ids."""
        return tuple(s for s in self.sectors if s.kind == "non_zero")

    def zero(self) -> tuple[Sector, ...]:
        """The scaleless sectors and those cut by a cycle, in the order of the ids."""
        return tuple(s for s in self.sectors if s.kind != "non_zero")

    def subsectors(self, sector_id: int) -> tuple[int, ...]:
        """The ids of the non-zero sectors strictly inside this one."""
        sector = self.sector(sector_id)
        return tuple(
            s.id
            for s in self.sectors
            if s.kind == "non_zero" and s.id != sector.id and s.id & ~sector.id == 0
        )

    def covers(self, sector_id: int) -> tuple[int, ...]:
        """The ids of the largest non-zero sectors strictly inside this one."""
        below = self.subsectors(sector_id)
        return tuple(i for i in below if not any(j != i and i & ~j == 0 for j in below))

    def totals(self) -> SectorTotals:
        """The numbers of sectors of each kind and the sums of the counts."""
        top = self.sectors[-1]
        counted = self.counts_source in ("torus", "critical")
        return SectorTotals(
            non_zero=sum(1 for s in self.sectors if s.kind == "non_zero"),
            scaleless=sum(1 for s in self.sectors if s.kind == "scaleless"),
            cycle=sum(1 for s in self.sectors if s.kind == "cycle"),
            generic=top.generic_count,
            count=top.count,
            sector_count=sum(s.sector_count or 0 for s in self.sectors) if counted else None,
            negative=tuple(s.propagators for s in self.sectors if s.negative_sector_count),
        )

    def faces(self, lattice: DecoratedFaceLattice) -> dict[int, DecoratedFace]:
        """The face F_S of each sector that has one, in the decorated face lattice.

        Parameters
        ----------
        lattice
            The face lattice of the same integral, as
            :meth:`FeynmanIntegral.face_lattice` gives it.

        Raises
        ------
        ValidationError
            If the points of some F_S are not those of a face of the lattice, which would be a
            fault in the lattice or in the hierarchy.
        """
        return {s.id: lattice.face(s.point_indices) for s in self.sectors if s.point_indices}


def _is_forest(vertices: Sequence[tuple[int, int]], chosen: Iterable[int]) -> bool:
    parent: dict[int, int] = {}

    def find(v: int) -> int:
        while parent.get(v, v) != v:
            v = parent[v]
        return v

    for i in chosen:
        a, b = find(vertices[i][0]), find(vertices[i][1])
        if a == b:
            return False
        parent[a] = b
    return True


@dataclass(frozen=True)
class _Point:
    """The terms of G at a rational kinematic point, and the point itself.

    ``numeric`` maps the exponents of each monomial to its coefficient at the point, the
    energy scale set to 1. ``given`` holds, for each symbol of G, whether the point fixes it
    by its square, and the value.
    """

    numeric: dict[tuple[int, ...], Fraction]
    given: dict[sp.Symbol, tuple[bool, Fraction]]


def _specialise(
    g: sp.Expr,
    variables: Sequence[sp.Symbol],
    scale: sp.Symbol,
    point: Mapping[sp.Expr, int | Fraction],
) -> _Point:
    """G at a point, with the energy scale set to 1.

    A key of the point is a symbol of G, or the square of one that occurs only to even powers,
    as :attr:`~feynkit.point_count.TorusCount.point` gives them.
    """
    g = g.subs(scale, 1)
    symbols = g.free_symbols - set(variables)
    given: dict[sp.Symbol, tuple[bool, Fraction]] = {}
    for key, raw in point.items():
        key = sp.sympify(key)
        if not isinstance(raw, (int, Fraction)) or isinstance(raw, bool):
            raise ValidationError(f"the point gives {key} the value {raw!r}, not a rational")
        value = Fraction(raw)
        if key.is_Pow and key.exp == 2 and key.base in symbols:
            if value < 0:
                raise ValidationError(f"the point gives the square {key} the value {value}")
            given[key.base] = (True, value)
        elif key in symbols:
            given[key] = (False, value)
        else:
            raise ValidationError(f"the point names {key}, which is not a symbol of G")
    missing = sorted(str(x) for x in symbols - set(given))
    if missing:
        raise ValidationError(f"the point misses {', '.join(missing)}")
    values = {
        x: (
            sp.sqrt(sp.Rational(v.numerator, v.denominator))
            if square
            else sp.Rational(v.numerator, v.denominator)
        )
        for x, (square, v) in given.items()
    }
    specialised = sp.expand(g.subs(values))
    if specialised == 0:
        raise ValidationError("G vanishes identically at the point")
    numeric = {
        monomial: Fraction(int(c.p), int(c.q))
        for monomial, c in sp.Poly(specialised, *variables).terms()
    }
    return _Point(numeric, given)


def _draw(
    symbolic: Sequence[tuple[tuple[int, ...], sp.Expr]], scale: sp.Symbol, seed: int
) -> tuple[tuple[sp.Expr, Fraction], ...]:
    """A kinematic point at which no coefficient of G vanishes.

    Each symbol that occurs only to even powers, such as a mass, takes a square from 1 to 20 and
    is keyed by it; each other symbol takes a non-zero integer from [-20, 20]. The first of 1000
    draws with every coefficient non-zero is returned.
    """
    symbols = sorted(
        {x for _, c in symbolic for x in c.free_symbols} - {scale}, key=lambda x: x.name
    )
    even = {
        x
        for x in symbols
        if all(e % 2 == 0 for _, c in symbolic for mon, _c in sp.Poly(c, x).terms() for e in mon)
    }
    rng = random.Random(seed)
    for _ in range(1000):
        values = [
            (
                Fraction(rng.randint(1, 20))
                if x in even
                else Fraction(rng.choice([-1, 1]) * rng.randint(1, 20))
            )
            for x in symbols
        ]
        substitution = {
            x: sp.sqrt(sp.Integer(int(v))) if x in even else sp.Integer(int(v))
            for x, v in zip(symbols, values, strict=True)
        }
        if all(sp.expand(c.subs(scale, 1).subs(substitution)) != 0 for _, c in symbolic):
            return tuple(
                (x**2 if x in even else x, v) for x, v in zip(symbols, values, strict=True)
            )
    raise ComputationError("no kinematic point without a vanishing coefficient was found")


def _polynomial(
    terms: Iterable[tuple[tuple[int, ...], sp.Expr]],
    variables: Sequence[sp.Symbol],
    chosen: Sequence[int],
) -> sp.Expr:
    """The terms that involve only the chosen variables, as a polynomial in them."""
    return sum(
        (
            c * sp.Mul(*(variables[i] ** m[i] for i in chosen))
            for m, c in terms
            if all(m[i] == 0 or i in chosen for i in range(len(m)))
        ),
        sp.Integer(0),
    )


def _sector_count(
    point: _Point,
    symbolic: Sequence[tuple[tuple[int, ...], sp.Expr]],
    variables: Sequence[sp.Symbol],
    scale: sp.Symbol,
    chosen: Sequence[int],
    source: CountsSource,
    seed: int,
    timeout: float,
) -> int:
    """t(T) at the point, for the sector whose propagators sit at the positions ``chosen``."""
    if not chosen:
        return 0
    support = [
        tuple(m[i] for i in chosen)
        for m in point.numeric
        if all(m[i] == 0 or i in chosen for i in range(len(m)))
    ]
    # No G_T, or one whose support has dimension below |T|, is quasi-homogeneous for some
    # non-zero weight (BBKP19, Prop. 41): it has no critical points.
    if not support or _exact.affine_rank(support) < len(chosen):
        return 0
    xs = [variables[i] for i in chosen]
    if source == "critical":
        numeric = _polynomial(
            ((m, sp.Rational(c.numerator, c.denominator)) for m, c in point.numeric.items()),
            variables,
            chosen,
        )
        return critical_point_count(numeric, xs, {}, seed=seed, timeout=timeout)
    # The finite-field count keeps the symbols of G_T: its fit depends on which of them are
    # squares, so it reads the point as the symbols it was drawn for.
    g = _polynomial(symbolic, variables, chosen)
    given: dict[sp.Expr, Fraction] = {}
    for x in g.free_symbols - set(xs) - {scale}:
        square, value = point.given[x]
        even = all(e % 2 == 0 for monomial, _ in sp.Poly(g, x).terms() for e in monomial)
        if even:
            given[x**2] = value if square else value**2
        else:
            given[x] = value
    result = count_torus_points(g, xs, scale=scale, seed=seed, point=given)
    if result.candidate_master_count is None:
        raise ComputationError(
            f"the point counts of the sector {list(chosen)} give no candidate: {result.reason}"
        )
    return result.candidate_master_count


def sector_hierarchy(
    fi: FeynmanIntegral,
    *,
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    counts: CountsSource | None = "generic",
    seed: int = 0,
    timeout: float = 300,
) -> SectorHierarchy:
    """Every sector of the integral, with its kind, face and counts.

    The kinds are read off the support of G at the integral's symbolic kinematics. The counts
    are described in the module's documentation.

    Parameters
    ----------
    fi
        The integral; its graph has N internal edges and 2^N sectors are built.
    point
        A rational kinematic point for ``counts`` ``"torus"`` or ``"critical"``, keyed as
        :attr:`~feynkit.point_count.TorusCount.point` is: by a symbol of G, or by the square of
        a symbol that occurs only to even powers, such as a mass. Drawn with ``seed`` when
        None, from the integers of [-20, 20] with every coefficient of G non-zero; a given
        point is better where the count must not depend on that draw.
    counts
        ``"generic"``, the default, gives only t_gen; ``"critical"`` adds t and m at a point by
        counting critical points with Singular (:func:`~feynkit.point_count.critical_point_count`);
        ``"torus"`` does so by finite-field point counts
        (:func:`~feynkit.point_count.count_torus_points`), which give no result for some
        sectors; None gives no counts.
    seed
        The seed of the point and of the random exponents of the critical-point count.
    timeout
        The most seconds for each call of Singular.

    Raises
    ------
    ValidationError
        If ``counts`` is not one of the above, ``point`` is given with a count that does not
        use it, or ``point`` is not a point of G .
    ComputationError
        If a count at a point fails, or the face of a sector with a cycle is not empty, which
        would be a fault.
    """
    if counts is not None and counts not in _COUNTS:
        raise ValidationError(
            f"counts must be one of {', '.join(map(repr, _COUNTS))} or None; got {counts!r}"
        )
    if point is not None and counts not in ("torus", "critical"):
        raise ValidationError("a point is used by the counts 'torus' and 'critical' only")

    edges = fi.graph.get_internal_edges()
    labels = tuple(e.idx for e in edges)
    vertices = [(e.v1, e.v2) for e in edges]
    n = len(edges)
    points = [tuple(int(x) for x in p) for p in fi.newton_polytope.points]

    kinds: list[SectorKind] = []
    faces: list[tuple[int, ...]] = []
    dimensions: list[int] = []
    for mask in range(1 << n):
        chosen = [j for j in range(n) if mask >> j & 1]
        contracted = [j for j in range(n) if not mask >> j & 1]
        face = tuple(j for j, p in enumerate(points) if not any(p[i] for i in contracted))
        if not _is_forest(vertices, contracted):
            if face:
                raise ComputationError(
                    f"the sector {[labels[j] for j in chosen]} contains a cycle and has a face"
                )
            kinds.append("cycle")
            faces.append(())
            dimensions.append(-1)
            continue
        restricted = [tuple(points[j][i] for i in chosen) for j in face]
        dimension = _exact.affine_rank(restricted) if face else -1
        # Lee's criterion: the origin lies off the affine hull of the support.
        zero = not face or _exact.affine_rank([*restricted, (0,) * len(chosen)]) > dimension
        kinds.append("scaleless" if zero else "non_zero")
        faces.append(face)
        dimensions.append(dimension)

    generic: list[int | None] = [None] * (1 << n)
    if counts is not None:
        for mask in range(1 << n):
            chosen = [j for j in range(n) if mask >> j & 1]
            generic[mask] = 0
            if chosen and dimensions[mask] == len(chosen):
                data = polytope_data([tuple(points[j][i] for i in chosen) for j in faces[mask]])
                generic[mask] = data.normalized_volume * data.sublattice_index

    t: list[int | None] = [None] * (1 << n)
    m: list[int | None] = [None] * (1 << n)
    drawn: tuple[tuple[sp.Expr, Fraction], ...] | None = None
    if counts in ("torus", "critical"):
        symanzik = fi.symanzik
        variables = list(symanzik.lp_parameters)
        scale = fi.graph.energy_scale
        if point is None:
            drawn = _draw(sp.Poly(symanzik.g, *variables).terms(), scale, seed)
            used: Mapping[sp.Expr, int | Fraction] = dict(drawn)
        else:
            used = point
        at_point = _specialise(symanzik.g, variables, scale, used)
        symbolic = sp.Poly(symanzik.g, *variables).terms()
        if drawn is None:
            drawn = tuple((sp.sympify(k), Fraction(v)) for k, v in used.items())
        for mask in range(1 << n):
            chosen = [j for j in range(n) if mask >> j & 1]
            t[mask] = (
                0
                if kinds[mask] != "non_zero"
                else _sector_count(
                    at_point, symbolic, variables, scale, chosen, counts, seed, timeout
                )
            )
        # Inclusion and exclusion over the subsectors, in place: the sum over T' in T.
        values = [int(v or 0) for v in t]
        for bit in range(n):
            for mask in range(1 << n):
                if mask >> bit & 1:
                    values[mask] -= values[mask ^ (1 << bit)]
        m = list(values)

    sectors = tuple(
        Sector(
            id=mask,
            propagators=tuple(labels[j] for j in range(n) if mask >> j & 1),
            contracted=tuple(labels[j] for j in range(n) if not mask >> j & 1),
            kind=kinds[mask],
            point_indices=faces[mask],
            dimension=dimensions[mask],
            generic_count=generic[mask],
            count=t[mask],
            sector_count=m[mask],
        )
        for mask in range(1 << n)
    )
    return SectorHierarchy(edges=labels, sectors=sectors, point=drawn, counts_source=counts)
