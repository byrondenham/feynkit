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

import itertools
import random
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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
    "fixed_point_euler_characteristic",
    "parameter_permutations",
    "sector_hierarchy",
    "stabiliser",
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
    unique
        The number of orbits of non-zero sectors, or None when the symmetries were not
        computed.
    symmetric, signed_symmetric
        The sums of the sector counts with symmetries over the unique sectors, in the absolute
        and the signed form, or None when they were not computed.
    signs_consistent
        False if some unique sector has two forms that differ, True if all agree, None when
        there are no such counts.
    """

    non_zero: int
    scaleless: int
    cycle: int
    generic: int | None
    count: int | None
    sector_count: int | None
    negative: tuple[tuple[int, ...], ...]
    unique: int | None = None
    symmetric: Fraction | None = None
    signed_symmetric: Fraction | None = None
    signs_consistent: bool | None = None


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
    symmetric
        Whether the orbits of the non-zero sectors were computed.
    """

    edges: tuple[int, ...]
    sectors: tuple[Sector, ...]
    point: tuple[tuple[sp.Expr, Fraction], ...] | None
    counts_source: CountsSource | None
    symmetric: bool = False
    _terms: Terms = field(default_factory=dict, repr=False, compare=False)

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

    def _sector_terms(self, sector: Sector) -> Terms:
        """The terms of G_T, over the variables of T alone."""
        positions = [self.edges.index(e) for e in sector.propagators]
        outside = [j for j in range(len(self.edges)) if j not in positions]
        return {
            tuple(m[i] for i in positions): c
            for m, c in self._terms.items()
            if not any(m[j] for j in outside)
        }

    def _non_zero_sector(self, sector_id: int) -> Sector:
        sector = self.sector(sector_id)
        if sector.kind != "non_zero":
            raise ValidationError(f"the sector {sector.id} is not non-zero, so it has no maps")
        return sector

    def maps(self, id1: int, id2: int) -> list[dict[int, int]]:
        """The bijections of propagators that map the terms of G_T1 onto those of G_T2.

        Each is a dict from the propagators of the first sector to those of the second, and the
        terms are matched with exactly equal coefficients, at the symbolic kinematics of the
        integral or at the point the symmetries were computed at. The list is empty when the
        sectors have different sizes or no such bijection exists.

        Raises
        ------
        ValidationError
            If a sector is not non-zero.
        """
        a, b = self._non_zero_sector(id1), self._non_zero_sector(id2)
        if len(a.propagators) != len(b.propagators):
            return []
        found = _maps(self._sector_terms(a), self._sector_terms(b), len(a.propagators))
        return [
            {a.propagators[i]: b.propagators[j] for i, j in enumerate(sigma)}
            for sigma in sorted(found)
        ]

    def stabiliser(self, sector_id: int) -> list[dict[int, int]]:
        """The group of G_T: the maps of the sector to itself, the identity among them."""
        return self.maps(sector_id, sector_id)

    def orbit(self, sector_id: int) -> int:
        """The least id in the orbit of a non-zero sector under the symmetries.

        Raises
        ------
        ValidationError
            If the symmetries were not computed or the sector is not non-zero.
        """
        sector = self._non_zero_sector(sector_id)
        if sector.orbit is None:
            raise ValidationError("the symmetries were not computed; pass symmetries=True")
        return sector.orbit

    def orbits(self) -> dict[int, tuple[int, ...]]:
        """The orbits of the non-zero sectors under the groupoid of maps, by least id.

        The orbits are the unique sectors, each represented by the least N_id it holds.

        Raises
        ------
        ValidationError
            If the symmetries were not computed.
        """
        if not self.symmetric:
            raise ValidationError("the symmetries were not computed; pass symmetries=True")
        found: dict[int, list[int]] = {}
        for s in self.sectors:
            if s.orbit is not None:
                found.setdefault(s.orbit, []).append(s.id)
        return {rep: tuple(members) for rep, members in found.items()}

    def representatives(self) -> tuple[int, ...]:
        """The least id of each orbit, in increasing order."""
        return tuple(self.orbits())

    def totals(self) -> SectorTotals:
        """The numbers of sectors of each kind and the sums of the counts."""
        top = self.sectors[-1]
        counted = self.counts_source in ("torus", "critical")
        reps = [s for s in self.sectors if s.orbit == s.id]
        with_n = [s for s in reps if s.symmetric_count is not None]
        flags = [s.signs_consistent for s in with_n if s.signs_consistent is not None]
        return SectorTotals(
            non_zero=sum(1 for s in self.sectors if s.kind == "non_zero"),
            scaleless=sum(1 for s in self.sectors if s.kind == "scaleless"),
            cycle=sum(1 for s in self.sectors if s.kind == "cycle"),
            generic=top.generic_count,
            count=top.count,
            sector_count=sum(s.sector_count or 0 for s in self.sectors) if counted else None,
            negative=tuple(s.propagators for s in self.sectors if s.negative_sector_count),
            unique=len(reps) if self.symmetric else None,
            symmetric=(
                sum((s.symmetric_count or Fraction(0) for s in with_n), Fraction(0))
                if with_n
                else None
            ),
            signed_symmetric=(
                sum((s.signed_symmetric_count or Fraction(0) for s in with_n), Fraction(0))
                if with_n
                else None
            ),
            signs_consistent=all(flags) if flags else None,
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
    values: dict[sp.Symbol, sp.Expr]


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
    return _Point(numeric, given, values)


def _draw(
    symbolic: Sequence[tuple[tuple[int, ...], sp.Expr]], scale: sp.Symbol, seed: int
) -> tuple[tuple[sp.Expr, Fraction], ...]:
    """A kinematic point at which no coefficient of G vanishes.

    Each symbol takes an integer from [1, 2^20], so that a discriminant vanishes by accident
    with negligible probability. A symbol that occurs only to even powers, such as a mass, is
    keyed by its square and takes the square from the same range. The first of 1000 draws with
    every coefficient non-zero is returned.
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
        values = [Fraction(rng.randint(1, 2**20)) for _ in symbols]
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


Terms = dict[tuple[int, ...], Hashable]
Permutation = tuple[int, ...]


def _terms(expr: sp.Expr, variables: Sequence[sp.Symbol]) -> Terms:
    """The terms of a polynomial in the variables, as {exponents: coefficient}."""
    if expr == 0:
        return {}
    poly = sp.Poly(sp.expand(expr), *variables)
    return dict(poly.terms())


def _signatures(terms: Terms, size: int) -> list[Hashable]:
    """An invariant of each variable under the permutations that preserve the terms.

    For the variable v, the multiset of (exponent of v, degree, coefficient) over the terms that
    contain it.
    """
    return [
        frozenset(Counter((m[v], sum(m), c) for m, c in terms.items() if m[v]).items())
        for v in range(size)
    ]


def _maps(f: Terms, g: Terms, size: int) -> Iterator[Permutation]:
    """The permutations sigma of range(size) with f(sigma x) = g(x) term by term.

    sigma[i] is the variable of g that variable i of f goes to. The variables are refined by
    their signatures, then assigned one at a time, rarest signature first, and a prefix is kept
    only when the terms of f and of g project to the same multiset on the assigned variables.
    """
    if len(f) != len(g) or Counter(f.values()) != Counter(g.values()):
        return
    sf, sg = _signatures(f, size), _signatures(g, size)
    if Counter(sf) != Counter(sg):
        return
    candidates = [[w for w in range(size) if sg[w] == sf[v]] for v in range(size)]
    order = sorted(range(size), key=lambda v: (len(candidates[v]), v))
    image: dict[int, int] = {}

    def projection(terms: Terms, variables: Sequence[int]) -> Counter[tuple[int, ...]]:
        return Counter(tuple(m[v] for v in variables) for m in terms)

    def extend(depth: int) -> Iterator[Permutation]:
        if depth == size:
            if all(
                g.get(tuple(m[i] for i in sorted(range(size), key=lambda i: image[i]))) == c
                for m, c in f.items()
            ):
                yield tuple(image[v] for v in range(size))
            return
        v = order[depth]
        taken = set(image.values())
        for w in candidates[v]:
            if w in taken:
                continue
            image[v] = w
            assigned = order[: depth + 1]
            if projection(f, assigned) == projection(g, [image[u] for u in assigned]):
                yield from extend(depth + 1)
            del image[v]

    yield from extend(0)


def _compose_conjugate(tau: Permutation, sigma: Permutation) -> Permutation:
    """tau sigma tau^-1."""
    inverse = [0] * len(tau)
    for i, j in enumerate(tau):
        inverse[j] = i
    return tuple(tau[sigma[inverse[i]]] for i in range(len(tau)))


def _cycles(sigma: Permutation) -> list[tuple[int, ...]]:
    seen: set[int] = set()
    cycles = []
    for start in range(len(sigma)):
        if start in seen:
            continue
        cycle, i = [], start
        while i not in seen:
            seen.add(i)
            cycle.append(i)
            i = sigma[i]
        cycles.append(tuple(cycle))
    return cycles


def _classes(group: Sequence[Permutation]) -> list[tuple[Permutation, int]]:
    """The conjugacy classes of a group of permutations, as (a member, the size)."""
    remaining = set(group)
    classes = []
    for sigma in group:
        if sigma not in remaining:
            continue
        orbit = {_compose_conjugate(tau, sigma) for tau in group}
        remaining -= orbit
        classes.append((sigma, len(orbit)))
    return classes


def _check_variables(f: sp.Expr, x: Sequence[sp.Symbol], where: str) -> tuple[sp.Symbol, ...]:
    variables = tuple(x)
    if len(set(variables)) != len(variables):
        raise ValidationError(f"the variables of {where} must be distinct symbols")
    if not sp.sympify(f).is_polynomial(*variables):
        raise ValidationError(f"{where} must be a polynomial in its variables")
    return variables


def parameter_permutations(
    f: sp.Expr, x: Sequence[sp.Symbol], g: sp.Expr, y: Sequence[sp.Symbol]
) -> list[Permutation]:
    """The bijections of the variables that map the terms of f onto those of g.

    A permutation sigma, with sigma[i] the position in y that x_i goes to, is returned when
    f(x_i -> y_sigma(i)) and g have the same monomials with exactly equal coefficients, as
    expressions in whatever other symbols they hold. These are the symmetries of Feynman
    parameters between two sector polynomials, S(G_1, G_2) of Duhr, Maggio, Semper and
    Stawinski (arXiv:2604.08332). The list is empty when the variables are not as many.

    Raises
    ------
    ValidationError
        If the variables are not distinct symbols, or f or g is not a polynomial in its own.
    """
    xs = _check_variables(f, x, "f")
    ys = _check_variables(g, y, "g")
    if len(xs) != len(ys):
        return []
    return sorted(_maps(_terms(f, xs), _terms(g, ys), len(xs)))


def stabiliser(f: sp.Expr, x: Sequence[sp.Symbol]) -> list[Permutation]:
    """The permutations of the variables that leave f unchanged, as :func:`parameter_permutations`
    gives them for g = f."""
    return parameter_permutations(f, x, f, x)


def _fixed_euler_characteristic(
    terms: Mapping[tuple[int, ...], Fraction],
    sigma: Permutation,
    seed: int,
    timeout: float,
    cache: dict[tuple[tuple[tuple[int, ...], Fraction], ...], int],
) -> int:
    """chi(X_sigma) for X = C^n minus {G = 0}, G given by its terms at a point.

    X_sigma is the set of points of X that sigma fixes, the points with equal coordinates on
    each cycle. It is stratified by which cycles of coordinates vanish; the stratum on which
    the cycles in ``chosen`` do not is a torus (C^*)^c minus the zeros of G with each cycle's
    variables set equal, and has chi = (-1)^c times the number of critical points (Fevola,
    Mizera and Telen, arXiv:2311.16219, proof of Thm. 3.1).
    """
    cycles = _cycles(sigma)
    total = 0
    for r in range(1, len(cycles) + 1):
        for chosen in itertools.combinations(cycles, r):
            inside = {i for cycle in chosen for i in cycle}
            merged: dict[tuple[int, ...], Fraction] = {}
            for m, c in terms.items():
                if any(m[i] for i in range(len(m)) if i not in inside):
                    continue
                exponents = tuple(sum(m[i] for i in cycle) for cycle in chosen)
                merged[exponents] = merged.get(exponents, Fraction(0)) + c
            live = {k: c for k, c in merged.items() if c}
            if not live or _exact.affine_rank(list(live)) < r:
                continue
            key = tuple(sorted(live.items()))
            if key not in cache:
                ys = sp.symbols(f"y0:{r}")
                g = sum(
                    (
                        sp.Rational(c.numerator, c.denominator)
                        * sp.Mul(*(y**e for y, e in zip(ys, k, strict=True)))
                        for k, c in live.items()
                    ),
                    sp.Integer(0),
                )
                cache[key] = critical_point_count(g, ys, {}, seed=seed, timeout=timeout)
            total += (-1) ** r * cache[key]
    return total


def fixed_point_euler_characteristic(
    f: sp.Expr,
    x: Sequence[sp.Symbol],
    sigma: Sequence[int],
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    *,
    seed: int = 0,
    timeout: float = 300,
) -> int:
    """chi(X_sigma) for X = C^n minus {f = 0}, at a kinematic point, by counting critical points.

    X_sigma is the set of points of X fixed by the permutation sigma of the variables, which
    must be a symmetry of f at the point (Duhr, Maggio, Semper and Stawinski,
    arXiv:2604.08332, Eqs. 8.21-8.26). Needs Singular, through
    :func:`~feynkit.point_count.critical_point_count`.

    Parameters
    ----------
    f, x
        The polynomial and its variables; any other symbol is given a value by ``point``, as
        :attr:`~feynkit.point_count.TorusCount.point` keys it.
    sigma
        sigma[i] is the position the variable x_i goes to.
    seed, timeout
        As for :func:`~feynkit.point_count.critical_point_count`.

    Raises
    ------
    ValidationError
        If sigma is not a permutation or is not a symmetry of f at the point, or the point
        does not fix every other symbol of f.
    """
    variables = _check_variables(f, x, "f")
    perm = tuple(int(i) for i in sigma)
    if sorted(perm) != list(range(len(variables))):
        raise ValidationError(f"sigma must be a permutation of range({len(variables)})")
    scale = sp.Dummy("scale")
    terms = _specialise(sp.sympify(f), variables, scale, point or {}).numeric
    if perm not in set(_maps(terms, terms, len(variables))):  # type: ignore[arg-type]
        raise ValidationError(f"{perm} is not a symmetry of f at the point")
    return _fixed_euler_characteristic(terms, perm, seed, timeout, {})


def sector_hierarchy(
    fi: FeynmanIntegral,
    *,
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    counts: CountsSource | None = "generic",
    symmetries: bool = True,
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
        a symbol that occurs only to even powers, such as a mass. A point given is used as
        given: nothing checks it. When None it is drawn with ``seed`` from the integers of
        [1, 2^20] with every coefficient of G non-zero, and the count of the top sector is
        recomputed at a second point, drawn with ``seed + 1``, and the two must agree.
    counts
        ``"generic"``, the default, gives only t_gen; ``"critical"`` adds t and m at a point by
        counting critical points with Singular (:func:`~feynkit.point_count.critical_point_count`);
        ``"torus"`` does so by finite-field point counts
        (:func:`~feynkit.point_count.count_torus_points`), which give no result for some
        sectors; None gives no counts.
    symmetries
        Whether to find the groupoid of parameter permutations between the non-zero sectors,
        their orbits and stabilisers, and, with a count at a point, the counts with symmetries
        N_T of each unique sector in both forms of Duhr, Maggio, Semper and Stawinski
        (arXiv:2604.08332, Eqs. 8.18 and 8.3). The permutations are found at the symbolic
        kinematics of the integral, or at ``point`` when one is given. The fixed-point Euler
        characteristics use Singular. N_T is not computed for a unique sector with m(T) = 0,
        and is then 0 with ``signs_consistent`` None.
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
        If a count at a point fails, the top sector's count differs at the second point, or the
        face of a sector with a cycle is not empty, which would be a fault.
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
    symanzik = fi.symanzik
    variables = list(symanzik.lp_parameters)
    scale = fi.graph.energy_scale
    symbolic = sp.Poly(symanzik.g, *variables).terms()
    at_point: _Point | None = None
    if counts in ("torus", "critical"):
        if point is None:
            drawn = _draw(symbolic, scale, seed)
            used: Mapping[sp.Expr, int | Fraction] = dict(drawn)
        else:
            used = point
        at_point = _specialise(symanzik.g, variables, scale, used)
        if point is not None:
            vanishing = [
                m for m, c in symbolic if sp.expand(c.subs(scale, 1).subs(at_point.values)) == 0
            ]
            if vanishing:
                raise ValidationError(
                    f"{len(vanishing)} coefficients of G vanish at the point, so its sectors "
                    "differ from those at the integral's kinematics: specialise the integral "
                    "to those kinematics first, for example with with_, rather than pass such "
                    "a point"
                )
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
        top = (1 << n) - 1
        if point is None and kinds[top] == "non_zero":
            second = _specialise(
                symanzik.g, variables, scale, dict(_draw(symbolic, scale, seed + 1))
            )
            again = _sector_count(
                second, symbolic, variables, scale, list(range(n)), counts, seed, timeout
            )
            if again != t[top]:
                raise ComputationError(
                    f"the count of the top sector is {t[top]} at the first point and {again} at "
                    "the second point, so the point is not generic; give a point or another seed"
                )
        # Inclusion and exclusion over the subsectors, in place: the sum over T' in T.
        values = [int(v or 0) for v in t]
        for bit in range(n):
            for mask in range(1 << n):
                if mask >> bit & 1:
                    values[mask] -= values[mask ^ (1 << bit)]
        m = list(values)

    # The terms the symmetries are searched in: symbolic, or at the point when one is given.
    searched: Terms = (
        dict(at_point.numeric) if at_point is not None and point is not None else dict(symbolic)
    )
    orbit: list[int | None] = [None] * (1 << n)
    order: list[int | None] = [None] * (1 << n)
    classes: dict[int, list[tuple[Permutation, int]]] = {}
    if symmetries:
        reps: dict[tuple[int, Hashable], list[int]] = {}
        local: dict[int, Terms] = {}

        def terms_of(mask: int) -> Terms:
            if mask not in local:
                chosen = [j for j in range(n) if mask >> j & 1]
                outside = [j for j in range(n) if not mask >> j & 1]
                local[mask] = {
                    tuple(mm[i] for i in chosen): c
                    for mm, c in searched.items()
                    if not any(mm[j] for j in outside)
                }
            return local[mask]

        for mask in range(1 << n):
            if kinds[mask] != "non_zero":
                continue
            size = mask.bit_count()
            invariant = (
                size,
                frozenset(Counter(_signatures(terms_of(mask), size)).items()),
            )
            for rep in reps.setdefault(invariant, []):
                if next(_maps(terms_of(rep), terms_of(mask), size), None) is not None:
                    orbit[mask] = rep
                    break
            else:
                reps[invariant].append(mask)
                orbit[mask] = mask
        group: dict[int, list[Permutation]] = {}
        for mask in range(1 << n):
            if orbit[mask] == mask:
                size = mask.bit_count()
                group[mask] = sorted(_maps(terms_of(mask), terms_of(mask), size))
                classes[mask] = _classes(group[mask])
        for mask in range(1 << n):
            if orbit[mask] is not None:
                order[mask] = len(group[orbit[mask]])  # type: ignore[index]

    symmetric: dict[int, tuple[Fraction, Fraction, bool | None, tuple[FixedPointClass, ...]]] = {}
    if symmetries and at_point is not None:
        cache: dict[tuple[tuple[tuple[int, ...], Fraction], ...], int] = {}
        for mask, kinds_of in classes.items():
            size = mask.bit_count()
            if not m[mask]:
                symmetric[mask] = (Fraction(0), Fraction(0), None, ())
                continue
            chosen = [j for j in range(n) if mask >> j & 1]
            outside = [j for j in range(n) if not mask >> j & 1]
            numeric = {
                tuple(mm[i] for i in chosen): c
                for mm, c in at_point.numeric.items()
                if not any(mm[j] for j in outside)
            }
            found = []
            absolute = Fraction(0)
            signed = Fraction(0)
            for sigma, count in kinds_of:
                chi = _fixed_euler_characteristic(numeric, sigma, seed, timeout, cache)
                cycles = len(_cycles(sigma))
                found.append(
                    FixedPointClass(
                        tuple(sorted((len(c) for c in _cycles(sigma)), reverse=True)), count, chi
                    )
                )
                absolute += count * abs(chi)
                signed += count * (-1) ** (size - cycles) * chi
            total = Fraction(sum(c for _, c in kinds_of))
            absolute /= total
            signed = signed * (-1) ** size / total
            symmetric[mask] = (absolute, signed, absolute == signed, tuple(found))

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
            orbit=orbit[mask],
            stabiliser_order=order[mask],
            fixed_points=symmetric[mask][3] if mask in symmetric else (),
            symmetric_count=symmetric[mask][0] if mask in symmetric else None,
            signed_symmetric_count=symmetric[mask][1] if mask in symmetric else None,
            signs_consistent=symmetric[mask][2] if mask in symmetric else None,
        )
        for mask in range(1 << n)
    )
    return SectorHierarchy(
        edges=labels,
        sectors=sectors,
        point=drawn,
        counts_source=counts,
        symmetric=symmetries,
        _terms=searched,
    )
