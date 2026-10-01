"""
The decorated face lattice of a GKZ configuration.

Every face G of a homogeneous configuration A, the empty face included, is
decorated with the values of eps at which it is resonant, beta in
span_C(A_G) + ZA, and admissible, beta in span_C(A_G) (Britto, Grimm and
Hoefnagels, arXiv:2606.09978, Eq. 23, p. 12, and pp. 10-11), for a parameter
beta(eps) = beta_0 + eps beta_1; with its lattice defect, the number of
columns off it, and whether A is a pyramid over it (Schulze and Walther,
arXiv:1009.3569, Def. 3.4, p. 6).

The face test. Let N_G be the group of linear forms on span_C(A) that take
integer values on ZA and vanish on the columns of G, free of rank
rank A - rank A_G. G is resonant exactly when beta lies in the span of A and
every form of a Z-basis of N_G is an integer on beta, and admissible exactly
when every such form vanishes on beta. For a facet, N_G is spanned by its
primitive form l_F, and the test is that of :mod:`feynkit.resonance`. The
forms l_F of the facets containing G span a subgroup of N_G of finite index
|T_G|, the lattice defect. G is resonant wherever all those facets are, for
every beta in the span of A, exactly when T_G is trivial. The design note
docs/design/2026-09-30-face-lattice.md proves both statements.

With p = H beta_0 and q = H beta_1 for a basis H of N_G, the resonant eps are
all or none when q = 0. Otherwise q = s u with s > 0 and u a primitive integer
vector, and with v . u = 1 and t_0 = -v . p they form the progression
(t_0 + Z)/s when p + t_0 u is integral, and are none when it is not. Both
sets are intersected with the eps at which beta lies in the span of A.

Reducibility. At a given eps the resonance centres are the minimal resonant
faces (Schulze and Walther, Def. 3.2, p. 5). M_A(beta) has reducible monodromy
when A is not a pyramid over a centre (Thm 4.1, p. 7) and irreducible
monodromy when A is a pyramid over one (Thm 5.1, pp. 7-8), which is then the
only centre (Prop. 3.8, p. 6). Their results need ZA of rank d, the number of
rows of A (Rem. 2.1, p. 3), so ``reducible`` is None when A has smaller rank.
It is also None when no face is resonant, which happens only off the span of A,
where the system has no non-zero solutions.

Everything is exact: Python integers and Fractions.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from . import _exact
from .core.exceptions import ComputationError, ValidationError
from .polytope import PolytopeData, polytope_data
from .resonance import (
    D0Source,
    EpsilonSet,
    FacetResonance,
    _distinct,
    _rational,
    _restrict,
    _rows,
    choose_d0,
    classify_facets,
)
from .systems.cayley import lp_to_cayley

if TYPE_CHECKING:
    from .face_identification import FaceIdentification
    from .integral import FeynmanIntegral

__all__ = [
    "DecoratedConfiguration",
    "DecoratedFace",
    "DecoratedFaceLattice",
    "Epsilon",
    "SchwingerCheck",
    "decorate_configuration",
    "decorate_faces",
]

Epsilon = int | Fraction | Literal["generic"]

_ALL = EpsilonSet("all")
_NEVER = EpsilonSet("never")


@dataclass(frozen=True)
class DecoratedFace:
    """One face G of a configuration A, with its resonance, admissibility and lattice data.

    Attributes
    ----------
    point_indices
        The columns of A on G, in increasing order; for a Newton polytope, the
        indices of its points on G. Empty for the empty face.
    dimension
        The dimension of G as a face of the polytope of A, rank A_G - 1; -1 for
        the empty face.
    codimension
        The dimension of the polytope minus that of G.
    facets
        The facets containing G, by their position in the ``facets`` of the
        lattice.
    resonant
        The eps at which G is resonant: beta(eps) in span_C(A_G) + ZA.
    admissible
        The eps at which beta(eps) lies in span_C(A_G): every eps, one value or
        none. The offset of a resonant progression is the admissible value
        when there is one, as for a facet in :mod:`feynkit.resonance`, and
        otherwise lies in [0, period).
    lattice_defect
        |T_G|, the index in N_G of the subgroup spanned by the forms of the
        facets containing G; 1 when they span N_G.
    columns_off
        The number of columns of A off G.
    pyramid
        Whether A is a pyramid over G, rank ZA = columns_off + rank ZA_G
        (Schulze and Walther, Def. 3.4, p. 6). It always is over the whole of A.
    through_origin
        For a Newton polytope, whether the origin lies in the affine hull of
        the points on G; for a facet m . x <= b, whether b = 0. False for the
        empty face, and None for any other configuration.
    identification
        For a Newton polytope, the graph of G that
        :func:`feynkit.face_identification.identify_faces` gives, when G lies
        within the codimension identified; None otherwise and for the empty
        face.
    """

    point_indices: tuple[int, ...]
    dimension: int
    codimension: int
    facets: tuple[int, ...]
    resonant: EpsilonSet
    admissible: EpsilonSet
    lattice_defect: int
    columns_off: int
    pyramid: bool
    through_origin: bool | None = None
    identification: FaceIdentification | None = None


@dataclass(frozen=True)
class DecoratedConfiguration:
    """Every face of a homogeneous configuration A, decorated for beta(eps) = beta_0 + eps beta_1.

    Attributes
    ----------
    columns
        The columns of A.
    rank
        The rank of A, that of ZA.
    span
        The eps at which beta(eps) lies in the span of A: every eps, one value
        or none. Every set of a face is contained in it.
    faces
        Every face, by dimension and then by point indices, so that the empty
        face comes first and the whole of A last.
    facets
        The positions in ``faces`` of the facets, the faces of codimension 1,
        in the order of ``PolytopeData.relative_facets``, by point indices.
    """

    columns: tuple[tuple[int, ...], ...]
    rank: int
    span: EpsilonSet
    faces: tuple[DecoratedFace, ...]
    facets: tuple[int, ...]

    @property
    def full_rank(self) -> bool:
        """Whether the rank of A equals its number of rows, as Schulze and Walther assume."""
        return self.rank == len(self.columns[0])

    @cached_property
    def _by_points(self) -> dict[tuple[int, ...], DecoratedFace]:
        return {face.point_indices: face for face in self.faces}

    def face(self, points: Iterable[int]) -> DecoratedFace:
        """The face on exactly these columns, in any order.

        Raises
        ------
        ValidationError
            If an index is not an integer, or the columns are not those of a face.
        """
        key = tuple(sorted({_exact._as_int(j, "a point index") for j in points}))
        found = self._by_points.get(key)
        if found is None:
            raise ValidationError(f"the columns {key} are not a face")
        return found

    def resonance(self, points: Iterable[int]) -> EpsilonSet:
        """The eps at which the face on these columns is resonant; see :meth:`face`."""
        return self.face(points).resonant

    def admissible(self, points: Iterable[int]) -> EpsilonSet:
        """The eps at which the face on these columns is admissible; see :meth:`face`."""
        return self.face(points).admissible

    def defect(self, points: Iterable[int]) -> int:
        """|T_G| of the face on these columns; see :meth:`face`."""
        return self.face(points).lattice_defect

    def resonant_faces(self, eps: Epsilon) -> tuple[DecoratedFace, ...]:
        """The faces resonant at eps, in the order of ``faces``.

        eps is an integer, a Fraction, or "generic" for every eps outside a
        countable set, where exactly the faces resonant for every eps are.

        Raises
        ------
        ValidationError
            If eps is none of these.
        """
        value = _epsilon(eps)
        if value is None:
            return tuple(face for face in self.faces if face.resonant.kind == "all")
        return tuple(face for face in self.faces if value in face.resonant)

    def centres(self, eps: Epsilon) -> tuple[DecoratedFace, ...]:
        """The resonance centres at eps, the minimal resonant faces, in the order of ``faces``.

        Raises
        ------
        ValidationError
            As :meth:`resonant_faces`.
        """
        found: list[DecoratedFace] = []
        sets: list[frozenset[int]] = []
        for face in self.resonant_faces(eps):
            points = frozenset(face.point_indices)
            if not any(smaller <= points for smaller in sets):
                found.append(face)
                sets.append(points)
        return tuple(found)

    def reducible(self, eps: Epsilon) -> bool | None:
        """Whether M_A(beta(eps)) has reducible monodromy, by Schulze and Walther.

        True when A is a pyramid over no resonance centre (Thm 4.1), False
        when it is over one (Thm 5.1). None when A does not have full rank, and
        when no face is resonant, where beta is off the span of A and the
        system has no non-zero solutions.

        Raises
        ------
        ValidationError
            As :meth:`resonant_faces`.
        ComputationError
            If a centre over which A is a pyramid is not the only one, which
            Prop. 3.8 of Schulze and Walther rules out; it would be a bug.
        """
        centres = self.centres(eps)
        if not self.full_rank or not centres:
            return None
        if any(face.pyramid for face in centres):
            if len(centres) != 1:
                raise ComputationError("a pyramid centre is not the only resonance centre")
            return False
        return True


def _epsilon(eps: object) -> Fraction | None:
    """eps as a Fraction, or None for "generic"."""
    if isinstance(eps, str):
        if eps != "generic":
            raise ValidationError(f'eps must be a rational number or "generic"; got {eps!r}')
        return None
    return _rational(eps, "eps")


# --- sets of eps ---------------------------------------------------------------------


def _bezout(u: Sequence[int]) -> list[int]:
    """Integers v with v . u = gcd(u)."""
    g, v = 0, [0] * len(u)
    for i, x in enumerate(u):
        if x:
            a, b, g = _exact._gcdex(g, x)
            v = [a * c for c in v]
            v[i] += b
    return v


def _integral(p: Sequence[Fraction], q: Sequence[Fraction]) -> EpsilonSet:
    """The eps with p + eps q an integer vector."""
    if not any(q):
        return _ALL if all(x.denominator == 1 for x in p) else _NEVER
    scale = math.lcm(*(x.denominator for x in q))
    whole = [int(x * scale) for x in q]
    g = math.gcd(*whole)
    u = [x // g for x in whole]
    s = Fraction(g, scale)
    t0 = -sum((a * b for a, b in zip(_bezout(u), p, strict=True)), Fraction(0))
    if any((a + t0 * b).denominator != 1 for a, b in zip(p, u, strict=True)):
        return _NEVER
    period = 1 / s
    return EpsilonSet("progression", (t0 / s) % period, period)


def _zero(p: Sequence[Fraction], q: Sequence[Fraction]) -> EpsilonSet:
    """The eps with p + eps q = 0."""
    if not any(q):
        return _NEVER if any(p) else _ALL
    i = next(i for i, x in enumerate(q) if x)
    value = -p[i] / q[i]
    return (
        EpsilonSet("point", value)
        if all(a + value * b == 0 for a, b in zip(p, q, strict=True))
        else _NEVER
    )


def _canonical(resonant: EpsilonSet, admissible: EpsilonSet) -> EpsilonSet:
    """A resonant progression with its offset at the admissible value, when there is one."""
    if resonant.kind == "progression" and admissible.kind == "point":
        return EpsilonSet("progression", admissible.offset, resonant.period)
    return resonant


# --- the decoration ----------------------------------------------------------------


def _dot(u: Sequence[int], v: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(u, v, strict=True)), Fraction(0))


def _kernel(vectors: Sequence[Sequence[int]], width: int) -> list[list[int]]:
    """A Z-basis of the integer forms h in Z^width with h . v = 0 for every v."""
    if not vectors:
        return [[int(i == j) for j in range(width)] for i in range(width)]
    return _exact.hermite_normal_form_with_transform([list(v) for v in vectors], width).kernel


class _Lattice:
    """ZA in coordinates: a basis H of ZA as the columns of a Hermite normal form."""

    def __init__(self, columns: Sequence[tuple[int, ...]]) -> None:
        d = len(columns[0])
        rows = [[c[i] for c in columns] for i in range(d)]
        form = _exact.hermite_normal_form_with_transform(rows, len(columns))
        self.hermite = form.hermite
        self.rank = form.rank
        self.vanishing = _kernel(columns, d)
        self.coordinates = []
        for column in columns:
            solved = _exact.solve_hermite_scaled(self.hermite, list(column))
            if solved is None or solved[1] != 1:
                raise ComputationError("a column of A has no integer coordinates in ZA")
            self.coordinates.append(solved[0])

    def coordinates_of(self, vector: Sequence[Fraction]) -> list[Fraction]:
        """The coordinates in the basis H of a vector in the span of A."""
        scale = math.lcm(*(x.denominator for x in vector))
        solved = _exact.solve_hermite(self.hermite, [int(x * scale) for x in vector])
        if solved is None:
            raise ComputationError("beta is not in the span of A where it should be")
        return [x / scale for x in solved]


def _decorate(
    columns: Sequence[tuple[int, ...]],
    beta0: Sequence[Fraction],
    beta1: Sequence[Fraction],
    faces: Sequence[tuple[int, tuple[int, ...]]],
) -> DecoratedConfiguration:
    """Decorate the faces of a homogeneous configuration, given as (dimension, points) with the
    faces of conv(columns) of dimension 0 and more."""
    lattice = _Lattice(columns)
    r = lattice.rank
    span = _ALL
    for h in lattice.vanishing:
        span = _restrict(_zero([_dot(h, beta0)], [_dot(h, beta1)]), span)
    if span.kind == "never":
        c0: list[Fraction] | None = None
        c1: list[Fraction] = []
    elif span.kind == "all":
        c0, c1 = lattice.coordinates_of(beta0), lattice.coordinates_of(beta1)
    else:
        assert span.offset is not None
        point = [a + span.offset * b for a, b in zip(beta0, beta1, strict=True)]
        c0, c1 = lattice.coordinates_of(point), [Fraction(0)] * r

    top = max(k for k, _ in faces)
    ordered = sorted([(-1, ()), *faces], key=lambda face: (face[0], face[1]))
    bases = {points: _kernel([lattice.coordinates[j] for j in points], r) for _, points in ordered}
    facets = [points for k, points in ordered if k == top - 1]
    forms = []
    for points in facets:
        (h,) = bases[points]
        outside = next(j for j in range(len(columns)) if j not in points)
        forms.append(h if _exact.dot(h, lattice.coordinates[outside]) > 0 else [-x for x in h])
    members = [frozenset(points) for points in facets]

    decorated = []
    for k, points in ordered:
        basis = bases[points]
        through = tuple(i for i, on in enumerate(members) if on >= frozenset(points))
        invariants = _exact.smith_invariants([forms[i] for i in through], r) if through else []
        if len(invariants) != len(basis):
            raise ComputationError("the facet forms through a face do not have full rank")
        if c0 is None:
            resonant = admissible = _NEVER
        else:
            p = [_dot(h, c0) for h in basis]
            q = [_dot(h, c1) for h in basis]
            admissible = _restrict(_zero(p, q), span)
            resonant = _canonical(_restrict(_integral(p, q), span), admissible)
        off = len(columns) - len(points)
        decorated.append(
            DecoratedFace(
                point_indices=points,
                dimension=k,
                codimension=top - k,
                facets=through,
                resonant=resonant,
                admissible=admissible,
                lattice_defect=math.prod(invariants),
                columns_off=off,
                pyramid=off + (r - len(basis)) == r,
            )
        )
    positions = {face.point_indices: i for i, face in enumerate(decorated)}
    return DecoratedConfiguration(
        columns=tuple(columns),
        rank=r,
        span=span,
        faces=tuple(decorated),
        facets=tuple(positions[points] for points in facets),
    )


def _vector(values: Sequence[object], d: int, what: str) -> list[Fraction]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValidationError(f"{what} must be a sequence of rational numbers")
    if len(values) != d:
        raise ValidationError(f"{what} needs one entry per row of A, {d}; got {len(values)}")
    return [_rational(x, what) for x in values]


def decorate_configuration(
    a_matrix: object,
    beta0: Sequence[int | Fraction],
    beta1: Sequence[int | Fraction] | None = None,
) -> DecoratedConfiguration:
    """Decorate every face of a homogeneous configuration for beta(eps) = beta0 + eps beta1.

    The faces are those of the columns of A with polytope_data, and the empty
    face. Some linear form must take the value 1 on every column, as for every
    GKZ system of a Feynman integral.

    Parameters
    ----------
    a_matrix
        The integer matrix A, a matrix or a sequence of rows, with distinct
        columns.
    beta0, beta1
        One integer or Fraction per row of A; beta1 is zero by default, for a
        parameter that does not depend on eps.

    Raises
    ------
    ValidationError
        If A is not an integer matrix with distinct columns, is not
        homogeneous, or beta0 or beta1 does not hold one exact rational number
        per row.
    """
    rows = _rows(a_matrix)
    d = len(rows)
    columns = [tuple(row[j] for row in rows) for j in range(len(rows[0]))]
    _distinct(columns)
    b0 = _vector(beta0, d, "beta0")
    b1 = [Fraction(0)] * d if beta1 is None else _vector(beta1, d, "beta1")
    data = polytope_data(columns)
    if not any(row[0] for row in data.affine_hull):
        raise ValidationError("A is not homogeneous: no linear form is 1 on every column")
    return _decorate(columns, b0, b1, data.faces)


# --- the face lattice of a Feynman integral ------------------------------------------------


@dataclass(frozen=True)
class SchwingerCheck:
    """The Cayley configuration of the Schwinger representation against the Lee-Pomeransky one.

    Attributes
    ----------
    bijective
        Whether T = lp_to_cayley(N, L) maps the Lee-Pomeransky columns
        (1, alpha_j) one to one onto the columns of the Cayley matrix.
    same_faces
        Whether that map carries the faces of the Lee-Pomeransky configuration
        onto those of the Cayley one, computed from its own columns.
    mismatches
        The faces, by their Lee-Pomeransky point indices, whose dimension,
        resonant or admissible set, lattice defect, columns off or pyramid
        differ from those of their image.
    faces
        The number of faces compared; 0 when the columns do not correspond.
    """

    bijective: bool
    same_faces: bool
    mismatches: tuple[tuple[int, ...], ...]
    faces: int

    @property
    def agrees(self) -> bool:
        """Whether the two configurations agree face by face."""
        return self.bijective and self.same_faces and not self.mismatches


def _same(one: DecoratedFace, other: DecoratedFace) -> bool:
    return (
        one.dimension == other.dimension
        and one.codimension == other.codimension
        and one.resonant == other.resonant
        and one.admissible == other.admissible
        and one.lattice_defect == other.lattice_defect
        and one.columns_off == other.columns_off
        and one.pyramid == other.pyramid
    )


def _apply(t: Sequence[Sequence[int]], vector: Sequence[int | Fraction]) -> list[Fraction]:
    return [
        sum((Fraction(a) * b for a, b in zip(row, vector, strict=True)), Fraction(0)) for row in t
    ]


@dataclass(frozen=True)
class DecoratedFaceLattice(DecoratedConfiguration):
    """Every face of the Newton polytope of a Feynman integral, decorated.

    The configuration is the Lee-Pomeransky one, with columns (1, alpha_j)
    for the points alpha_j of ``FeynmanIntegral.newton_polytope``, in their
    order, and beta = (-D/2, -nu_1, ..., -nu_N) with D = D_0 - 2 eps and
    integer powers. A face's point indices are indices of those points. A is
    of full rank exactly when P is full-dimensional.

    Attributes
    ----------
    data
        The polytope data of P.
    d0, d0_source
        D_0, and where it comes from: "given", "dimension" or "default"; see
        :func:`feynkit.resonance.choose_d0`.
    nu
        The power of each internal edge, in edge order.
    facet_resonance
        The facets as :func:`feynkit.resonance.classify_facets` classifies
        them, in the order of ``facets``, with their forms l_F(beta).
    identify_codimension
        The largest codimension whose faces carry an identification; None for
        every face.
    gkz_columns
        For each point j, the column of ``FeynmanIntegral.gkz.a_matrix`` that
        is (1, alpha_j); the GKZ system numbers its columns differently.
    cayley_columns
        For each point j, the column of ``FeynmanIntegral.schwinger_gkz.a_matrix``
        that is T (1, alpha_j), T = lp_to_cayley(N, L).
    lp_to_cayley
        T, as rows of integers.
    cayley_matrix
        The Cayley matrix of the Schwinger representation, as rows of
        integers.
    """

    data: PolytopeData
    d0: Fraction
    d0_source: D0Source
    nu: tuple[int, ...]
    facet_resonance: tuple[FacetResonance, ...]
    identify_codimension: int | None
    gkz_columns: tuple[int, ...]
    cayley_columns: tuple[int, ...]
    lp_to_cayley: tuple[tuple[int, ...], ...]
    cayley_matrix: tuple[tuple[int, ...], ...]

    def beta(self) -> tuple[list[Fraction], list[Fraction]]:
        """(beta_0, beta_1) with beta = beta_0 + eps beta_1 = (-D/2, -nu) at D = D_0 - 2 eps."""
        beta0 = [-self.d0 / 2, *(Fraction(-x) for x in self.nu)]
        beta1 = [Fraction(1), *(Fraction(0) for _ in self.nu)]
        return beta0, beta1

    def check_schwinger(self) -> SchwingerCheck:
        """Decorate the Cayley configuration from its own columns and compare face by face.

        T maps the Lee-Pomeransky columns onto the Cayley ones and beta_LP to
        the Cayley parameter; being unimodular, it carries ZA, the spans of
        the faces and the forms vanishing on them across, so the two sides
        should agree exactly. The check does not rely on this: it computes the
        faces of the Cayley columns with polytope_data, their decorations for
        T beta_0 + eps T beta_1, and compares them with the faces here under
        the column map.
        """
        rows = self.cayley_matrix
        cayley = [tuple(row[j] for row in rows) for j in range(len(rows[0]))]
        where = {column: j for j, column in enumerate(cayley)}
        images = [tuple(int(x) for x in _apply(self.lp_to_cayley, c)) for c in self.columns]
        found = [where.get(image) for image in images]
        sigma = [j for j in found if j is not None]
        if len(set(sigma)) != len(found) or len(found) != len(cayley):
            return SchwingerCheck(bijective=False, same_faces=False, mismatches=(), faces=0)
        beta0, beta1 = self.beta()
        other = decorate_configuration(
            [list(row) for row in rows],
            _apply(self.lp_to_cayley, beta0),
            _apply(self.lp_to_cayley, beta1),
        )
        theirs = {face.point_indices: face for face in other.faces}
        mapped = {tuple(sorted(sigma[j] for j in face.point_indices)): face for face in self.faces}
        mismatches = tuple(
            face.point_indices
            for key, face in mapped.items()
            if key in theirs and not _same(face, theirs[key])
        )
        return SchwingerCheck(
            bijective=True,
            same_faces=set(mapped) == set(theirs),
            mismatches=mismatches,
            faces=len(mapped),
        )


def _powers(fi: FeynmanIntegral, nu: Mapping[int, int] | None) -> tuple[int, ...]:
    """The integer power of each internal edge: the integral's, or those of nu."""
    edges = fi.graph.get_internal_edges()
    source = fi.propagator_exponents if nu is None else dict(nu)
    if set(source) != {e.idx for e in edges}:
        raise ValidationError("nu needs one power for each internal edge, keyed by its index")
    try:
        return tuple(_exact._as_int(source[e.idx], "the powers") for e in edges)
    except ValidationError:
        raise ValidationError(
            "the face lattice needs integer powers; the exponents of the integral are not all "
            "integers, so pass nu, as in nu={1: 1, 2: 1}"
        ) from None


def _through_origin(points: Sequence[tuple[int, ...]]) -> bool:
    if not points:
        return False
    origin = (0,) * len(points[0])
    return _exact.affine_rank([*points, origin]) == _exact.affine_rank(points)


def _column_map(columns: Sequence[tuple[int, ...]], matrix: object, what: str) -> tuple[int, ...]:
    rows = _rows(matrix)
    where = {tuple(row[j] for row in rows): j for j in range(len(rows[0]))}
    found = tuple(where.get(column, -1) for column in columns)
    if -1 in found or len(set(found)) != len(where) or len(where) != len(columns):
        raise ComputationError(f"the columns of {what} do not correspond to the Newton polytope")
    return found


def decorate_faces(
    fi: FeynmanIntegral,
    d0: int | Fraction | None = None,
    *,
    nu: Mapping[int, int] | None = None,
    identify_codimension: int | None = 2,
) -> DecoratedFaceLattice:
    """Decorate every face of the Newton polytope of a Feynman integral.

    Parameters
    ----------
    fi
        The integral.
    d0
        D_0, an integer or a Fraction, with D = D_0 - 2 eps. None, the
        default, reads it from the dimension of the integral when that is
        D_0 - 2 eps with D_0 a number and eps the symbol named epsilon, and
        takes 4 otherwise; see :func:`feynkit.resonance.choose_d0`.
    nu
        The power of each internal edge, a mapping from edge index to
        integer; by default the propagator exponents of the integral, which
        must then be integers.
    identify_codimension
        The faces of codimension up to this carry their identification from
        ``fi.face_identification``; None for every face.

    Raises
    ------
    ValidationError
        If the powers are not one integer per internal edge, d0 is not an
        integer or a Fraction, or identify_codimension is not None or a
        non-negative integer.
    """
    from .face_identification import check_codimension

    check_codimension(identify_codimension)
    d0_value, source = choose_d0(d0, fi.dimension)
    powers = _powers(fi, nu)
    points = [tuple(int(x) for x in p) for p in fi.newton_polytope.points]
    data = polytope_data(points)
    columns = [(1, *p) for p in points]
    n = len(powers)
    beta0 = [-d0_value / 2, *(Fraction(-x) for x in powers)]
    beta1 = [Fraction(1), *(Fraction(0) for _ in powers)]
    base = _decorate(columns, beta0, beta1, data.faces)
    identified = {face.point_indices: face for face in fi.face_identification(identify_codimension)}
    faces = tuple(
        dataclasses.replace(
            face,
            through_origin=_through_origin([points[j] for j in face.point_indices]),
            identification=identified.get(face.point_indices),
        )
        for face in base.faces
    )
    t = lp_to_cayley(n, fi.loop_count)
    rows_t = tuple(tuple(int(t[r, k]) for k in range(n + 1)) for r in range(n + 1))
    cayley = _rows(fi.schwinger_gkz.a_matrix)
    images = [tuple(int(x) for x in _apply(rows_t, c)) for c in columns]
    return DecoratedFaceLattice(
        columns=base.columns,
        rank=base.rank,
        span=base.span,
        faces=faces,
        facets=base.facets,
        data=data,
        d0=d0_value,
        d0_source=source,
        nu=powers,
        facet_resonance=classify_facets(data, powers, d0_value),
        identify_codimension=identify_codimension,
        gkz_columns=_column_map(columns, fi.gkz.a_matrix, "the GKZ matrix"),
        cayley_columns=_column_map(images, cayley, "the Cayley matrix"),
        lp_to_cayley=rows_t,
        cayley_matrix=tuple(tuple(row) for row in cayley),
    )
