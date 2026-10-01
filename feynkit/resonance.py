"""
Resonance and admissibility of the facets of a GKZ configuration.

A face F of the configuration A is resonant for beta when beta lies in
span_C(F) + ZA, and, in feynkit's term, admissible when beta lies in span_C(F)
itself; the paper notes that the face system (A_F, beta) is then a true
subsystem, its solutions solving the full system (Britto, Grimm and
Hoefnagels, arXiv:2606.09978, eqs. 19-23, pp. 10-12). For a facet with the linear functional l_F, zero on F, positive
off it and mapping ZA onto Z, resonance is l_F(beta) in Z and admissibility
l_F(beta) = 0 (their eq. 24 and appendix A, for ZA = Z^d; the design note
docs/design/2026-09-30-resonance.md proves both for any ZA of full rank).

The powers nu_e are integers and D = D_0 - 2 eps. For the Lee-Pomeransky
parameter beta = (-D/2, -nu_1, ..., -nu_N) and a facet m . x <= b of the
Newton polytope with lattice index g_F,

    l_F(beta) = (c_F + b eps) / g_F,    c_F = m . nu - b D_0/2,

so the facet is resonant for every eps (b = 0 and g_F divides m . nu), for
none (b = 0 otherwise), or on the progression eps_F + (g_F/|b|) Z with
eps_F = D_0/2 - m . nu / b, the one value at which it is admissible. Below
full dimension beta must first lie in the span of A, which fixes eps or
rules it out.

A resonant facet over which A is not a pyramid makes M_A(beta) reducible:
it contains a resonance centre over which A is not a pyramid either, and
Theorem 4.1 of Schulze and Walther (arXiv:1009.3569) applies to that centre.
Everything is exact: integers, Fractions and, in :func:`admissible`, SymPy.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, get_args

import sympy as sp

from . import _exact
from .core.constants import DEFAULT_EPSILON
from .core.exceptions import ValidationError
from .polytope import Facet, PolytopeData, polytope_data

__all__ = [
    "EpsilonKind",
    "EpsilonSet",
    "FacetResonance",
    "ParameterForm",
    "D0Source",
    "admissible",
    "choose_d0",
    "classify_configuration",
    "classify_facets",
    "lee_pomeransky_beta",
    "span_epsilons",
]

EpsilonKind = Literal["all", "progression", "point", "never"]
D0Source = Literal["given", "dimension", "default"]


def _rational(value: object, where: str) -> Fraction:
    """value as a Fraction; integers and exact rationals only, never a float."""
    if isinstance(value, numbers.Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    raise ValidationError(f"{where} must be an integer or a Fraction; got {value!r}")


# --- value types -----------------------------------------------------------------


@dataclass(frozen=True)
class EpsilonSet:
    """A set of values of eps: every complex number, a progression, one value, or none.

    Attributes
    ----------
    kind
        "all", "progression" (offset + period Z), "point" (the one value
        offset) or "never".
    offset
        The offset of a progression or the value of a point; None otherwise.
        For a facet's resonant progression it is eps_F, where the facet is
        admissible, not reduced modulo the period.
    period
        The positive period of a progression; None otherwise.

    Raises
    ------
    ValidationError
        If kind is unknown, or offset and period do not match it.
    """

    kind: EpsilonKind
    offset: Fraction | None = None
    period: Fraction | None = None

    def __post_init__(self) -> None:
        if self.kind not in get_args(EpsilonKind):
            raise ValidationError(f"unknown kind of eps set {self.kind!r}")
        wants_offset = self.kind in ("progression", "point")
        wants_period = self.kind == "progression"
        if (self.offset is not None) != wants_offset or (self.period is not None) != wants_period:
            raise ValidationError(
                f"an eps set of kind {self.kind!r} has the wrong offset or period"
            )
        if self.offset is not None:
            object.__setattr__(self, "offset", _rational(self.offset, "the offset"))
        if self.period is not None:
            period = _rational(self.period, "the period")
            if period <= 0:
                raise ValidationError(f"the period must be positive; got {period}")
            object.__setattr__(self, "period", period)

    def __contains__(self, epsilon: object) -> bool:
        """Whether the exact rational eps lies in the set."""
        eps = _rational(epsilon, "eps")
        if self.kind in ("all", "never"):
            return self.kind == "all"
        assert self.offset is not None
        if self.kind == "point":
            return eps == self.offset
        assert self.period is not None
        return ((eps - self.offset) / self.period).denominator == 1

    def window(self, low: int | Fraction, high: int | Fraction) -> tuple[Fraction, ...]:
        """The values with low <= eps <= high, in increasing order.

        Raises
        ------
        ValidationError
            If the set is "all", which no list holds, or low or high is not an
            integer or a Fraction.
        """
        lo, hi = _rational(low, "low"), _rational(high, "high")
        if self.kind == "all":
            raise ValidationError("every eps lies in the set; there is no list to give")
        if self.kind == "never":
            return ()
        assert self.offset is not None
        if self.kind == "point":
            return (self.offset,) if lo <= self.offset <= hi else ()
        assert self.period is not None
        first = math.ceil((lo - self.offset) / self.period)
        last = math.floor((hi - self.offset) / self.period)
        return tuple(self.offset + k * self.period for k in range(first, last + 1))


_ALL = EpsilonSet("all")
_NEVER = EpsilonSet("never")


@dataclass(frozen=True)
class ParameterForm:
    """The linear form dimension * D + sum_e nu[e] nu_e in the dimension and the powers.

    beta depends linearly on D and the nu_e, without a constant term, so
    l_F(beta) is such a form for every facet, whatever the values. At
    D = D_0 - 2 eps it is c + s eps, with c = dimension * D_0 + sum_e nu[e] nu_e
    and s = -2 * dimension.

    Attributes
    ----------
    dimension
        The coefficient of D.
    nu
        The coefficient of each power nu_e, in edge order.
    """

    dimension: Fraction
    nu: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _rational(self.dimension, "the coefficient of D"))
        object.__setattr__(self, "nu", tuple(_rational(x, "a coefficient of nu") for x in self.nu))

    def __add__(self, other: ParameterForm) -> ParameterForm:
        if len(self.nu) != len(other.nu):
            raise ValidationError("the forms have different numbers of powers")
        return ParameterForm(
            self.dimension + other.dimension,
            tuple(a + b for a, b in zip(self.nu, other.nu, strict=True)),
        )

    def __rmul__(self, factor: int | Fraction) -> ParameterForm:
        k = _rational(factor, "the factor")
        return ParameterForm(k * self.dimension, tuple(k * x for x in self.nu))

    def at(self, nu: Sequence[int], d0: int | Fraction) -> tuple[Fraction, Fraction]:
        """(c, s), with the form equal to c + s eps at these powers and D = d0 - 2 eps."""
        if len(nu) != len(self.nu):
            raise ValidationError(f"expected {len(self.nu)} powers; got {len(nu)}")
        constant = self.dimension * _rational(d0, "D_0") + sum(
            (a * b for a, b in zip(self.nu, nu, strict=True)), Fraction(0)
        )
        return constant, -2 * self.dimension

    def expression(self, dimension: sp.Expr, exponents: Sequence[sp.Expr]) -> sp.Expr:
        """The form as a SymPy expression in the given dimension and exponents."""
        if len(exponents) != len(self.nu):
            raise ValidationError(f"expected {len(self.nu)} exponents; got {len(exponents)}")
        terms = [_sympy(self.dimension) * dimension]
        terms += [_sympy(a) * e for a, e in zip(self.nu, exponents, strict=True)]
        return sp.Add(*terms)


def _sympy(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


@dataclass(frozen=True)
class FacetResonance:
    """How one facet F of A behaves as D = D_0 - 2 eps varies, the powers fixed.

    Attributes
    ----------
    facet
        The facet as polytope_data gives it; its point_indices are the columns
        of A on F. For :func:`classify_facets` its inequality is m . x <= b on
        the Newton polytope, and b and m enter c_F and eps_F.
    functional
        l_F on the rows of A: zero on F, positive on the other columns, and
        mapping ZA onto Z. For :func:`classify_facets` it is
        ``facet.lattice_form``. When A does not have full rank it is fixed
        only modulo the forms vanishing on A, as is ``form``.
    form
        l_F(beta), a linear form in D and the powers.
    resonant
        The eps at which F is resonant, beta in span_C(F) + ZA.
    admissible
        The eps at which beta lies in span_C(F): "all", "point" or "never".
        For a progression in ``resonant`` it is the point at its offset.
    columns_off
        The number of columns of A off F. A is a pyramid over F exactly when
        it is 1 (Schulze and Walther, Def. 3.4).
    reducible
        True when M_A(beta) has reducible monodromy at every eps in
        ``resonant``, which holds when F is resonant somewhere, A has full rank
        and A is not a pyramid over F (Schulze and Walther, Thm 4.1). None when
        this facet does not decide it; another face may.
    """

    facet: Facet
    functional: tuple[Fraction, ...]
    form: ParameterForm
    resonant: EpsilonSet
    admissible: EpsilonSet
    columns_off: int
    reducible: bool | None

    @property
    def kind(self) -> EpsilonKind:
        """The kind of ``resonant``: "all", "progression", "point" or "never"."""
        return self.resonant.kind

    @property
    def lattice_index(self) -> int:
        """g_F, as the facet carries it."""
        return self.facet.lattice_index

    @property
    def resonant_at_zero(self) -> bool:
        """Whether F is resonant at eps = 0, that is at D = D_0."""
        return 0 in self.resonant

    @property
    def pyramid(self) -> bool:
        """Whether A is a pyramid over F: exactly one column lies off it."""
        return self.columns_off == 1


# --- sets of eps ---------------------------------------------------------------------


def _integral(constant: Fraction, slope: Fraction) -> EpsilonSet:
    """The eps with constant + slope eps an integer."""
    if slope == 0:
        return _ALL if constant.denominator == 1 else _NEVER
    return EpsilonSet("progression", -constant / slope, 1 / abs(slope))


def _zero(constant: Fraction, slope: Fraction) -> EpsilonSet:
    """The eps with constant + slope eps = 0."""
    if slope == 0:
        return _ALL if constant == 0 else _NEVER
    return EpsilonSet("point", -constant / slope)


def _restrict(found: EpsilonSet, span: EpsilonSet) -> EpsilonSet:
    """found intersected with span, which is "all", "point" or "never"."""
    if span.kind == "all":
        return found
    if span.kind == "never":
        return _NEVER
    assert span.offset is not None
    return span if span.offset in found else _NEVER


def _span(kernel: Sequence[ParameterForm], nu: Sequence[int], d0: Fraction) -> EpsilonSet:
    """The eps at which every form vanishing on A vanishes on beta."""
    span = _ALL
    for form in kernel:
        span = _restrict(_zero(*form.at(nu, d0)), span)
    return span


# --- inputs ------------------------------------------------------------------------


def _powers(nu: Sequence[int], n: int) -> tuple[int, ...]:
    if isinstance(nu, (str, bytes)) or not isinstance(nu, Sequence):
        raise ValidationError("the powers must be a sequence of integers")
    if len(nu) != n:
        raise ValidationError(f"expected {n} powers; got {len(nu)}")
    return tuple(_exact._as_int(x, "the powers") for x in nu)


def _rows(a_matrix: object) -> list[list[int]]:
    """The rows of A as lists of Python integers."""
    raw = a_matrix.tolist() if hasattr(a_matrix, "tolist") else a_matrix
    if not isinstance(raw, Sequence) or not raw:
        raise ValidationError("A must be a non-empty matrix of integers")
    rows = [[_exact._as_int(x, "A") for x in row] for row in raw]
    if any(len(row) != len(rows[0]) for row in rows) or not rows[0]:
        raise ValidationError("the rows of A must be non-empty and of one length")
    return rows


def lee_pomeransky_beta(n: int) -> tuple[ParameterForm, ...]:
    """beta = (-D/2, -nu_1, ..., -nu_n), the parameter of the Lee-Pomeransky GKZ system."""
    zero = (Fraction(0),) * n
    head = ParameterForm(Fraction(-1, 2), zero)
    return (
        head,
        *(ParameterForm(Fraction(0), zero[:e] + (Fraction(-1),) + zero[e + 1 :]) for e in range(n)),
    )


def _combine(functional: Sequence[Fraction], beta: Sequence[ParameterForm]) -> ParameterForm:
    total = ParameterForm(Fraction(0), (Fraction(0),) * len(beta[0].nu))
    for coefficient, form in zip(functional, beta, strict=True):
        if coefficient:
            total = total + coefficient * form
    return total


def _classify(
    facets: Sequence[tuple[Facet, tuple[Fraction, ...]]],
    kernel: Sequence[Sequence[Fraction]],
    beta: Sequence[ParameterForm],
    n_columns: int,
    nu: Sequence[int],
    d0: Fraction,
) -> tuple[FacetResonance, ...]:
    span = _span([_combine(k, beta) for k in kernel], nu, d0)
    full_rank = not kernel
    records = []
    for facet, functional in facets:
        form = _combine(functional, beta)
        constant, slope = form.at(nu, d0)
        resonant = _restrict(_integral(constant, slope), span)
        columns_off = n_columns - len(facet.point_indices)
        decided = full_rank and columns_off > 1 and resonant.kind != "never"
        records.append(
            FacetResonance(
                facet=facet,
                functional=functional,
                form=form,
                resonant=resonant,
                admissible=_restrict(_zero(constant, slope), span),
                columns_off=columns_off,
                reducible=True if decided else None,
            )
        )
    return tuple(records)


def _distinct(points: Sequence[Sequence[int]]) -> None:
    if len({tuple(p) for p in points}) != len(points):
        raise ValidationError("the configuration has repeated columns")


# --- public API --------------------------------------------------------------


def choose_d0(d0: int | Fraction | None, dimension: object) -> tuple[Fraction, D0Source]:
    """D_0 and where it comes from: given, read from the dimension, or the default 4.

    With d0 None, D_0 is read from ``dimension`` when that is D_0 - 2 eps, with
    D_0 a rational number and eps the regulator, the symbol named ``epsilon``
    whatever its assumptions, as in the prefactors of the parametric
    representations. Any other dimension, a symbol D or a number included,
    gives the default 4.

    Raises
    ------
    ValidationError
        If d0 is given and is not an integer or a Fraction.
    """
    if d0 is not None:
        return _rational(d0, "D_0"), "given"
    expr = sp.sympify(dimension)
    symbols = expr.free_symbols
    if len(symbols) == 1:
        (eps,) = symbols
        if getattr(eps, "name", None) == DEFAULT_EPSILON:
            head = sp.expand(expr.subs(eps, 0))
            if head.is_Rational and sp.expand(expr - head + 2 * eps) == 0:
                return Fraction(int(head.p), int(head.q)), "dimension"
    return Fraction(4), "default"


def classify_facets(
    data: PolytopeData, nu: Sequence[int], d0: int | Fraction = 4
) -> tuple[FacetResonance, ...]:
    """Classify the facets of a Newton polytope for the Lee-Pomeransky parameter.

    The configuration is A = (1; alpha_j), the homogenised points of data, and
    beta = (-D/2, -nu_1, ..., -nu_N) with D = d0 - 2 eps. The facets are
    data.relative_facets, in their order, so the facets already computed are
    used; below full dimension their resonance and admissibility hold only
    where beta lies in the span of A (see :func:`span_epsilons`).

    Parameters
    ----------
    data
        The polytope, from polytope_data, with distinct points.
    nu
        The integer power of each coordinate, in order.
    d0
        D_0, an integer or a Fraction; 4 by default.

    Raises
    ------
    ValidationError
        If nu does not hold one integer per coordinate, d0 is not an integer
        or a Fraction, or the points repeat.
    """
    n = data.ambient_dimension
    powers = _powers(nu, n)
    d = _rational(d0, "D_0")
    _distinct(data.points)
    kernel = [tuple(Fraction(x) for x in row) for row in data.affine_hull]
    facets = [(f, f.lattice_form) for f in data.relative_facets]
    return _classify(facets, kernel, lee_pomeransky_beta(n), len(data.points), powers, d)


def span_epsilons(data: PolytopeData, nu: Sequence[int], d0: int | Fraction = 4) -> EpsilonSet:
    """The eps at which the Lee-Pomeransky beta lies in the span of A = (1; alpha_j).

    Each equation h_0 + h . x = 0 of the affine hull asks h_0 D/2 + h . nu = 0.
    Off this set the Euler equations are inconsistent and the GKZ system has
    no non-zero solutions. It is "all" when data is full-dimensional.

    Raises
    ------
    ValidationError
        As :func:`classify_facets`.
    """
    n = data.ambient_dimension
    powers = _powers(nu, n)
    beta = lee_pomeransky_beta(n)
    kernel = [_combine([Fraction(x) for x in row], beta) for row in data.affine_hull]
    return _span(kernel, powers, _rational(d0, "D_0"))


def classify_configuration(
    a_matrix: object,
    beta: Sequence[ParameterForm],
    nu: Sequence[int],
    d0: int | Fraction = 4,
) -> tuple[FacetResonance, ...]:
    """Classify the facets of any homogeneous configuration A for a parameter linear in D and nu.

    The facets are computed from the columns of A, a matrix or a sequence of
    rows, with polytope_data; their point_indices index the columns. Some
    linear form must take the value 1 on every column, as for every GKZ
    system of a Feynman integral.

    Parameters
    ----------
    a_matrix
        The integer matrix A, with distinct columns.
    beta
        One ParameterForm per row of A, all with the same number of powers.
    nu, d0
        As for :func:`classify_facets`.

    Raises
    ------
    ValidationError
        If A is not an integer matrix with distinct columns, is not
        homogeneous, or beta does not have one form per row; or as
        :func:`classify_facets`.
    """
    rows = _rows(a_matrix)
    if len(beta) != len(rows):
        raise ValidationError(f"expected one form per row of A, {len(rows)}; got {len(beta)}")
    powers = _powers(nu, len(beta[0].nu))
    d = _rational(d0, "D_0")
    columns = [tuple(row[j] for row in rows) for j in range(len(rows[0]))]
    _distinct(columns)
    data = polytope_data(columns)
    # Each row (h_0, h) of the affine hull has h . y = -h_0 on every column.
    pivot = next((row for row in data.affine_hull if row[0]), None)
    if pivot is None:
        raise ValidationError("A is not homogeneous: no linear form is 1 on every column")
    eta = [Fraction(x, -pivot[0]) for x in pivot[1:]]
    kernel = [
        tuple(Fraction(pivot[0] * r - row[0] * p) for r, p in zip(row[1:], pivot[1:], strict=True))
        for row in data.affine_hull
        if row is not pivot
    ]
    facets = []
    for facet in data.relative_facets:
        heights = [facet.offset - _exact.dot(facet.normal, column) for column in columns]
        g = math.gcd(*heights)
        functional = tuple(
            (facet.offset * e - m) / g for e, m in zip(eta, facet.normal, strict=True)
        )
        facets.append((facet, functional))
    return _classify(facets, kernel, beta, len(columns), powers, d)


def admissible(a_matrix: object, face: Iterable[int], beta: Sequence[object]) -> bool:
    """Whether beta lies in the complex span of the columns of A indexed by face.

    For a face F of A this makes (A_F, beta) a true subsystem, its solutions
    solving (A, beta) (Britto, Grimm and Hoefnagels, arXiv:2606.09978,
    pp. 10-11). That face is a face of A is not checked. The entries of beta
    may be numbers or SymPy expressions; with symbols, beta must lie in the
    span for all their values.

    Raises
    ------
    ValidationError
        If A is not an integer matrix, an index is out of range, or beta does
        not have one entry per row.
    """
    rows = _rows(a_matrix)
    d, n = len(rows), len(rows[0])
    if len(beta) != d:
        raise ValidationError(f"expected one entry of beta per row of A, {d}; got {len(beta)}")
    indices = [_exact._as_int(j, "the face") for j in face]
    if any(not 0 <= j < n for j in indices):
        raise ValidationError(f"a column index of the face is outside 0..{n - 1}")
    if indices:
        vanishing = _exact.hermite_normal_form_with_transform(
            [[rows[r][j] for r in range(d)] for j in indices], d
        ).kernel
    else:
        vanishing = [[int(r == k) for r in range(d)] for k in range(d)]
    entries = [sp.sympify(b) for b in beta]
    return all(
        sp.cancel(sp.Add(*(h * b for h, b in zip(form, entries, strict=True)))) == 0
        for form in vanishing
    )
