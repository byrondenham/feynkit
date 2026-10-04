"""
Integral families: a momentum routing, a complete basis of quadratic functions, and the
family's Lee-Pomeransky polynomial.

An integral family (R. N. Lee, arXiv:1310.1145, section 2, Eqs. 1-2) has L loop momenta
l_i, E independent external momenta p_k and a complete basis of
N = L(L + 1)/2 + L E functions

    D_alpha = a_alpha^{ij} l_i . l_j + 2 b_alpha^{ik} l_i . p_k + c_alpha,

complete meaning that every scalar product l_i . l_j and l_i . p_k is a unique affine
combination of them. Its integrals are J(n) = int prod_alpha D_alpha^(-n_alpha) for n in
Z^N. The functions that are not propagators of the graph are irreducible numerators
(ISPs) and occur only with n_alpha <= 0.

The family of a :class:`~feynkit.FeynmanIntegral` takes

- a momentum routing (:func:`momentum_routing`): a spanning tree chosen in edge-index
  order, whose chords carry l_1, ..., l_L from v1 to v2. The legs are incoming, legs
  1, ..., n - 1 carry the independent momenta and p_n = -(p_1 + ... + p_(n-1)); the
  momenta of the tree edges follow from momentum conservation;
- the functions in a fixed order: the propagators D_e = -q_e^2 + m_e^2, feynkit's sign,
  by edge index, then the ISPs, given or completed greedily from -(l_i + p_k)^2 and
  -(l_i - l_j)^2.

With z_alpha the parameter of D_alpha, sum_alpha z_alpha D_alpha = l^T A l + 2 l . B + C,
and completing the square gives

    U = (-1)^L det A,    F = (-1)^L (C det A - sum_ij adj(A)_ij B_i . B_j),

Lee's Eqs. 8-9 with the signs of a Minkowski -q^2. G = U + F is the Lee-Pomeransky
polynomial of the family. Its parameters are the u_e of the integral, then z_1, z_2, ...
for the ISPs, and F carries the factor 1/mu^2 of the energy scale, as feynkit's does, so
that G at z = 0 is the G of the graph.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import sympy as sp

from .core.constants import LEE_POMERANSKY_PARAM_PREFIX
from .core.exceptions import ValidationError
from .systems.gkz import construct_gkz_matrix_from_exponents
from .systems.monomial import extract_monomial_support
from .types import NewtonPolytope

if TYPE_CHECKING:
    from .integral import FeynmanIntegral

__all__ = [
    "FamilyFunction",
    "FamilyPolynomials",
    "FunctionKind",
    "IntegralFamily",
    "MomentumRouting",
    "momentum_routing",
]

FunctionKind = Literal["propagator", "isp"]

# The parameter of the k-th ISP is z_k.
ISP_PARAM_PREFIX = "z"

Matrix = tuple[tuple[Fraction, ...], ...]
Momentum = tuple[tuple[int, ...], tuple[int, ...]]


def _fraction(value: object, where: str) -> Fraction:
    """value as a Fraction; integers and exact rationals only, never a float or a bool."""
    if isinstance(value, numbers.Rational) and not isinstance(value, bool):
        return Fraction(int(value.numerator), int(value.denominator))
    raise ValidationError(f"{where} must be an integer or a Fraction; got {value!r}")


def _matrix(rows: object, where: str) -> Matrix:
    if not isinstance(rows, Sequence) or any(not isinstance(row, Sequence) for row in rows):
        raise ValidationError(f"{where} must be a sequence of rows")
    return tuple(tuple(_fraction(x, f"an entry of {where}") for x in row) for row in rows)


def _symmetric(matrix: Matrix, size: int, where: str) -> None:
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValidationError(f"{where} must be {size} x {size}")
    if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(i)):
        raise ValidationError(f"{where} must be symmetric")


def _rational(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


def _momentum_label(loop: Sequence[Fraction], external: Sequence[Fraction]) -> str:
    """The momentum as l1 - l2 + p1."""
    text = ""
    names = [f"l{i}" for i in range(1, len(loop) + 1)]
    names += [f"p{k}" for k in range(1, len(external) + 1)]
    for c, name in zip((*loop, *external), names, strict=True):
        if not c:
            continue
        size = abs(c)
        term = name if size == 1 else f"{size} {name}"
        if not text:
            text = term if c > 0 else f"-{term}"
        else:
            text += f" + {term}" if c > 0 else f" - {term}"
    return text or "0"


@dataclass(frozen=True)
class FamilyFunction:
    """
    One function of an integral family, D = a^{ij} l_i . l_j + 2 b^{ik} l_i . p_k + c.

    The constant is split into the external products and the rest:
    c = e^{km} p_k . p_m + constant, so that a function is defined before the kinematics
    are; :meth:`offset` gives c from the products p_k . p_m.

    Attributes
    ----------
    label
        A name for the function, such as ``-(l1 + p1)^2``.
    quadratic
        a, the symmetric L x L matrix of the coefficients of l_i . l_j.
    linear
        b, the L x E matrix whose entry b^{ik} is half the coefficient of l_i . p_k.
    external
        e, the symmetric E x E matrix of the coefficients of p_k . p_m.
    constant
        The rest of c, such as a squared mass.
    kind
        "propagator" for a propagator of the graph, "isp" for an irreducible numerator.
    edge
        The index of the edge of a propagator; None for an ISP.

    Raises
    ------
    ValidationError
        If the matrices are not exact rationals of consistent shapes with L >= 1, a or e is
        not symmetric, the constant holds a float, the kind is unknown, or a propagator
        has no edge or an ISP has one.
    """

    label: str
    quadratic: Matrix
    linear: Matrix
    external: Matrix
    constant: sp.Expr = sp.Integer(0)
    kind: FunctionKind = "isp"
    edge: int | None = None

    def __post_init__(self) -> None:
        quadratic = _matrix(self.quadratic, "the quadratic part")
        linear = _matrix(self.linear, "the linear part")
        external = _matrix(self.external, "the external part")
        loops, externals = len(quadratic), len(external)
        if loops < 1:
            raise ValidationError("a family function needs at least one loop momentum")
        _symmetric(quadratic, loops, "the quadratic part")
        _symmetric(external, externals, "the external part")
        if len(linear) != loops or any(len(row) != externals for row in linear):
            raise ValidationError(f"the linear part must be {loops} x {externals}")
        constant = sp.sympify(self.constant)
        if not isinstance(constant, sp.Expr) or constant.has(sp.Float):
            raise ValidationError(f"the constant must be exact; got {self.constant!r}")
        if self.kind not in ("propagator", "isp"):
            raise ValidationError(f"the kind must be 'propagator' or 'isp'; got {self.kind!r}")
        if (self.kind == "propagator") != (self.edge is not None):
            raise ValidationError("a propagator has the index of its edge and an ISP has none")
        object.__setattr__(self, "quadratic", quadratic)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "external", external)
        object.__setattr__(self, "constant", constant)

    @property
    def loops(self) -> int:
        """L, the number of loop momenta."""
        return len(self.quadratic)

    @property
    def externals(self) -> int:
        """E, the number of independent external momenta."""
        return len(self.external)

    @property
    def form(self) -> tuple[Matrix, Matrix, Matrix, sp.Expr]:
        """The quadratic form (a, b, e, constant), without the label, kind and edge."""
        return self.quadratic, self.linear, self.external, self.constant

    def offset(self, products: Sequence[Sequence[sp.Expr]]) -> sp.Expr:
        """c = e^{km} p_k . p_m + constant, for the E x E matrix of the products p_k . p_m."""
        total = self.constant
        for k, row in enumerate(self.external):
            for m, e in enumerate(row):
                if e:
                    total += _rational(e) * sp.sympify(products[k][m])
        return sp.expand(total)

    def coefficients(self) -> tuple[Fraction, ...]:
        """
        The coefficients of the scalar products in D.

        They come in the order l_i . l_j for i <= j, row by row, then l_i . p_k: a^{ii},
        2 a^{ij} for i < j, and 2 b^{ik}.
        """
        a, b = self.quadratic, self.linear
        loops = len(a)
        row = [a[i][j] if i == j else 2 * a[i][j] for i in range(loops) for j in range(i, loops)]
        row += [2 * x for line in b for x in line]
        return tuple(row)

    @classmethod
    def squared(
        cls,
        loop: Sequence[int | Fraction],
        external: Sequence[int | Fraction],
        mass_squared: object = 0,
        *,
        label: str | None = None,
        kind: FunctionKind = "isp",
        edge: int | None = None,
    ) -> FamilyFunction:
        """
        -q^2 + m^2 for q = sum_i loop[i] l_i + sum_k external[k] p_k, in feynkit's sign.

        The label is by default ``-(q)^2``, with ``+ m^2`` when the mass is not zero.
        """
        lam = [_fraction(x, "a loop coefficient") for x in loop]
        kap = [_fraction(x, "an external coefficient") for x in external]
        mass = sp.sympify(mass_squared)
        if label is None:
            label = f"-({_momentum_label(lam, kap)})^2"
            if mass != 0:
                label += f" + {mass}"
        return cls(
            label=label,
            quadratic=tuple(tuple(-x * y for y in lam) for x in lam),
            linear=tuple(tuple(-x * y for y in kap) for x in lam),
            external=tuple(tuple(-x * y for y in kap) for x in kap),
            constant=mass,
            kind=kind,
            edge=edge,
        )

    @classmethod
    def product(
        cls, i: int, k: int, *, loops: int, externals: int, label: str | None = None
    ) -> FamilyFunction:
        """The scalar product l_i . p_k, an ISP, among L = loops and E = externals momenta."""
        if not (1 <= i <= loops and 1 <= k <= externals):
            raise ValidationError(
                f"l_{i} . p_{k} needs 1 <= i <= {loops} and 1 <= k <= {externals}"
            )
        zero = Fraction(0)
        linear = tuple(
            tuple(Fraction(1, 2) if (r, s) == (i - 1, k - 1) else zero for s in range(externals))
            for r in range(loops)
        )
        return cls(
            label=label if label is not None else f"l{i}.p{k}",
            quadratic=((zero,) * loops,) * loops,
            linear=linear,
            external=((zero,) * externals,) * externals,
        )


@dataclass(frozen=True)
class MomentumRouting:
    """
    The momentum of every propagator in terms of the loop and independent external momenta.

    Attributes
    ----------
    tree
        The edges of the spanning tree, by index.
    chords
        The other edges; chords[i] carries l_(i+1) from its v1 to its v2.
    edge_momenta
        For each internal edge in index order, (edge, loop, external): its momentum from v1
        to v2 is sum_i loop[i] l_(i+1) + sum_k external[k] p_(k+1), with integer
        coefficients.
    externals
        E, the number of independent external momenta: n - 1 for n >= 1 legs, else 0.
    """

    tree: tuple[int, ...]
    chords: tuple[int, ...]
    edge_momenta: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
    externals: int

    @property
    def loops(self) -> int:
        """L, the number of loop momenta, one per chord."""
        return len(self.chords)

    def momentum(self, edge: int) -> Momentum:
        """(loop, external), the coefficients of the momentum of an edge."""
        for index, loop, external in self.edge_momenta:
            if index == edge:
                return loop, external
        raise ValidationError(f"{edge} is not the index of an internal edge")


def _find(parent: list[int], v: int) -> int:
    while parent[v] != v:
        parent[v] = parent[parent[v]]
        v = parent[v]
    return v


def momentum_routing(fi: FeynmanIntegral, chords: Sequence[int] | None = None) -> MomentumRouting:
    """
    A momentum routing of the integral, from a spanning tree.

    By default the tree is grown in edge-index order (Kruskal), an edge joining it when it
    closes no cycle, and the chords are the other edges in index order. Chord i carries the
    loop momentum l_i from its v1 to its v2. The legs are incoming: leg j, numbered by its
    external vertex as for :attr:`FeynmanIntegral.momentum_products`, carries p_j into its
    vertex, legs 1, ..., n - 1 are independent and p_n = -(p_1 + ... + p_(n-1)). The
    momentum of a tree edge is then fixed by conservation: it is the momentum that flows
    into the side of the tree that holds its v1, from legs and chords.

    Parameters
    ----------
    fi
        The integral.
    chords
        The chords in the order of the loop momenta they carry, in place of the default;
        the other edges must form a spanning tree.

    Raises
    ------
    ValidationError
        If the integral has no loops or its propagators do not connect its vertices, or if
        the chords are not L distinct internal edges whose complement is a spanning tree.
    """
    graph = fi.graph
    edges = graph.get_internal_edges()
    vertices = graph.internal_vertices
    loops = graph.get_loop_count()
    if loops < 1:
        raise ValidationError("an integral family needs at least one loop")
    indices = [e.idx for e in edges]
    parent = list(range(vertices + 1))
    tree: list[int] = []
    if chords is None:
        chosen: list[int] = []
        for e in edges:
            a, b = _find(parent, e.v1), _find(parent, e.v2)
            if a == b:
                chosen.append(e.idx)
            else:
                parent[a] = b
                tree.append(e.idx)
    else:
        chosen = [int(c) for c in chords]
        if len(set(chosen)) != len(chosen) or not set(chosen) <= set(indices):
            raise ValidationError(f"the chords {tuple(chords)} are not distinct internal edges")
        if len(chosen) != loops:
            raise ValidationError(f"the integral has {loops} loops; got {len(chosen)} chords")
        for e in edges:
            if e.idx in chosen:
                continue
            a, b = _find(parent, e.v1), _find(parent, e.v2)
            if a == b:
                raise ValidationError(
                    f"the edges other than the chords {tuple(chords)} contain a cycle"
                )
            parent[a] = b
            tree.append(e.idx)
    if len(tree) != vertices - 1:
        raise ValidationError("the propagators do not connect the vertices of the graph")

    n = graph.external_legs
    externals = max(n - 1, 0)
    by_index = {e.idx: e for e in edges}
    legs = {leg.v2 - vertices: leg.v1 for leg in graph.get_external_edges()}

    def reduced(vector: list[int]) -> Momentum:
        """Coefficients of l and of p_1, ..., p_n, with p_n = -(p_1 + ... + p_(n-1))."""
        loop, external = vector[:loops], vector[loops:]
        if n:
            last = external[n - 1]
            external = [c - last for c in external[: n - 1]]
        return tuple(loop), tuple(external)

    momenta: dict[int, Momentum] = {}
    for i, c in enumerate(chosen):
        unit = [0] * (loops + n)
        unit[i] = 1
        momenta[c] = reduced(unit)
    neighbours: dict[int, list[tuple[int, int]]] = {v: [] for v in range(1, vertices + 1)}
    for t in tree:
        e = by_index[t]
        neighbours[e.v1].append((e.v2, t))
        neighbours[e.v2].append((e.v1, t))
    for t in tree:
        start = by_index[t].v1
        side, stack = {start}, [start]
        while stack:
            v = stack.pop()
            for w, through in neighbours[v]:
                if through != t and w not in side:
                    side.add(w)
                    stack.append(w)
        vector = [0] * (loops + n)
        for j, v in legs.items():
            if v in side:
                vector[loops + j - 1] += 1
        for i, c in enumerate(chosen):
            e = by_index[c]
            vector[i] += (e.v2 in side) - (e.v1 in side)
        momenta[t] = reduced(vector)
    return MomentumRouting(
        tree=tuple(tree),
        chords=tuple(chosen),
        edge_momenta=tuple((e, *momenta[e]) for e in indices),
        externals=externals,
    )


def _rank(rows: Sequence[Sequence[Fraction]]) -> int:
    """The rank of a matrix of Fractions, by Gaussian elimination."""
    work = [list(row) for row in rows]
    rank = 0
    columns = len(work[0]) if work else 0
    for col in range(columns):
        pivot = next((r for r in range(rank, len(work)) if work[r][col]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        head = work[rank]
        for r in range(rank + 1, len(work)):
            factor = work[r][col] / head[col]
            if factor:
                work[r] = [x - factor * y for x, y in zip(work[r], head, strict=True)]
        rank += 1
    return rank


def _gram(fi: FeynmanIntegral, n: int) -> list[list[sp.Expr]]:
    """p_j . p_k for j, k < n, with p_j^2 = -sum_(k != j) p_j . p_k since sum p = 0."""
    products = fi.momentum_products

    def product(j: int, k: int) -> sp.Expr:
        return sp.sympify(products.get((j, k), products.get((k, j), 0)))

    gram = [[sp.Integer(0)] * (n - 1) for _ in range(n - 1)]
    for j in range(1, n):
        for k in range(1, n):
            if j != k:
                gram[j - 1][k - 1] = product(j, k)
        gram[j - 1][j - 1] = sp.expand(-sum(product(j, k) for k in range(1, n + 1) if k != j))
    return gram


@dataclass(frozen=True)
class FamilyPolynomials:
    """
    U, F and G = U + F of an integral family, in its parameters.

    F carries the factor 1/mu^2 of the energy scale, as
    :attr:`FeynmanIntegral.symanzik` does.
    """

    u: sp.Expr
    f: sp.Expr
    g: sp.Expr
    variables: tuple[sp.Symbol, ...]


def _determinant(
    matrix: Sequence[Sequence[sp.Poly]],
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    memo: dict[tuple[tuple[int, ...], tuple[int, ...]], sp.Poly],
) -> sp.Poly:
    """The minor on rows x cols, by Laplace expansion along the first row."""
    key = (rows, cols)
    if key in memo:
        return memo[key]
    if len(rows) == 1:
        total = matrix[rows[0]][cols[0]]
    else:
        total = matrix[rows[0]][cols[0]] * 0
        for j, col in enumerate(cols):
            entry = matrix[rows[0]][col]
            if not entry.is_zero:
                minor = _determinant(matrix, rows[1:], cols[:j] + cols[j + 1 :], memo)
                total = total + entry * minor if j % 2 == 0 else total - entry * minor
    memo[key] = total
    return total


@dataclass(frozen=True)
class IntegralFamily:
    """
    The integral family of a Feynman integral: its routing, functions and parameters.

    The functions are the propagators of the graph by edge index, then the ISPs; the
    parameters are the integral's u_e, then z_1, z_2, ... for the ISPs. Build a family with
    :meth:`from_integral`, which returns only complete families. A family built directly
    need not be complete, and its polynomials are still those of its quadratic forms.

    Attributes
    ----------
    integral
        The Feynman integral; its graph and momentum products give the propagators and
        the products p_k . p_m.
    routing
        The momentum routing of the propagators.
    functions
        The functions D_alpha, propagators first.
    variables
        The parameter of each function.

    Raises
    ------
    ValidationError
        If the propagators are not those of the graph in edge order, a function after them
        is not an ISP, a function has another number of loop or external momenta than the
        routing, the parameters are not one distinct symbol per function, or a function
        repeats the form of an earlier one.
    """

    integral: FeynmanIntegral
    routing: MomentumRouting
    functions: tuple[FamilyFunction, ...]
    variables: tuple[sp.Symbol, ...]

    def __post_init__(self) -> None:
        functions = tuple(self.functions)
        variables = tuple(self.variables)
        object.__setattr__(self, "functions", functions)
        object.__setattr__(self, "variables", variables)
        if any(not isinstance(f, FamilyFunction) for f in functions):
            raise ValidationError("the functions of a family must be FamilyFunction instances")
        edges = [e.idx for e in self.integral.graph.get_internal_edges()]
        heads = functions[: len(edges)]
        if [f.edge for f in heads] != edges or any(f.kind != "propagator" for f in heads):
            raise ValidationError(
                f"the first functions must be the propagators of the edges {edges} in order"
            )
        for f in functions[len(edges) :]:
            if f.kind != "isp":
                raise ValidationError(f"{f.label} follows the propagators but is not an ISP")
        for f in functions:
            if (f.loops, f.externals) != (self.routing.loops, self.routing.externals):
                raise ValidationError(
                    f"{f.label} has {f.loops} loop and {f.externals} external momenta; the "
                    f"family has {self.routing.loops} and {self.routing.externals}"
                )
        if len(variables) != len(functions) or len(set(variables)) != len(variables):
            raise ValidationError("a family needs one distinct parameter per function")
        for k, f in enumerate(functions):
            for j in range(k):
                g = functions[j]
                if f.form[:3] == g.form[:3] and sp.expand(f.constant - g.constant) == 0:
                    raise ValidationError(
                        f"function {k + 1}, {f.label}, repeats function {j + 1}, {g.label}"
                    )

    @classmethod
    def from_integral(
        cls,
        fi: FeynmanIntegral,
        isps: str | Sequence[FamilyFunction] = "auto",
        chords: Sequence[int] | None = None,
    ) -> IntegralFamily:
        """
        The family of an integral, with its ISPs given or completed automatically.

        Parameters
        ----------
        fi
            The integral.
        isps
            "auto", the default, completes the propagators greedily: it adds
            -(l_i + p_k)^2 for each i and then k, and then -(l_i - l_j)^2 for i < j, each
            when it raises the rank. Otherwise the ISPs, in order.
        chords
            Passed to :func:`momentum_routing`.

        Raises
        ------
        ValidationError
            As :func:`momentum_routing` and the class raise; if the propagators are
            linearly dependent in the scalar products, as with edges in series or a
            bridge, so that no basis holding them is complete; if isps is neither "auto"
            nor a sequence of FamilyFunction; or if the functions do not form a complete
            basis.
        """
        routing = momentum_routing(fi, chords)
        loops, externals = routing.loops, routing.externals
        total = loops * (loops + 1) // 2 + loops * externals
        edges = fi.graph.get_internal_edges()
        propagators = [
            FamilyFunction.squared(
                *routing.momentum(e.idx),
                sp.expand(e.get_mass() ** 2),
                kind="propagator",
                edge=e.idx,
            )
            for e in edges
        ]
        rows = [f.coefficients() for f in propagators]
        rank = _rank(rows)
        if rank < len(rows):
            raise ValidationError(
                f"the {len(rows)} propagators span only {rank} dimensions in the scalar "
                "products, so no basis that holds them is complete; edges in series or a "
                "bridge make propagators dependent"
            )
        if isinstance(isps, str):
            if isps != "auto":
                raise ValidationError(f"isps must be 'auto' or a sequence of ISPs; got {isps!r}")
            chosen: list[FamilyFunction] = []
            units = [tuple(int(i == r) for r in range(loops)) for i in range(loops)]
            outer = [tuple(int(k == r) for r in range(externals)) for k in range(externals)]
            candidates = [
                FamilyFunction.squared(units[i], outer[k])
                for i in range(loops)
                for k in range(externals)
            ]
            candidates += [
                FamilyFunction.squared(
                    tuple(a - b for a, b in zip(units[i], units[j], strict=True)),
                    (0,) * externals,
                )
                for i in range(loops)
                for j in range(i + 1, loops)
            ]
            for candidate in candidates:
                if rank == total:
                    break
                grown = _rank([*rows, candidate.coefficients()])
                if grown > rank:
                    rows.append(candidate.coefficients())
                    chosen.append(candidate)
                    rank = grown
        else:
            chosen = list(isps)
            if any(not isinstance(f, FamilyFunction) for f in chosen):
                raise ValidationError("isps must be 'auto' or a sequence of FamilyFunction")
        variables = [
            sp.Symbol(f"{LEE_POMERANSKY_PARAM_PREFIX}_{e.idx}", nonnegative=True, real=True)
            for e in edges
        ]
        variables += [
            sp.Symbol(f"{ISP_PARAM_PREFIX}_{k}", nonnegative=True, real=True)
            for k in range(1, len(chosen) + 1)
        ]
        family = cls(fi, routing, (*propagators, *chosen), tuple(variables))
        if not family.complete:
            span = _rank(family.coefficient_matrix)
            raise ValidationError(
                f"the {family.size} functions do not form a complete basis: they span {span} "
                f"of the {total} scalar products l_i . l_j and l_i . p_k, and a complete basis "
                f"has exactly {total} functions"
            )
        return family

    @property
    def size(self) -> int:
        """N, the number of functions."""
        return len(self.functions)

    @property
    def coefficient_matrix(self) -> tuple[tuple[Fraction, ...], ...]:
        """The coefficients of the scalar products in each function, one row per function.

        The columns are those of :meth:`FamilyFunction.coefficients`: l_i . l_j for
        i <= j, row by row, then l_i . p_k.
        """
        return tuple(f.coefficients() for f in self.functions)

    @property
    def complete(self) -> bool:
        """Whether the functions form a complete basis: N = L(L + 1)/2 + L E of full rank."""
        loops, externals = self.routing.loops, self.routing.externals
        total = loops * (loops + 1) // 2 + loops * externals
        return self.size == total and _rank(self.coefficient_matrix) == total

    @cached_property
    def _polynomials(self) -> FamilyPolynomials:
        gens = self.variables
        loops, externals = self.routing.loops, self.routing.externals
        gram = _gram(self.integral, externals + 1) if externals else []

        def linear(coefficients: Sequence[object]) -> sp.Poly:
            expr = sp.Add(*(sp.sympify(c) * z for c, z in zip(coefficients, gens, strict=True)))
            return sp.Poly(expr, *gens)

        fs = self.functions
        a = [
            [linear([_rational(f.quadratic[i][j]) for f in fs]) for j in range(loops)]
            for i in range(loops)
        ]
        b = [
            [linear([_rational(f.linear[i][k]) for f in fs]) for k in range(externals)]
            for i in range(loops)
        ]
        c = linear([f.offset(gram) for f in fs])
        memo: dict[tuple[tuple[int, ...], tuple[int, ...]], sp.Poly] = {}
        everything = tuple(range(loops))
        det = _determinant(a, everything, everything, memo)
        # P b_j, the products with the external momenta, then sum_ij adj(A)_ij B_i . B_j.
        products = [
            [sp.Poly(gram[k][m], *gens) for m in range(externals)] for k in range(externals)
        ]
        pb = [
            [
                sum((products[k][m] * b[j][m] for m in range(externals)), det * 0)
                for k in range(externals)
            ]
            for j in range(loops)
        ]
        quadratic = det * 0
        for i in range(loops):
            for j in range(i, loops):
                if loops == 1:
                    adjugate = det * 0 + 1
                else:
                    rest_i = everything[:i] + everything[i + 1 :]
                    rest_j = everything[:j] + everything[j + 1 :]
                    adjugate = _determinant(a, rest_j, rest_i, memo)
                    adjugate = adjugate if (i + j) % 2 == 0 else -adjugate
                inner = sum((b[i][k] * pb[j][k] for k in range(externals)), det * 0)
                quadratic += adjugate * inner if i == j else 2 * adjugate * inner
        sign = 1 if loops % 2 == 0 else -1
        u = (sign * det).as_expr()
        scale = self.integral.graph.energy_scale
        f = sp.expand((sign * (c * det - quadratic)).as_expr() / scale**2)
        return FamilyPolynomials(u=u, f=f, g=sp.expand(u + f), variables=gens)

    def polynomials(self) -> FamilyPolynomials:
        """U, F and G = U + F of the family, from its quadratic forms (Lee's Eqs. 8-9)."""
        return self._polynomials

    @cached_property
    def _newton_polytope(self) -> NewtonPolytope:
        g = self.polynomials().g
        support = extract_monomial_support(g, list(self.variables))
        a_matrix = construct_gkz_matrix_from_exponents(
            [alpha for alpha, _ in support], len(self.variables)
        )
        return NewtonPolytope(support=support, a_matrix=a_matrix, parameters=list(self.variables))

    def newton_polytope(self) -> NewtonPolytope:
        """The Newton polytope of the family's G, with its support and GKZ matrix."""
        return self._newton_polytope
