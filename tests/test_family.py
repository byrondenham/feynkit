"""Integral families: a momentum routing, a complete basis and the family's polynomial.

Lee14 is R. N. Lee, LiteRed 1.4: a powerful tool for the reduction of the multiloop
integrals, arXiv:1310.1145; Wei22 is S. Weinzierl, Feynman Integrals, Springer 2022,
arXiv:2201.03593. Equations are cited by number.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core import Edge, Graph
from feynkit.core.exceptions import ValidationError
from feynkit.family import (
    FamilyFunction,
    IntegralFamily,
    MomentumRouting,
    momentum_routing,
)
from feynkit.generate import generate_graphs

ZERO = sp.Integer(0)

# The five instances of the design, and three more: masses on every line, an off-shell
# vertex and a three-loop ladder.
BOX = ("13e|2e|3e|e|:zzzz", "massless_on_shell")
SUNRISE = ("111e|e|:nnn", None)
KITE = ("12e|23|3e|e|:nnnnn", None)
DOUBLE_BOX = ("15e|24|3e|4e|5|e|:zzzzzzz", "massless_on_shell")
DOUBLE_BOX_OFF_SHELL = ("15e|24|3e|4e|5|e|:zzzzzzz", None)
MASSIVE_DOUBLE_BOX = ("15e|24|3e|4e|5|e|:nnnnnnn", None)
CROSSED_VERTEX = ("123|4e|4e|4e||:zzzzzz", None)
TRIPLE_BOX = ("123|45|46|5e|7|e|7e|e|:zzzzzzzzzz", "massless_on_shell")

INSTANCES = [
    BOX,
    SUNRISE,
    KITE,
    DOUBLE_BOX,
    DOUBLE_BOX_OFF_SHELL,
    MASSIVE_DOUBLE_BOX,
    CROSSED_VERTEX,
    TRIPLE_BOX,
]


def _integral(instance: tuple[str, str | None]) -> FeynmanIntegral:
    cnickel, kinematics = instance
    return FeynmanIntegral.from_cnickel(cnickel, kinematics=kinematics)  # type: ignore[arg-type]


def _generated(cnickel: str) -> FeynmanIntegral:
    """The integral of a generated graph; Mandelstam invariants need two legs or more."""
    legs = Graph.from_cnickel(cnickel).external_legs
    return FeynmanIntegral.from_cnickel(cnickel, use_mandelstam=legs >= 2)


def _graph(vertices: int, internal: list[tuple[int, int]], legs: list[int]) -> Graph:
    """Massless propagators e = 1, 2, ... from v1 to v2, and leg j at the vertex legs[j - 1]."""
    edges = [
        Edge(idx=i, v1=a, v2=b, is_internal=True, mass=ZERO)
        for i, (a, b) in enumerate(internal, start=1)
    ]
    for j, v in enumerate(legs, start=1):
        edges.append(Edge(idx=len(internal) + j, v1=v, v2=vertices + j, is_internal=False))
    return Graph(internal_vertices=vertices, external_legs=len(legs), edges=edges)


def _weinzierl_double_box() -> FeynmanIntegral:
    """Wei22's double box (Fig. 2.3, Eq. 2.30): q_e from v1 to v2, all legs outgoing."""
    graph = _graph(
        6,
        [(1, 2), (2, 4), (3, 1), (4, 3), (6, 4), (3, 5), (5, 6)],
        [1, 2, 6, 5],
    )
    return FeynmanIntegral(graph).with_kinematics("massless_on_shell")


def _lee_vertex() -> FeynmanIntegral:
    """Lee14's two-loop vertex (Eq. 17): l, r, p - l, q - r, p + r - l, l + q - r."""
    graph = _graph(5, [(1, 4), (2, 3), (1, 3), (2, 4), (3, 5), (4, 5)], [1, 2, 5])
    return FeynmanIntegral(graph).with_kinematics("massless_on_shell")


def _series(fi: FeynmanIntegral) -> bool:
    """Whether two propagators carry the same momentum up to its sign, as edges in series do."""
    routing = momentum_routing(fi)
    seen: set[tuple[int, ...]] = set()
    for _, loop, external in routing.edge_momenta:
        vector = (*loop, *external)
        if vector in seen or tuple(-c for c in vector) in seen:
            return True
        seen.add(vector)
    return False


def _conserved(fi: FeynmanIntegral, routing: MomentumRouting) -> bool:
    """Momentum conservation at every vertex, with the legs incoming and p_n = -sum p_j."""
    graph = fi.graph
    n_int = graph.internal_vertices
    loops, externals = routing.loops, routing.externals
    for v in range(1, n_int + 1):
        total = [0] * (loops + externals)
        for e in graph.get_internal_edges():
            loop, external = routing.momentum(e.idx)
            sign = (e.v2 == v) - (e.v1 == v)
            for i, c in enumerate((*loop, *external)):
                total[i] += sign * c
        for leg in graph.get_external_edges():
            if leg.v1 != v:
                continue
            j = leg.v2 - n_int
            if j <= externals:
                total[loops + j - 1] += 1
            else:
                for k in range(externals):
                    total[loops + k] -= 1
        if any(total):
            return False
    return True


class TestFamilyFunction:
    def test_squared_is_minus_q_squared_plus_m_squared(self) -> None:
        """-(l1 - l2 + p1)^2 + m^2 in Lee14's form a l.l + 2 b l.p + c (Eq. 2)."""
        m = sp.Symbol("m", positive=True)
        f = FamilyFunction.squared((1, -1), (1, 0), m**2)
        assert f.quadratic == ((-1, 1), (1, -1))
        assert f.linear == ((-1, 0), (1, 0))
        assert f.external == ((-1, 0), (0, 0))
        assert f.constant == m**2
        assert (f.kind, f.edge) == ("isp", None)
        assert f.label == "-(l1 - l2 + p1)^2 + m**2"
        assert all(isinstance(x, Fraction) for row in f.quadratic for x in row)

    def test_product(self) -> None:
        """l_1 . p_2 has b^{12} = 1/2 and nothing else."""
        f = FamilyFunction.product(1, 2, loops=2, externals=3)
        assert f.quadratic == ((0, 0), (0, 0))
        assert f.linear == ((0, Fraction(1, 2), 0), (0, 0, 0))
        assert f.external == ((0, 0, 0),) * 3
        assert f.constant == 0
        assert f.label == "l1.p2"

    def test_offset_adds_the_external_products(self) -> None:
        f = FamilyFunction.squared((1,), (1, 1), 5)
        p11, p12, p22 = sp.symbols("p11 p12 p22")
        assert sp.expand(f.offset(((p11, p12), (p12, p22))) - (5 - p11 - 2 * p12 - p22)) == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"quadratic": ((1, 2), (0, 1)), "linear": ((0,), (0,)), "external": ((0,),)},
            {"quadratic": ((1,),), "linear": ((0, 0),), "external": ((0,),)},
            {"quadratic": ((0.5,),), "linear": ((0,),), "external": ((0,),)},
            {"quadratic": (), "linear": (), "external": ()},
            {"quadratic": ((1,),), "linear": ((0,),), "external": ((0,),), "kind": "other"},
        ],
    )
    def test_rejects_bad_forms(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            FamilyFunction(label="D", **kwargs)

    def test_rejects_bad_products(self) -> None:
        with pytest.raises(ValidationError):
            FamilyFunction.product(3, 1, loops=2, externals=1)
        with pytest.raises(ValidationError):
            FamilyFunction.product(1, 2, loops=2, externals=1)


class TestMomentumRouting:
    def test_box(self) -> None:
        fi = _integral(BOX)
        routing = momentum_routing(fi)
        assert routing.tree == (1, 2, 3)
        assert routing.chords == (4,)
        assert (routing.loops, routing.externals) == (1, 3)
        assert routing.momentum(4) == ((1,), (0, 0, 0))
        assert _conserved(fi, routing)

    def test_weinzierl_double_box(self) -> None:
        """With k_1 = q_3 and k_2 = q_6, the momenta of Wei22's Eq. 2.32, legs incoming."""
        fi = _weinzierl_double_box()
        routing = momentum_routing(fi, chords=(3, 6))
        assert routing.chords == (3, 6)
        assert set(routing.tree) == {1, 2, 4, 5, 7}
        expected = {
            1: ((1, 0), (1, 0, 0)),
            2: ((1, 0), (1, 1, 0)),
            3: ((1, 0), (0, 0, 0)),
            4: ((1, 1), (0, 0, 0)),
            5: ((0, 1), (-1, -1, 0)),
            6: ((0, 1), (0, 0, 0)),
            7: ((0, 1), (-1, -1, -1)),
        }
        assert {e: routing.momentum(e) for e in range(1, 8)} == expected
        assert _conserved(fi, routing)

    @pytest.mark.parametrize("chords", [(3,), (3, 6, 7), (1, 3), (3, 3), (3, 99)])
    def test_rejects_bad_chords(self, chords: tuple[int, ...]) -> None:
        """Chords number L, are distinct internal edges, and leave a spanning tree."""
        with pytest.raises(ValidationError):
            momentum_routing(_weinzierl_double_box(), chords=chords)

    @pytest.mark.parametrize(
        "cnickel",
        [s for n in (0, 1, 2, 3, 4) for s in generate_graphs(2, n, masses="z")]
        + list(generate_graphs(3, 3, edges=9, masses="z")),
    )
    def test_conservation(self, cnickel: str) -> None:
        fi = _generated(cnickel)
        routing = momentum_routing(fi)
        assert _conserved(fi, routing)
        for i, chord in enumerate(routing.chords):
            loop, external = routing.momentum(chord)
            assert loop == tuple(int(k == i) for k in range(routing.loops))
            assert not any(external)


class TestBasis:
    @pytest.mark.parametrize(
        ("instance", "isps"),
        [
            (BOX, []),
            (SUNRISE, ["-(l1 + p1)^2", "-(l2 + p1)^2"]),
            (KITE, ["-(l1 + p1)^2", "-(l1 + p2)^2"]),
            (DOUBLE_BOX, ["-(l1 + p1)^2", "-(l2 + p2)^2"]),
        ],
    )
    def test_automatic_isps(self, instance: tuple[str, str | None], isps: list[str]) -> None:
        fi = _integral(instance)
        family = IntegralFamily.from_integral(fi)
        edges = [e.idx for e in fi.graph.get_internal_edges()]
        assert [f.label for f in family.functions[len(edges) :]] == isps
        assert [f.edge for f in family.functions] == edges + [None] * len(isps)
        assert [f.kind for f in family.functions] == ["propagator"] * len(edges) + ["isp"] * len(
            isps
        )
        assert family.size == len(edges) + len(isps)
        assert family.complete
        names = [str(x) for x in family.variables]
        assert names == [f"u_{e}" for e in edges] + [f"z_{k}" for k in range(1, len(isps) + 1)]

    def test_lee_vertex_basis(self) -> None:
        """Lee14's two-loop vertex: D_7 = (l - r)^2 completes it, with determinant 32 (Eq. 17)."""
        fi = _lee_vertex()
        d7 = FamilyFunction.squared((1, -1), (0, 0))
        family = IntegralFamily.from_integral(fi, isps=[d7], chords=(1, 2))
        momenta = [family.routing.momentum(e) for e in range(1, 7)]
        assert momenta == [
            ((1, 0), (0, 0)),
            ((0, 1), (0, 0)),
            ((-1, 0), (1, 0)),
            ((0, -1), (0, 1)),
            ((-1, 1), (1, 0)),
            ((1, -1), (0, 1)),
        ]
        assert family.complete
        matrix = sp.Matrix(family.coefficient_matrix)
        assert matrix.shape == (7, 7)
        assert abs(matrix.det()) == 32

    @pytest.mark.parametrize(("loops", "legs"), [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)])
    def test_maximal_diagrams(self, loops: int, legs: int) -> None:
        """M = E + 3L - 2 propagators leave (L - 1)(L + 2E - 4)/2 ISPs (Lee14, section 2)."""
        externals = legs - 1
        expected = (loops - 1) * (loops + 2 * externals - 4) // 2
        maximal = externals + 3 * loops - 2
        graphs = [_generated(s) for s in generate_graphs(loops, legs, edges=maximal, masses="z")]
        assert any(not _series(fi) for fi in graphs)
        for fi in graphs:
            if _series(fi):
                with pytest.raises(ValidationError, match="propagators"):
                    IntegralFamily.from_integral(fi)
                continue
            family = IntegralFamily.from_integral(fi)
            assert family.size - maximal == expected, fi.cnickel
            assert family.complete

    def test_given_isps(self) -> None:
        fi = _integral(SUNRISE)
        isps = [
            FamilyFunction.product(1, 1, loops=2, externals=1),
            FamilyFunction.squared((0, 1), (1,)),
        ]
        family = IntegralFamily.from_integral(fi, isps=isps)
        assert family.functions[3:] == tuple(isps)
        assert family.complete

    def test_incomplete(self) -> None:
        fi = _integral(SUNRISE)
        with pytest.raises(ValidationError, match="complete"):
            IntegralFamily.from_integral(fi, isps=[FamilyFunction.squared((1, 0), (1,))])
        with pytest.raises(ValidationError, match="complete"):
            IntegralFamily.from_integral(fi, isps=[])

    def test_dependent_isp(self) -> None:
        """-(l1 + l2 - p1)^2 is the first propagator of the sunrise without its mass."""
        fi = _integral(SUNRISE)
        isps = [FamilyFunction.squared((1, 0), (1,)), FamilyFunction.squared((1, 1), (-1,))]
        with pytest.raises(ValidationError, match="complete"):
            IntegralFamily.from_integral(fi, isps=isps)

    def test_too_many(self) -> None:
        fi = _integral(BOX)
        with pytest.raises(ValidationError, match="complete"):
            IntegralFamily.from_integral(fi, isps=[FamilyFunction.squared((1,), (1, 1, 0))])

    def test_repeated(self) -> None:
        fi = _integral(SUNRISE)
        isp = FamilyFunction.squared((1, 0), (1,))
        with pytest.raises(ValidationError, match="repeats"):
            IntegralFamily.from_integral(fi, isps=[isp, isp])
        massless = FeynmanIntegral.from_cnickel("111e|e|:zzz")
        with pytest.raises(ValidationError, match="repeats"):
            IntegralFamily.from_integral(massless, isps=[FamilyFunction.squared((1, 0), (0,))])

    def test_rejects_bad_isps(self) -> None:
        fi = _integral(SUNRISE)
        with pytest.raises(ValidationError):
            IntegralFamily.from_integral(fi, isps=[FamilyFunction.squared((1,), (1,))])
        with pytest.raises(ValidationError):
            IntegralFamily.from_integral(
                fi, isps=[FamilyFunction.squared((1, 0), (1,), kind="propagator", edge=1)]
            )
        with pytest.raises(ValidationError):
            IntegralFamily.from_integral(fi, isps="manual")  # type: ignore[arg-type]

    def test_dependent_propagators(self) -> None:
        """Two propagators in series carry one momentum, so no basis is complete."""
        graph = _graph(3, [(1, 2), (2, 3), (3, 1), (1, 3)], [1, 3])
        fi = FeynmanIntegral(graph)
        with pytest.raises(ValidationError, match="propagators"):
            IntegralFamily.from_integral(fi)


class TestPolynomials:
    @pytest.mark.parametrize("instance", INSTANCES, ids=[f"{c}-{k}" for c, k in INSTANCES])
    def test_graph_polynomials_at_zero(self, instance: tuple[str, str | None]) -> None:
        """U and G of the family at z = 0 are feynkit's u_lp and g, exactly."""
        fi = _integral(instance)
        family = IntegralFamily.from_integral(fi)
        polys = family.polynomials()
        assert polys.variables == family.variables
        zero = dict.fromkeys(family.variables[len(fi.graph.get_internal_edges()) :], 0)
        assert sp.expand(polys.u.subs(zero) - fi.symanzik.u_lp) == 0
        assert sp.expand(polys.g.subs(zero) - fi.symanzik.g) == 0
        assert sp.expand(polys.g - polys.u - polys.f) == 0

    @pytest.mark.parametrize("instance", INSTANCES, ids=[f"{c}-{k}" for c, k in INSTANCES])
    def test_graph_polytope_is_a_face(self, instance: tuple[str, str | None]) -> None:
        """The points of the family with no ISP exponent are the points of the graph."""
        fi = _integral(instance)
        family = IntegralFamily.from_integral(fi)
        n = len(fi.graph.get_internal_edges())
        points = family.newton_polytope().points
        face = {p[:n] for p in points if not any(p[n:])}
        assert face == set(fi.newton_polytope.points)
        assert all(max(p[k] for p in points) <= 2 for k in range(n, family.size))

    @pytest.mark.parametrize(
        "cnickel",
        [s for n in (0, 1, 2, 3, 4) for s in generate_graphs(2, n, masses="z")]
        + [s for n in (1, 2) for s in generate_graphs(2, n, masses="zn")],
    )
    def test_two_loop_graphs(self, cnickel: str) -> None:
        fi = _generated(cnickel)
        if _series(fi):
            with pytest.raises(ValidationError, match="propagators"):
                IntegralFamily.from_integral(fi)
            return
        family = IntegralFamily.from_integral(fi)
        zero = dict.fromkeys(family.variables[len(fi.graph.get_internal_edges()) :], 0)
        assert sp.expand(family.polynomials().g.subs(zero) - fi.symanzik.g) == 0

    def test_bubble(self) -> None:
        """Lee14's Eqs. 8-9 for the massive bubble, in feynkit's convention D = -q^2 + m^2."""
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        family = IntegralFamily.from_integral(fi)
        u1, u2 = family.variables
        m1, m2 = (e.get_mass() for e in fi.graph.get_internal_edges())
        s = -fi.momentum_products[(1, 2)]
        mu = fi.graph.energy_scale
        polys = family.polynomials()
        assert polys.u == u1 + u2
        expected = ((u1 + u2) * (m1**2 * u1 + m2**2 * u2) - s * u1 * u2) / mu**2
        assert sp.expand(polys.f - expected) == 0
