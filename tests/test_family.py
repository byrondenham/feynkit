"""Integral families: a momentum routing, a complete basis and the family's polynomial.

Lee14 is R. N. Lee, LiteRed 1.4: a powerful tool for the reduction of the multiloop
integrals, arXiv:1310.1145; Wei22 is S. Weinzierl, Feynman Integrals, Springer 2022,
arXiv:2201.03593. Equations are cited by number.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

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


def test_exported() -> None:
    import feynkit

    for name in ("FamilyFunction", "FamilyPolynomials", "IntegralFamily", "MomentumRouting"):
        assert name in feynkit.__all__
    assert feynkit.momentum_routing is momentum_routing


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
        assert f.label == "-(l1 - l2 + p1)^2 + m^2"
        assert FamilyFunction.squared((-1, 1), (-1, 0), m**2).label == f.label
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

    @pytest.mark.parametrize("chords", [(3,), (3, 6, 7), (1, 3), (3, 3), (3, 99), ("3", 6.0)])
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

    def test_edges_in_series(self) -> None:
        """Two propagators in series carry one momentum, so no basis is complete."""
        graph = _graph(3, [(1, 2), (2, 3), (3, 1), (1, 3)], [1, 3])
        with pytest.raises(ValidationError, match="propagators") as raised:
            IntegralFamily.from_integral(FeynmanIntegral(graph))
        message = str(raised.value)
        assert "edges 1 and 2 are in series" in message
        assert "higher power" in message
        assert "partial fractions" in message

    def test_bridge(self) -> None:
        """A bridge carries no loop momentum, so its propagator is a constant."""
        graph = _graph(4, [(1, 2), (2, 1), (2, 3), (3, 4), (4, 3)], [1, 4])
        with pytest.raises(ValidationError, match="edge 3 carries no loop momentum"):
            IntegralFamily.from_integral(FeynmanIntegral(graph))


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


def _weinzierl_family() -> IntegralFamily:
    """Wei22's family of the double box (Exercise 44, p. 707): q_8 = k_1 - p_1 - p_3 and
    q_9 = k_2 + p_1 + p_3 with outgoing legs, so l_1 + p_1 + p_3 and l_2 - p_1 - p_3 here."""
    isps = [
        FamilyFunction.squared((1, 0), (1, 0, 1)),
        FamilyFunction.squared((0, 1), (-1, 0, -1)),
    ]
    return IntegralFamily.from_integral(_weinzierl_double_box(), isps=isps, chords=(3, 6))


class TestSectors:
    def test_weinzierl_functions(self) -> None:
        """The nine functions are Wei22's -q_j^2, in his order (Eq. 2.32, p. 707)."""
        family = _weinzierl_family()
        momenta = [
            ((1, 0), (1, 0, 0)),
            ((1, 0), (1, 1, 0)),
            ((1, 0), (0, 0, 0)),
            ((1, 1), (0, 0, 0)),
            ((0, 1), (-1, -1, 0)),
            ((0, 1), (0, 0, 0)),
            ((0, 1), (-1, -1, -1)),
            ((1, 0), (1, 0, 1)),
            ((0, 1), (-1, 0, -1)),
        ]
        assert [f.form for f in family.functions] == [
            FamilyFunction.squared(*q).form for q in momenta
        ]

    def test_weinzierl_masters(self) -> None:
        """The sectors of the double box's masters (Wei22, Eq. 6.83)."""
        family = _weinzierl_family()
        masters = {
            "001110000": 28,
            "100100100": 73,
            "011011000": 54,
            "100111000": 57,
            "111100100": 79,
            "101110100": 93,
            "111111100": 127,
        }
        for word, nid in masters.items():
            n = tuple(int(c) for c in word)
            assert family.sector_id(n) == nid
            assert family.sector(n) == n
            assert family.corner(nid) == n
        assert family.sector_id((1, 1, 1, 1, 1, 1, 1, -1, 0)) == 127

    def test_sector_of_any_indices(self) -> None:
        family = IntegralFamily.from_integral(_integral(SUNRISE))
        assert family.sector((2, 0, -3, -1, 0)) == (1, 0, 0, 0, 0)
        assert family.sector_id((2, 0, -3, -1, 0)) == 1
        assert family.sector_id((1, 1, 1, 0, -2)) == 7
        assert [family.corner(k) for k in range(8)] == [
            tuple((k >> j) & 1 for j in range(3)) + (0, 0) for k in range(8)
        ]

    @pytest.mark.parametrize(
        "n", [(1, 1, 1, 1, 0), (1, 1, 1, 0), (1, 1, 1, 0, 0, 0), (1, 1, Fraction(1, 2), 0, 0)]
    )
    def test_rejects_bad_indices(self, n: tuple) -> None:
        """An ISP is never a denominator, and indices are integers, one per function."""
        family = IntegralFamily.from_integral(_integral(SUNRISE))
        for method in (family.sector_id, family.sector, family.beta):
            with pytest.raises(ValidationError):
                method(n)

    @pytest.mark.parametrize("nid", [-1, 8, 16, 1.0])
    def test_rejects_bad_ids(self, nid: object) -> None:
        family = IntegralFamily.from_integral(_integral(SUNRISE))
        with pytest.raises(ValidationError):
            family.corner(nid)  # type: ignore[arg-type]

    def test_beta(self) -> None:
        """beta = (-D/2, -n) at D = d0 - 2 eps, as beta0 + eps beta1."""
        family = IntegralFamily.from_integral(_integral(SUNRISE))
        beta0, beta1 = family.beta((1, 2, 1, -1, 0), 4)
        assert beta0 == (-2, -1, -2, -1, 1, 0)
        assert beta1 == (1, 0, 0, 0, 0, 0)
        assert all(isinstance(x, Fraction) for x in (*beta0, *beta1))
        assert family.beta((1, 1, 1, 0, 0), Fraction(3))[0][0] == Fraction(-3, 2)
        eps = sp.Symbol("epsilon")
        fi = FeynmanIntegral.from_cnickel(SUNRISE[0], dimension=6 - 2 * eps)
        assert IntegralFamily.from_integral(fi).beta((1, 1, 1, 0, 0))[0][0] == -3


class TestNumeratorLayers:
    @pytest.mark.parametrize(
        ("instance", "size"),
        [(SUNRISE, 14), (KITE, 36), (DOUBLE_BOX, 25), (DOUBLE_BOX_OFF_SHELL, 44)],
    )
    def test_height_one(self, instance: tuple[str, str | None], size: int) -> None:
        assert len(IntegralFamily.from_integral(_integral(instance)).numerator_layer(1)) == size

    @pytest.mark.parametrize("instance", [SUNRISE, KITE, DOUBLE_BOX])
    def test_layers_partition_the_support(self, instance: tuple[str, str | None]) -> None:
        """Height 0 is the graph's G, and the layers are the terms by their ISP degree."""
        fi = _integral(instance)
        family = IntegralFamily.from_integral(fi)
        n = len(fi.graph.get_internal_edges())
        graph = {(*p, *(0,) * (family.size - n)): c for p, c in fi.newton_polytope.support}
        layer = dict(family.numerator_layer(0))
        assert layer.keys() == graph.keys()
        assert all(sp.expand(layer[p] - graph[p]) == 0 for p in graph)
        support = dict(family.newton_polytope().support)
        layers = [family.numerator_layer(h) for h in range(fi.loop_count + 2)]
        assert sum(len(x) for x in layers) == len(support)
        for h, terms in enumerate(layers):
            assert [p for p, _ in terms] == sorted(p for p, _ in terms)
            for p, c in terms:
                assert sum(p[n:]) == h
                assert c == support[p]
        assert family.numerator_layer(fi.loop_count + 2) == ()

    def test_rejects_bad_heights(self) -> None:
        family = IntegralFamily.from_integral(_integral(SUNRISE))
        for height in (-1, 1.5, True):
            with pytest.raises(ValidationError):
                family.numerator_layer(height)  # type: ignore[arg-type]


def _one_loop_numerator(g: sp.Expr, u: sp.Symbol, z: sp.Symbol, d: sp.Expr) -> sp.Expr:
    """J with index 1 on u and -1 on z at one loop, by Lee14's Eq. 12.

    Eq. 11 makes the z-integration -d/dz at z = 0, which leaves
    Gamma(D/2)/Gamma(D) (D/2) int_0^oo G_0^(-D/2-1) G_1 du with G_0 = u + M u^2 and
    G_1 = dG/dz at z = 0, a sum of Beta functions.
    """
    g0 = sp.expand(g.subs(z, 0))
    g1 = sp.expand(sp.diff(g, z).subs(z, 0))
    mass = g0.coeff(u, 2)
    assert sp.expand(g0 - u - mass * u**2) == 0
    s = d / 2 + 1
    total = sum(
        c * mass ** (s - k - 1) * sp.beta(k + 1 - s, 2 * s - k - 1)
        for (k,), c in sp.Poly(g1, u).terms()
    )
    return sp.gamma(d / 2) / sp.gamma(d) * d / 2 * total


class TestNumeratorSign:
    """Lee14's Eq. 11: an index n <= 0 becomes (-1)^n (d/dz)^(-n) at z = 0."""

    d = sp.Rational(13, 5)
    point = {sp.Symbol("mu", positive=True): 1}

    @staticmethod
    def _tadpole(d: sp.Expr, mass_squared: sp.Expr) -> sp.Expr:
        """int d^Dl/(i pi^(D/2)) (-l^2 + m^2)^(-1)."""
        return sp.gamma(1 - d / 2) * mass_squared ** (d / 2 - 1)

    def test_bubble(self) -> None:
        """J(-1, 1) = int D_1/D_2 = (m_1^2 - m_2^2 - p^2) T(m_2^2) after a shift of l."""
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        family = IntegralFamily.from_integral(fi)
        u1, u2 = family.variables
        m1, m2 = (e.get_mass() for e in fi.graph.get_internal_edges())
        s = -fi.momentum_products[(1, 2)]
        values = {fi.graph.energy_scale: 1, m1: 2, m2: 3, s: sp.Rational(7, 5)}
        g = family.polynomials().g.subs(values)
        found = _one_loop_numerator(g, u2, u1, self.d)
        expected = (4 - 9 - sp.Rational(7, 5)) * self._tadpole(self.d, 9)
        assert abs(sp.N(found - expected, 40)) < sp.Float(10) ** -30
        assert abs(sp.N(expected, 40)) > 1

    def test_tadpole_with_an_isp(self) -> None:
        """J(1, -1) = int -(l + p)^2/(-l^2 + m^2) = -(m^2 + p^2) T(m^2)."""
        m = sp.Symbol("m", positive=True)
        graph = Graph(
            internal_vertices=1,
            external_legs=2,
            edges=[
                Edge(idx=1, v1=1, v2=1, is_internal=True, mass=m),
                Edge(idx=2, v1=1, v2=2, is_internal=False),
                Edge(idx=3, v1=1, v2=3, is_internal=False),
            ],
        )
        fi = FeynmanIntegral(graph)
        family = IntegralFamily.from_integral(fi)
        assert [f.label for f in family.functions] == ["-(l1)^2 + m^2", "-(l1 + p1)^2"]
        u, z = family.variables
        s = -fi.momentum_products[(1, 2)]
        values = {fi.graph.energy_scale: 1, m: 2, s: sp.Rational(7, 5)}
        g = family.polynomials().g.subs(values)
        found = _one_loop_numerator(g, u, z, self.d)
        expected = -(4 + sp.Rational(7, 5)) * self._tadpole(self.d, 4)
        assert abs(sp.N(found - expected, 40)) < sp.Float(10) ** -30


def test_guide_examples_print_what_the_guide_says(capsys: pytest.CaptureFixture[str]) -> None:
    """The code blocks of the guide's section on integral families, run in turn."""
    guide = (Path(__file__).resolve().parents[1] / "docs" / "guide.md").read_text(encoding="utf-8")
    heading = "\n## Integral families\n"
    assert heading in guide
    section = guide.split(heading, 1)[1].split("\n## ", 1)[0]
    blocks = [block.split("```", 1)[0] for block in section.split("```python\n")[1:]]
    assert len(blocks) == 4
    namespace: dict[str, object] = {}
    for block in blocks:
        exec(compile(block, "docs/guide.md", "exec"), namespace)
    assert capsys.readouterr().out.splitlines() == [
        "9 True",
        "propagator 1 -(l2 - p2 - p3)^2",
        "propagator 2 -(l2 - p1 - p2 - p3)^2",
        "propagator 3 -(l1 - p2 - p3)^2",
        "propagator 4 -(l1 - l2)^2",
        "propagator 5 -(l1 - p3)^2",
        "propagator 6 -(l1)^2",
        "propagator 7 -(l2)^2",
        "isp None -(l1 + p1)^2",
        "isp None -(l2 + p2)^2",
        "0",
        "56 26",
        "[26, 25, 5, 0]",
        "28",
        "(1, 1, 1, 1, 1, 1, 1, 0, 0)",
    ]
