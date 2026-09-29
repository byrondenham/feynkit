"""
Tests for compute_polytope_automorphisms, compute_graph_automorphisms,
and coefficient_preserving_indices.

Expected group orders (unimodular automorphisms of Newton polytopes):
  massless bubble    (11e|e|:zz)      -> |Aut| = 6  (S_3: Newton polytope is a 2-simplex)
  massless triangle  (12e|2e|e|:zzz)  -> |Aut| = 48 (B_3: Newton polytope is octahedron)
  one-mass triangle  (12e|2e|e|:nzz)  -> |Aut| = 6  (S_3: reduced symmetry)
  massless box       (12e|3e|3e|e|:zzzz) -> |Aut| = 120 (hyperoctahedral group)
  massless 3-banana  (111e|e|:zzz)    -> |Aut| = 24 (S_4: Newton polytope is 3-simplex)

Graph automorphisms (vertex permutations only, edge permutations not counted):
  massless bubble:   order 2  (swap vertices 1 <-> 2)
  massless triangle: order 6  (S_3 on three vertices)
  one-mass triangle: order 2  (swap the two massless-edge vertices)
  massless box:      order 8  (D_4 on four vertices)
  massless 3-banana: order 2  (Z/2: swap vertices 1 <-> 2; S_3 edge perms not vertex perms)
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from feynkit import FeynmanIntegral, PolytopeAutomorphisms, symmetry_pairs
from feynkit.normal_forms import is_unimodular_equivalent
from feynkit.normal_forms.polytope_automorphisms import (
    coefficient_preserving_indices,
    compute_graph_automorphisms,
    compute_polytope_automorphisms,
)
from feynkit.polytope import polytope_data

# -- helpers ------------------------------------------------------------------


def _fi(cnickel: str) -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel(cnickel)


def _points(cnickel: str):
    return _fi(cnickel).newton_polytope.points


# -- PolytopeAutomorphisms structure ------------------------------------------


class TestReturnType:
    def test_is_dataclass(self):
        auts = compute_polytope_automorphisms(_points("11e|e|:zz"))
        assert isinstance(auts, PolytopeAutomorphisms)

    def test_identity_always_present(self):
        for cnickel in ["11e|e|:zz", "12e|2e|e|:zzz", "12e|2e|e|:nzz"]:
            auts = compute_polytope_automorphisms(_points(cnickel))
            n_dim = len(auts.maps[0][1])
            ident = sp.ImmutableMatrix(sp.eye(n_dim))
            zero = sp.ImmutableMatrix(sp.zeros(n_dim, 1))
            assert (ident, zero) in auts.maps, f"Identity missing for {cnickel}"

    def test_order_equals_maps_length(self):
        for cnickel in ["11e|e|:zz", "12e|2e|e|:zzz"]:
            auts = compute_polytope_automorphisms(_points(cnickel))
            assert auts.order == len(auts.maps)
            assert auts.order == len(auts.vertex_permutations)

    def test_no_duplicate_vertex_permutations(self):
        for cnickel in ["12e|2e|e|:zzz", "111e|e|:zzz"]:
            auts = compute_polytope_automorphisms(_points(cnickel))
            seen = set()
            for vp in auts.vertex_permutations:
                key = tuple(vp)
                assert key not in seen, f"Duplicate vperm {key} for {cnickel}"
                seen.add(key)

    def test_orbits_partition_vertices(self):
        for cnickel in ["11e|e|:zz", "12e|2e|e|:zzz"]:
            auts = compute_polytope_automorphisms(_points(cnickel))
            pts = _points(cnickel)
            flat = sorted(v for orb in auts.vertex_orbits for v in orb)
            assert flat == list(range(len(pts))), f"Orbits don't cover all vertices for {cnickel}"


# -- Group orders --------------------------------------------------------------


class TestGroupOrders:
    def test_massless_bubble_order_6(self):
        # Newton polytope is a 2-simplex -> |Aut| = 6 = |S_3|
        auts = compute_polytope_automorphisms(_points("11e|e|:zz"))
        assert auts.order == 6

    def test_massless_triangle_order_48(self):
        # Newton polytope is a regular octahedron -> |Aut| = 48 = |B_3|
        auts = compute_polytope_automorphisms(_points("12e|2e|e|:zzz"))
        assert auts.order == 48

    def test_one_mass_triangle_order_6(self):
        # Broken S_3 symmetry -> |Aut| = 6
        auts = compute_polytope_automorphisms(_points("12e|2e|e|:nzz"))
        assert auts.order == 6

    def test_massless_box_order_120(self):
        # Hyperoctahedral symmetry -> |Aut| = 120
        auts = compute_polytope_automorphisms(_points("12e|3e|3e|e|:zzzz"))
        assert auts.order == 120

    def test_massless_banana_3prop_order_24(self):
        # Newton polytope is a 3-simplex -> |Aut| = 24 = |S_4|
        auts = compute_polytope_automorphisms(_points("111e|e|:zzz"))
        assert auts.order == 24


# -- Witness map validity ------------------------------------------------------


class TestWitnessValidity:
    def _check_maps(self, cnickel: str):

        pts = _points(cnickel)
        auts = compute_polytope_automorphisms(pts)
        pt_set = set(pts)
        for U, t in auts.maps:
            for p in pts:
                p_col = sp.Matrix(list(p))
                img = U * p_col + t
                img_key = tuple(int(x) for x in img)
                assert (
                    img_key in pt_set
                ), f"Map ({U}, {t}) sends {p} to {img_key} not in polytope for {cnickel}"

    def test_bubble(self):
        self._check_maps("11e|e|:zz")

    def test_triangle(self):
        self._check_maps("12e|2e|e|:zzz")

    def test_one_mass_triangle(self):
        self._check_maps("12e|2e|e|:nzz")

    def test_box(self):
        self._check_maps("12e|3e|3e|e|:zzzz")

    def test_banana(self):
        self._check_maps("111e|e|:zzz")


# -- Graph automorphisms -------------------------------------------------------


class TestGraphAutomorphisms:
    def test_massless_bubble_order_2(self):
        fi = _fi("11e|e|:zz")
        auts = compute_graph_automorphisms(fi.graph)
        assert len(auts) == 2

    def test_massless_triangle_order_6(self):
        fi = _fi("12e|2e|e|:zzz")
        auts = compute_graph_automorphisms(fi.graph)
        assert len(auts) == 6

    def test_identity_always_present(self):
        for cnickel in ["11e|e|:zz", "12e|2e|e|:zzz", "12e|2e|e|:nzz"]:
            fi = _fi(cnickel)
            V = fi.graph.internal_vertices
            auts = compute_graph_automorphisms(fi.graph)
            assert list(range(1, V + 1)) in auts

    def test_one_mass_triangle_order_2(self):
        fi = _fi("12e|2e|e|:nzz")
        auts = compute_graph_automorphisms(fi.graph)
        assert len(auts) == 2

    def test_massless_box_order_8(self):
        fi = _fi("12e|3e|3e|e|:zzzz")
        auts = compute_graph_automorphisms(fi.graph)
        assert len(auts) == 8

    def test_banana_vertex_aut_order_2(self):
        # 3-prop banana: vertex automorphisms are Z/2 (swap the two vertices).
        # S_3 permutations of the 3 propagators are edge symmetries, not vertex perms.
        fi = _fi("111e|e|:zzz")
        auts = compute_graph_automorphisms(fi.graph)
        assert len(auts) == 2


# -- FeynmanIntegral facade ----------------------------------------------------


class TestFacade:
    def test_polytope_automorphisms_property(self):
        fi = _fi("12e|2e|e|:zzz")
        auts = fi.polytope_automorphisms
        assert isinstance(auts, PolytopeAutomorphisms)
        assert auts.order == 48

    def test_graph_automorphisms_property(self):
        fi = _fi("12e|2e|e|:zzz")
        gauts = fi.graph_automorphisms
        assert isinstance(gauts, list)
        assert len(gauts) == 6  # S_3 on triangle vertices

    def test_cached(self):
        fi = _fi("12e|2e|e|:zzz")
        a1 = fi.polytope_automorphisms
        a2 = fi.polytope_automorphisms
        assert a1 is a2


# -- Coefficient-preserving subgroup ------------------------------------------


class TestCoefficientPreserving:
    def test_identity_always_preserving(self):
        for cnickel in ["11e|e|:zz", "12e|2e|e|:zzz", "12e|2e|e|:nzz"]:
            fi = _fi(cnickel)
            auts = fi.polytope_automorphisms
            idx = coefficient_preserving_indices(fi, auts)
            assert 0 in idx

    def test_bubble_has_two_preserving(self):
        # G = u_1 + u_2 + s/(2 mu^2) u_1u_2.  The swap u_1<->u_2 is the only non-trivial
        # coefficient-preserving automorphism (it fixes the (1,1) monomial).
        fi = _fi("11e|e|:zz")
        auts = fi.polytope_automorphisms
        idx = coefficient_preserving_indices(fi, auts)
        assert len(idx) == 2

    def test_preserving_strictly_fewer_than_order(self):
        # With distinct Mandelstam coefficients, not all polytope auts preserve.
        fi = _fi("12e|2e|e|:zzz")
        auts = fi.polytope_automorphisms
        idx = coefficient_preserving_indices(fi, auts)
        assert len(idx) < auts.order


# -- Exact vertices and edges --------------------------------------------------

# The support of 123e|4e|4e|4e||:nzznnn with its coordinates and points permuted, one point
# per word. A floating hull took the point (1, 1, 0, 0, 1, 0), which is not a vertex, for one,
# and found order 1.
_RELABELLED_WORDS = (
    "010101 001101 101000 021000 010020 100200 001100 000110 101100 001001 110010 100110 "
    "100100 010110 011000 110001 000120 011010 010011 000102 000210 120000 000201 001002 "
    "100011 020001 020010 000021 101010 010001 000111 001200 001011 110100 011001 100101 "
    "000011 100010 101001 010002 001110 111000 010010 000012 000101 110000 011100 100020"
)
RELABELLED = [tuple(int(c) for c in word) for word in _RELABELLED_WORDS.split()]


class TestExactVertices:
    """The vertices and the 1-skeleton come from the certified face lattice."""

    @pytest.mark.parametrize(
        ("cnickel", "order"),
        [
            # A floating hull found 46, 49 and 38 vertices where there are 43, 45 and 36,
            # and orders 4, 12 and 12.
            ("112|3|4e|5e|5e|e|:nnnnnzz", 24),
            ("112|3|4e|5e|5e|e|:nnnnzzz", 24),
            ("112|3|4e|5e|5e|e|:zznnnzz", 72),
        ],
    )
    def test_two_loop_orders(self, cnickel: str, order: int) -> None:
        fi = _fi(cnickel)
        assert fi.polytope_automorphisms.order == order
        # symmetry_pairs on the Newton points gave 4, 12 and 12.
        assert len(symmetry_pairs(np.array(fi.newton_polytope.points))) == order

    @pytest.mark.parametrize(
        ("cnickel", "order"),
        [
            ("112|3|4e|5e|5e|e|:nnnnnzz", 24),
            ("112|3|4e|5e|5e|e|:nnnnzzz", 24),
            ("112|3|4e|5e|5e|e|:zznnnzz", 72),
        ],
    )
    def test_two_loop_pairs_from_the_columns_of_a(self, cnickel: str, order: int) -> None:
        # The columns of A come in graded order.
        assert len(_fi(cnickel).symmetry_pairs) == order

    def test_permutations_index_the_vertices_of_polytope_data(self) -> None:
        points = _points("112|3|4e|5e|5e|e|:nnnnnzz")
        vertices = polytope_data(points).vertices
        auts = compute_polytope_automorphisms(points)
        assert sorted(v for orbit in auts.vertex_orbits for v in orbit) == list(
            range(len(vertices))
        )
        for (U, t), perm in zip(auts.maps, auts.vertex_permutations, strict=True):
            for i, v in enumerate(vertices):
                assert tuple(U * sp.Matrix(v) + t) == vertices[perm[i]]

    def test_relabelled_support(self) -> None:
        original = _points("123e|4e|4e|4e||:nzznnn")
        assert sorted(RELABELLED) == sorted((p[1], p[0], p[2], p[3], p[5], p[4]) for p in original)
        assert compute_polytope_automorphisms(original).order == 4
        assert compute_polytope_automorphisms(RELABELLED).order == 4
        assert len(symmetry_pairs(np.array(RELABELLED))) == 4
        assert is_unimodular_equivalent(original, RELABELLED).equivalent

    def test_massive_tadpole(self) -> None:
        # A segment in R^1: the identity and the reflection.
        auts = compute_polytope_automorphisms([(1,), (2,)])
        assert auts.order == 2
        assert sorted(auts.vertex_permutations) == [[0, 1], [1, 0]]
        assert len(symmetry_pairs([(1,), (2,)])) == 2

    @pytest.mark.parametrize("shear", [8 * 10**7, 10**9])
    def test_a_sheared_triangle(self, shear: int) -> None:
        # The triangle (0, 0), (2, 0), (0, 1), with (1, 0) on an edge, sheared by
        # (x, y) -> (x + shear y, y). The differences (2, 0) and (shear, 1) of its vertices
        # have floating rank 1, and the floating choice of a basis found none: order 1.
        small = [(0, 0), (1, 0), (2, 0), (0, 1)]
        sheared = [(x + shear * y, y) for x, y in small]
        auts = compute_polytope_automorphisms(sheared)
        assert auts.order == 2
        vertices = polytope_data(sheared).vertices
        for (U, t), perm in zip(auts.maps, auts.vertex_permutations, strict=True):
            for i, v in enumerate(vertices):
                assert tuple(U * sp.Matrix(v) + t) == vertices[perm[i]]


# -- Below full dimension ------------------------------------------------------


def _at_zero(cnickel: str, names: tuple[str, ...] = ()) -> FeynmanIntegral:
    """The integral of the string with the named invariants set to 0."""
    graph_legs = cnickel.split(":")[0].count("e")
    fi = FeynmanIntegral.from_cnickel(cnickel, use_mandelstam=graph_legs >= 2)
    symbols = {s for v in fi.momentum_products.values() for s in sp.sympify(v).free_symbols}
    zero = {s: 0 for s in symbols if s.name in names}
    assert len(zero) == len(names)
    products = {k: sp.expand(sp.sympify(v).subs(zero)) for k, v in fi.momentum_products.items()}
    return fi.with_(momentum_products=products)


BOX_ZERO = ("p1^2", "p2^2", "p3^2", "p4^2", "s12", "s23")


class TestBelowFullDimension:
    """The group of P in the affine lattice aff(P) cap Z^n, with one lift per element."""

    @pytest.mark.parametrize(
        ("cnickel", "zero", "dimension", "ambient", "order"),
        [
            ("011e|e|:znn", (), 2, 3, 2),
            ("012e|2e|e|:zzzz", (), 3, 4, 48),
            ("012e|2e|e|:znnn", (), 3, 4, 6),
            ("012e|2e|e|:znzz", (), 3, 4, 6),
            ("111e|e|:zzz", ("s",), 2, 3, 6),
            ("12e|2e|e|:zzz", ("p1^2", "p2^2", "p3^2"), 2, 3, 6),
            ("12e|3e|3e|e|:zzzz", BOX_ZERO, 3, 4, 24),
            ("11e|e|:zz", ("s",), 1, 2, 2),
            ("01e|e|:zn", (), 1, 2, 2),
            ("0|:n", (), 1, 1, 2),
            ("0|:z", (), 0, 1, 1),
        ],
    )
    def test_orders_and_lifts(
        self, cnickel: str, zero: tuple[str, ...], dimension: int, ambient: int, order: int
    ) -> None:
        fi = _at_zero(cnickel, zero)
        points = [tuple(int(x) for x in p) for p in fi.newton_polytope.points]
        data = polytope_data(points)
        assert (data.dimension, data.ambient_dimension) == (dimension, ambient)
        auts = fi.polytope_automorphisms
        assert auts.order == order == len(auts.maps) == len(auts.vertex_permutations)
        for (U, t), perm in zip(auts.maps, auts.vertex_permutations, strict=True):
            assert abs(U.det()) == 1
            assert all(x.is_Integer for x in [*U, *t])

            def image(p: tuple[int, ...], U: sp.Matrix = U, t: sp.Matrix = t) -> tuple[int, ...]:
                return tuple(int(x) for x in U * sp.Matrix(p) + t)

            assert [image(v) for v in data.vertices] == [data.vertices[j] for j in perm]
            assert {image(p) for p in points} == set(points)

    @pytest.mark.parametrize(
        ("with_loop", "without"),
        [
            ("011e|e|:znn", "11e|e|:nn"),
            ("012e|2e|e|:zzzz", "12e|2e|e|:zzz"),
            ("012e|2e|e|:znnn", "12e|2e|e|:nnn"),
            ("012e|2e|e|:znzz", "12e|2e|e|:nzz"),
        ],
    )
    def test_a_massless_self_loop_keeps_the_group(self, with_loop: str, without: str) -> None:
        # G is the loop's parameter times the G of the graph without it.
        assert (
            _fi(with_loop).polytope_automorphisms.order == _fi(without).polytope_automorphisms.order
        )

    def test_coefficient_preserving_lifts(self) -> None:
        # With p_1^2 = p_2^2 = p_3^2 every permutation of the three legs preserves the
        # coefficients, with the massless self-loop and without it.
        for cnickel in ("012e|2e|e|:zzzz", "12e|2e|e|:zzz"):
            fi = _fi(cnickel)
            p = [sp.Symbol(f"p{i}^2", real=True) for i in (1, 2, 3)]
            equal = {p[1]: p[0], p[2]: p[0]}
            fi = fi.with_(
                momentum_products={
                    k: sp.expand(sp.sympify(v).subs(equal)) for k, v in fi.momentum_products.items()
                }
            )
            assert len(coefficient_preserving_indices(fi, fi.polytope_automorphisms)) == 6

    def test_chart_labels(self) -> None:
        # In the ambient space every label of 012e|2e|e|:znnn is 0; in the chart they separate
        # the vertices u_1 u_i from u_1 u_i^2.
        auts = compute_polytope_automorphisms(_points("012e|2e|e|:znnn"))
        assert auts.vertex_orbits == [[0, 1, 2], [3, 4, 5]]


def _random_unimodular(n: int, rng: np.random.Generator) -> np.ndarray:
    """A product of elementary integer matrices and a permutation, in GL_n(Z)."""
    U = np.eye(n, dtype=np.int64)
    for _ in range(8):
        i, j = rng.choice(n, size=2, replace=False)
        U[i] += int(rng.choice([-1, 1])) * U[j]
    return U[rng.permutation(n)]


class TestEmbedding:
    """The group of a lattice polytope does not depend on how it is embedded."""

    def test_triangle_of_index_two(self) -> None:
        # Its differences span a sublattice of index 2; only the reflection that fixes
        # (0, 1) preserves the integer points of its plane, where Aut_L would give 6.
        for points in (
            [(0, 0), (2, 0), (0, 1)],
            [(0, 0, 0), (2, 0, 0), (0, 1, 0)],
            [(0, 0, 0), (2, 2, 0), (0, 1, 1)],
        ):
            assert compute_polytope_automorphisms(points).order == 2

    @pytest.mark.parametrize("seed", range(4))
    def test_random_embeddings(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        polytopes = [
            [(0, 0), (2, 0), (0, 1)],
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            _points("11e|e|:nn"),
            _points("12e|2e|e|:zzz"),
        ]
        for points in polytopes:
            n = len(points[0])
            m = n + int(rng.integers(1, 3))
            Q = _random_unimodular(m, rng)
            o = rng.integers(-3, 4, size=m)
            embedded = [
                tuple(int(x) for x in Q @ np.array([*p, *[0] * (m - n)]) + o) for p in points
            ]
            assert (
                compute_polytope_automorphisms(embedded).order
                == compute_polytope_automorphisms(points).order
            )


def _large_image(points: list[tuple[int, ...]], k: int) -> list[tuple[int, ...]]:
    """The points under S = [[1, k], [k, k^2 + 1]] in GL_2(Z), or its extension to GL_3(Z)."""
    S = sp.Matrix([[1, k], [k, k * k + 1]])
    if len(points[0]) == 3:
        S = sp.Matrix([[1, k, 0], [k, k * k + 1, 0], [0, k, 1]])
    return [tuple(int(x) for x in S * sp.Matrix(p)) for p in points]


LARGE = {
    "triangle": ([(0, 0), (1, 0), (0, 1)], 6),
    "square": ([(0, 0), (1, 0), (0, 1), (1, 1)], 8),
    "simplex": ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], 24),
}


class TestLargeCoordinates:
    """The candidate maps are found in integers, so a unimodular image keeps its group."""

    @pytest.mark.parametrize("k", [700, 1000, 10**4, 10**5])
    @pytest.mark.parametrize("name", sorted(LARGE))
    def test_orders(self, name: str, k: int) -> None:
        # In floating point W_b W_a^-1 was not close enough to an integer matrix: at
        # k = 1000 the triangle kept 2 of its 6 automorphisms, and from 3000 only the
        # identity. k = 700 runs in int64; the others need Python integers, and at 10^5
        # the coordinates exceed 2^31, so the labels would overflow int64 as well.
        points, order = LARGE[name]
        image = _large_image(points, k)
        auts = compute_polytope_automorphisms(image)
        assert auts.order == order
        vertices = polytope_data(image).vertices
        for (U, t), perm in zip(auts.maps, auts.vertex_permutations, strict=True):
            assert abs(U.det()) == 1
            for i, v in enumerate(vertices):
                assert tuple(U * sp.Matrix(v) + t) == vertices[perm[i]]


class TestNeighbourSearch:
    """The search maps an anchor's neighbours to neighbours of its image."""

    @pytest.mark.parametrize(
        ("cnickel", "order"),
        [("12e|3e|4e|4e|e|:zzzzz", 720), ("12e|3e|4e|5e|5e|e|:zzzzzz", 5040)],
        ids=["pentagon", "hexagon"],
    )
    def test_massless_pentagon_and_hexagon(self, cnickel: str, order: int) -> None:
        # Every vertex carries the same label, and the search over all vertices took about a
        # minute for the pentagon and did not finish in 25 minutes for the hexagon.
        auts = compute_polytope_automorphisms(_points(cnickel))
        assert auts.order == order
        assert len({tuple(perm) for perm in auts.vertex_permutations}) == order
        assert len(auts.vertex_orbits) == 1

    def test_the_maps_keep_their_order(self) -> None:
        # The maps come in the order in which the search over all vertices, anchored at
        # vertex 0 with a label-diverse basis, found them.
        auts = compute_polytope_automorphisms(_points("12e|2e|e|:nzz"))
        assert auts.vertex_permutations == [
            [0, 1, 2, 3, 4, 5, 6],
            [0, 2, 1, 3, 5, 4, 6],
            [4, 1, 6, 3, 0, 5, 2],
            [4, 6, 1, 3, 5, 0, 2],
            [5, 2, 6, 3, 0, 4, 1],
            [5, 6, 2, 3, 4, 0, 1],
        ]
