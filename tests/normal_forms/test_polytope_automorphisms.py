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
            pytest.param("112|3|4e|5e|5e|e|:nnnnzzz", 24, marks=pytest.mark.slow),
            ("112|3|4e|5e|5e|e|:zznnnzz", 72),
        ],
    )
    def test_two_loop_pairs_from_the_columns_of_a(self, cnickel: str, order: int) -> None:
        # The columns of A come in graded order; the second graph takes about 35 s.
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
