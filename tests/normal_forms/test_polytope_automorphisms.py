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

import sympy as sp

from feynkit import FeynmanIntegral, PolytopeAutomorphisms
from feynkit.normal_forms.polytope_automorphisms import (
    coefficient_preserving_indices,
    compute_graph_automorphisms,
    compute_polytope_automorphisms,
)

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
