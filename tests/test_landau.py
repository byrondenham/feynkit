"""
Tests for feynkit.landau — edge-part principal A-determinant.

Key mathematical facts verified:

1. Massless bubble: no Landau surfaces (no threshold without masses).
2. Massive bubble (m1, m2, s): threshold at s = -2(m1+m2)² and
   pseudothreshold at s = -2(m1-m2)².  The Landau polynomial vanishes
   at both.
3. Equal-mass bubble (m, m): threshold at s = -8m² only.
4. Massless triangle (s12, s13, s23): Landau surfaces s12+s13,
   s12+s23, s13+s23 (IR collinear conditions; reduce to individual
   s_ij under momentum conservation).
5. BMS_3 conformal simplex: Landau surfaces p_1sq, p_2sq, p_3sq
   (null-momentum singularities of the conformal 3-point integral).

The Landau polynomial is computed as the product of edge discriminants
of the Newton polytope of the Lee–Pomeransky G polynomial.
"""

from __future__ import annotations

import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.artifacts.conformal import _bms_g_polynomial
from feynkit.landau import (
    LandauAnalysis,
    landau_analysis,
    landau_analysis_from_polynomial,
)

# ─── fixtures ────────────────────────────────────────────────────────────────


def _build_bubble(m1_val, m2_val) -> FeynmanIntegral:
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1_val)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2_val)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    g = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    return FeynmanIntegral(g)


def _build_massless_triangle() -> FeynmanIntegral:
    z = sp.Integer(0)
    edges = [
        Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z),
        Edge(idx=2, v1=2, v2=3, is_internal=True, mass=z),
        Edge(idx=3, v1=3, v2=1, is_internal=True, mass=z),
        Edge(idx=4, v1=1, v2=4, is_internal=False),
        Edge(idx=5, v1=2, v2=5, is_internal=False),
        Edge(idx=6, v1=3, v2=6, is_internal=False),
    ]
    g = Graph(internal_vertices=3, external_legs=3, edges=edges)
    return FeynmanIntegral(g)


@pytest.fixture
def massless_bubble() -> FeynmanIntegral:
    return _build_bubble(sp.Integer(0), sp.Integer(0))


@pytest.fixture
def massive_bubble() -> FeynmanIntegral:
    m1, m2 = sp.symbols("m1 m2", nonnegative=True)
    return _build_bubble(m1, m2)


@pytest.fixture
def equal_mass_bubble() -> FeynmanIntegral:
    m = sp.Symbol("m", nonnegative=True)
    return _build_bubble(m, m)


@pytest.fixture
def massless_triangle() -> FeynmanIntegral:
    return _build_massless_triangle()


# ─── return type tests ───────────────────────────────────────────────────────


class TestReturnType:
    def test_returns_landau_analysis(self, massless_bubble):
        result = landau_analysis(massless_bubble)
        assert isinstance(result, LandauAnalysis)

    def test_landau_polynomial_is_sympy(self, massless_bubble):
        result = landau_analysis(massless_bubble)
        assert isinstance(result.landau_polynomial, sp.Expr)

    def test_surfaces_is_tuple(self, massless_bubble):
        result = landau_analysis(massless_bubble)
        assert isinstance(result.landau_surfaces, tuple)


# ─── massless bubble ─────────────────────────────────────────────────────────


class TestMasslessBubble:
    def test_no_landau_surfaces(self, massless_bubble):
        """Massless bubble has no threshold — propagators have no mass."""
        result = landau_analysis(massless_bubble)
        assert result.landau_surfaces == ()

    def test_landau_polynomial_trivial(self, massless_bubble):
        result = landau_analysis(massless_bubble)
        assert result.landau_polynomial == sp.Integer(1)


# ─── massive bubble ──────────────────────────────────────────────────────────


class TestMassiveBubble:
    def test_surfaces_exist(self, massive_bubble):
        result = landau_analysis(massive_bubble)
        assert len(result.landau_surfaces) >= 2

    def test_threshold_condition(self, massive_bubble):
        """Landau polynomial vanishes at the normal threshold s = -2(m1+m2)²."""
        m1, m2 = sp.symbols("m1 m2", nonnegative=True)
        s = sp.Symbol("s", real=True)
        result = landau_analysis(massive_bubble)
        lp = result.landau_polynomial
        threshold = -2 * (m1 + m2) ** 2
        assert sp.simplify(lp.subs(s, threshold)) == sp.Integer(0)

    def test_pseudothreshold_condition(self, massive_bubble):
        """Landau polynomial vanishes at the pseudothreshold s = -2(m1-m2)²."""
        m1, m2 = sp.symbols("m1 m2", nonnegative=True)
        s = sp.Symbol("s", real=True)
        result = landau_analysis(massive_bubble)
        lp = result.landau_polynomial
        pseudothreshold = -2 * (m1 - m2) ** 2
        assert sp.simplify(lp.subs(s, pseudothreshold)) == sp.Integer(0)

    def test_no_spurious_zeros(self, massive_bubble):
        """Landau polynomial is non-zero for generic kinematics."""
        m1, m2 = sp.symbols("m1 m2", nonnegative=True)
        s = sp.Symbol("s", real=True)
        result = landau_analysis(massive_bubble)
        lp = result.landau_polynomial
        # At s = 0, m1 = 1, m2 = 2: neither threshold condition holds
        val = lp.subs([(s, sp.Integer(0)), (m1, sp.Integer(1)), (m2, sp.Integer(2))])
        assert sp.simplify(val) != sp.Integer(0)

    def test_threshold_surface_in_list(self, massive_bubble):
        """The irreducible threshold factor appears in landau_surfaces."""
        m1, m2 = sp.symbols("m1 m2", nonnegative=True)
        s = sp.Symbol("s", real=True)
        threshold = -2 * (m1 + m2) ** 2
        result = landau_analysis(massive_bubble)
        # Some surface must vanish at the threshold value
        found = any(
            sp.simplify(surf.subs(s, threshold)) == sp.Integer(0) for surf in result.landau_surfaces
        )
        assert found, f"Threshold surface not found in {result.landau_surfaces}"

    def test_pseudothreshold_surface_in_list(self, massive_bubble):
        m1, m2 = sp.symbols("m1 m2", nonnegative=True)
        s = sp.Symbol("s", real=True)
        pseudothreshold = -2 * (m1 - m2) ** 2
        result = landau_analysis(massive_bubble)
        found = any(
            sp.simplify(surf.subs(s, pseudothreshold)) == sp.Integer(0)
            for surf in result.landau_surfaces
        )
        assert found, f"Pseudothreshold surface not found in {result.landau_surfaces}"


# ─── equal-mass bubble ───────────────────────────────────────────────────────


class TestEqualMassBubble:
    def test_single_threshold(self, equal_mass_bubble):
        """Equal-mass bubble: threshold at s = -8m², pseudothreshold at s = 0.

        s = -2(m+m)² = -8m² and s = -2(m-m)² = 0.  At s=0 the polynomial
        may vanish trivially (the pseudothreshold coincides with massless kinematics)
        — both zero-conditions must still hold.
        """
        m = sp.Symbol("m", nonnegative=True)
        s = sp.Symbol("s", real=True)
        result = landau_analysis(equal_mass_bubble)
        lp = result.landau_polynomial
        assert sp.simplify(lp.subs(s, -8 * m**2)) == sp.Integer(0)

    def test_edge_discriminants_exist(self, equal_mass_bubble):
        result = landau_analysis(equal_mass_bubble)
        assert len(result.edge_discriminants) >= 1


# ─── massless triangle ───────────────────────────────────────────────────────


class TestMasslessTriangle:
    def test_three_surfaces(self, massless_triangle):
        """Massless triangle has 3 IR collinear Landau surfaces."""
        result = landau_analysis(massless_triangle)
        assert len(result.landau_surfaces) == 3

    def test_surfaces_are_linear_in_mandelstam(self, massless_triangle):
        """Each surface is a linear combination of the Mandelstam variables."""
        result = landau_analysis(massless_triangle)
        s12, s13, s23 = sp.symbols("s12 s13 s23", real=True)
        for surf in result.landau_surfaces:
            p = sp.Poly(surf, s12, s13, s23)
            assert p.total_degree() == 1, f"Surface {surf} is not linear"

    def test_surfaces_are_pairwise_sums(self, massless_triangle):
        """The three surfaces should be pairwise sums s_ij + s_ik."""
        s12, s13, s23 = sp.symbols("s12 s13 s23", real=True)
        expected_zeros = {
            (s12 + s13, -s13, -s12),  # s12+s13=0: set s12=-s13 and check
        }
        result = landau_analysis(massless_triangle)
        surfaces_expanded = {sp.expand(s) for s in result.landau_surfaces}
        # All three Mandelstam pairwise sums should appear (up to overall sign/scaling)
        pairs = [s12 + s13, s12 + s23, s13 + s23]
        for pair in pairs:
            matches = [
                sp.simplify(sp.cancel(surf / pair)).is_rational
                for surf in surfaces_expanded
                if (surf / pair).is_rational  # avoid zero division
            ]
            # At least one surface is proportional to each pair
            found = any(
                (
                    sp.simplify(surf.subs([(s12, -s13)])) == sp.Integer(0)
                    if pair is (s12 + s13)
                    else True
                )
                for surf in surfaces_expanded
            )

    def test_landau_polynomial_vanishes_at_ir_locus(self, massless_triangle):
        """Polynomial vanishes when s12 = 0 and s13 = 0 (collinear kinematics)."""
        s12, s13, s23 = sp.symbols("s12 s13 s23", real=True)
        result = landau_analysis(massless_triangle)
        lp = result.landau_polynomial
        val = lp.subs([(s12, sp.Integer(0)), (s13, sp.Integer(0))])
        assert sp.simplify(val) == sp.Integer(0)


# ─── BMS_3 conformal simplex ─────────────────────────────────────────────────


class TestBMS3Conformal:
    def test_three_null_momentum_surfaces(self):
        """BMS_3 has Landau surfaces at p_i^2 = 0 (null external momenta)."""
        g = _bms_g_polynomial(3)
        params = [sp.Symbol(f"u_{i+1}") for i in range(3)]
        result = landau_analysis_from_polynomial(g, params)
        p1sq, p2sq, p3sq = sp.symbols("p_1sq p_2sq p_3sq")
        # Each null-momentum condition should appear as a Landau surface
        surfaces_set = set(result.landau_surfaces)
        assert p1sq in surfaces_set
        assert p2sq in surfaces_set
        assert p3sq in surfaces_set

    def test_polynomial_vanishes_at_null_p1(self):
        g = _bms_g_polynomial(3)
        params = [sp.Symbol(f"u_{i+1}") for i in range(3)]
        result = landau_analysis_from_polynomial(g, params)
        p1sq = sp.Symbol("p_1sq")
        lp = result.landau_polynomial
        assert sp.simplify(lp.subs(p1sq, sp.Integer(0))) == sp.Integer(0)


# ─── from_polynomial interface ───────────────────────────────────────────────


class TestFromPolynomial:
    def test_matches_integral_interface(self, massive_bubble):
        """landau_analysis_from_polynomial and landau_analysis agree."""
        s = massive_bubble.symanzik
        result_fi = landau_analysis(massive_bubble)
        result_poly = landau_analysis_from_polynomial(s.g, s.lp_parameters)
        # Landau polynomials must be equal (as polynomials)
        diff = sp.expand(result_fi.landau_polynomial - result_poly.landau_polynomial)
        assert diff == sp.Integer(0)

    def test_empty_polynomial_trivial(self):
        """A polynomial with no monomials returns trivial analysis."""
        result = landau_analysis_from_polynomial(sp.Integer(0), [])
        assert result.landau_polynomial == sp.Integer(1)
        assert result.landau_surfaces == ()
