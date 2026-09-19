"""Tests for the principal A-determinant and Landau surfaces.

Oracles: the one-loop closed form of Dlapa, Helmer, Papathanasiou and
Tellander (arXiv:2304.02629, eq. 1LoopEA): the reduced principal
A-determinant of G = U + F is the product of the principal minors of the
modified Cayley matrix. Face-by-face computation must reproduce it.
"""

from __future__ import annotations

import pytest
import sympy as sp

from feynkit import FeynmanIntegral, landau_analysis, landau_analysis_from_polynomial
from feynkit.landau import (
    _singular_binary,
    one_loop_landau_surfaces,
    one_loop_principal_a_determinant,
)

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")


def _monic(expr: sp.Expr) -> sp.Expr:
    """Normalise a polynomial factor up to a constant, for set comparison."""
    expr = sp.expand(expr)
    return sp.Poly(expr, *sorted(expr.free_symbols, key=str)).monic().as_expr()


def _factor_set(
    factors: tuple[sp.Expr, ...] | list[sp.Expr], symbols: set[sp.Symbol]
) -> set[sp.Expr]:
    """Normalised set of the given irreducible factors that involve the symbols."""
    return {_monic(f) for f in factors if f.free_symbols & symbols}


def _same_surfaces(fi: FeynmanIntegral) -> bool:
    kin = _kin(fi)
    faces = _factor_set(landau_analysis(fi).landau_surfaces, kin)
    closed = _factor_set(one_loop_landau_surfaces(fi), kin)
    assert faces == closed, faces ^ closed
    return True


def _kin(fi: FeynmanIntegral) -> set[sp.Symbol]:
    return (
        fi.symanzik.f.free_symbols - set(fi.symanzik.schwinger_parameters) - {fi.graph.energy_scale}
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def massive_bubble() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("11e|e|:nn")


@pytest.fixture(scope="module")  # type: ignore[misc]
def massless_triangle() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


class TestMassiveBubble:
    def test_face_result_matches_one_loop_closed_form(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        assert _same_surfaces(massive_bubble)

    def test_threshold_and_pseudothreshold_are_surfaces(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        """The Kallen factor splits over the masses into s = (m1 + m2)^2 and s = (m1 - m2)^2."""
        m1, m2 = (sp.Symbol(f"m_{i}", nonnegative=True, real=True) for i in (1, 2))
        s = sp.Symbol("s", real=True)
        surfaces = _factor_set(landau_analysis(massive_bubble).landau_surfaces, {s, m1, m2})
        normal = _monic((m1 + m2) ** 2 - s)
        pseudo = _monic((m1 - m2) ** 2 - s)
        assert normal in surfaces and pseudo in surfaces
        assert s in surfaces  # second-type (Gram) singularity p^2 = 0
        assert m1 in surfaces and m2 in surfaces  # vertex factors: mass singularities

    def test_kallen_factor_appears_once(self, massive_bubble: FeynmanIntegral) -> None:
        """Regression: the edge exponent bug squared the Kallen factor."""
        s = sp.Symbol("s", real=True)
        la = landau_analysis(massive_bubble)
        edge = [
            f
            for f in la.face_discriminants
            if f.dimension == 1 and s in f.discriminant.free_symbols
        ]
        assert len(edge) == 1
        assert sp.degree(sp.expand(edge[0].discriminant), s) == 2


class TestMasslessTriangle:
    def test_edges_are_trivial(self, massless_triangle: FeynmanIntegral) -> None:
        """Every edge of the massless triangle's polytope is a two-point simplex."""
        la = landau_analysis(massless_triangle)
        assert all(f.discriminant == 1 for f in la.face_discriminants if f.dimension == 1)

    def test_surfaces_are_external_masses_and_gram_determinant(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        p1, p2, p3 = (sp.Symbol(f"p{i}^2", real=True) for i in (1, 2, 3))
        surfaces = _factor_set(landau_analysis(massless_triangle).landau_surfaces, {p1, p2, p3})
        gram = sp.expand(p1**2 + p2**2 + p3**2 - 2 * p1 * p2 - 2 * p1 * p3 - 2 * p2 * p3)
        assert surfaces == {p1, p2, p3, _monic(gram)}

    def test_face_result_matches_one_loop_closed_form(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        assert _same_surfaces(massless_triangle)


class TestOneMassTriangle:
    @requires_singular
    def test_face_result_matches_one_loop_closed_form(self) -> None:
        assert _same_surfaces(FeynmanIntegral.from_cnickel("12e|2e|e|:nzz"))


class TestMasslessBox:
    @requires_singular
    def test_face_result_matches_one_loop_closed_form(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
        assert not landau_analysis(fi).skipped_faces
        assert _same_surfaces(fi)


class TestStructure:
    def test_massless_bubble_is_singular_only_at_p_squared_zero(self) -> None:
        la = landau_analysis(FeynmanIntegral.from_cnickel("11e|e|:zz"))
        assert la.landau_surfaces == (sp.Symbol("s", real=True),)

    def test_simplex_faces_have_unit_discriminant(self, massive_bubble: FeynmanIntegral) -> None:
        la = landau_analysis(massive_bubble)
        assert all(
            f.discriminant == 1 for f in la.face_discriminants if f.is_simplex and f.dimension > 0
        )

    def test_faces_include_every_dimension(self, massive_bubble: FeynmanIntegral) -> None:
        dims = {f.dimension for f in landau_analysis(massive_bubble).face_discriminants}
        assert dims == {0, 1, 2}

    def test_polynomial_interface_matches_integral_interface(
        self, massive_bubble: FeynmanIntegral
    ) -> None:
        sym = massive_bubble.symanzik
        la_poly = landau_analysis_from_polynomial(sym.g, list(sym.lp_parameters))
        la_int = landau_analysis(massive_bubble)
        assert la_poly.landau_surfaces == la_int.landau_surfaces

    def test_closed_form_rejects_multi_loop(self) -> None:
        with pytest.raises(ValueError):
            one_loop_principal_a_determinant(FeynmanIntegral.from_cnickel("111e|e|:nnn"))

    def test_empty_polynomial_trivial(self) -> None:
        u = sp.symbols("u1:3")
        la = landau_analysis_from_polynomial(sp.Integer(0), list(u))
        assert la.landau_surfaces == ()


class TestBanana:
    @requires_singular
    def test_two_loop_banana_thresholds(self) -> None:
        """PLD example 3.4: the massive banana B_3 has factors m_e^2, s and
        s - (m_1 +/- m_2 +/- m_3)^2."""
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        la = landau_analysis(fi)
        s = sp.Symbol("s", real=True)
        m = [sp.Symbol(f"m_{i}", nonnegative=True, real=True) for i in (1, 2, 3)]
        surfaces = _factor_set(la.landau_surfaces, {s, *m})
        for e2 in (1, -1):
            for e3 in (1, -1):
                assert _monic((m[0] + e2 * m[1] + e3 * m[2]) ** 2 - s) in surfaces
        assert s in surfaces
