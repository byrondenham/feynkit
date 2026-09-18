"""Tests for the GKZ form of the Schwinger representation (dehomogenised Symanzik polynomials)."""

from __future__ import annotations

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.parametrisations.schwinger import SchwingerParametrisation


def _schwinger(fi: FeynmanIntegral) -> SchwingerParametrisation:
    sym = fi.symanzik
    return SchwingerParametrisation(
        fi.graph, fi.dimension, fi.loop_count, fi.propagator_exponents, sym.u, sym.f
    )


@pytest.fixture  # type: ignore[misc]
def massless_bubble() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("11e|e|:zz")


@pytest.fixture  # type: ignore[misc]
def massless_triangle() -> FeynmanIntegral:
    return FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")


class TestDehomogenisedSymanzikPolynomials:
    def test_returns_one_variable_fewer_than_propagators(self, massless_triangle: FeynmanIntegral) -> None:
        _, new_vars = _schwinger(massless_triangle).dehomogenised_symanzik_polynomials()
        assert len(new_vars) == 2
        assert all(isinstance(v, sp.Symbol) for v in new_vars)

    def test_bubble_polynomials_set_last_parameter_to_one(self, massless_bubble: FeynmanIntegral) -> None:
        (u_tilde, f_tilde), (u1,) = _schwinger(massless_bubble).dehomogenised_symanzik_polynomials()
        sym = massless_bubble.symanzik
        a1, a2 = sym.schwinger_parameters
        assert sp.expand(u_tilde - sym.u.subs({a1: u1, a2: 1})) == 0
        assert sp.expand(f_tilde - sym.f.subs({a1: u1, a2: 1})) == 0

    def test_polynomials_contain_only_new_variables(self, massless_triangle: FeynmanIntegral) -> None:
        sch = _schwinger(massless_triangle)
        (u_tilde, f_tilde), new_vars = sch.dehomogenised_symanzik_polynomials()
        old = set(massless_triangle.symanzik.schwinger_parameters)
        assert not (u_tilde.free_symbols & old)
        assert not (f_tilde.free_symbols & old)
        assert set(new_vars) <= u_tilde.free_symbols | f_tilde.free_symbols

    def test_homogeneity_recovers_original_polynomials(self, massless_triangle: FeynmanIntegral) -> None:
        """U(t u_1, ..., t u_{N-1}, t) = t^L Ũ(u) and F(...) = t^{L+1} F̃(u)."""
        sch = _schwinger(massless_triangle)
        (u_tilde, f_tilde), new_vars = sch.dehomogenised_symanzik_polynomials()
        sym = massless_triangle.symanzik
        t = sp.Symbol("t", positive=True)
        alphas = sym.schwinger_parameters
        subs = {a: t * v for a, v in zip(alphas[:-1], new_vars)}
        subs[alphas[-1]] = t
        L = massless_triangle.loop_count
        assert sp.expand(sym.u.subs(subs) - t**L * u_tilde) == 0
        assert sp.expand(sym.f.subs(subs) - t ** (L + 1) * f_tilde) == 0


class TestSchwingerAMatrix:
    def test_bubble_a_matrix_columns(self, massless_bubble: FeynmanIntegral) -> None:
        """Ũ = u1 + 1 and F̃ ∝ u1 give the 3×3 block matrix with two homogenising rows."""
        A = _schwinger(massless_bubble).get_A_matrix()
        assert A.shape == (3, 3)
        cols = {tuple(A[:, j]) for j in range(A.cols)}
        assert cols == {(1, 0, 1), (1, 0, 0), (0, 1, 1)}

    def test_block_structure(self, massless_triangle: FeynmanIntegral) -> None:
        sch = _schwinger(massless_triangle)
        (u_tilde, f_tilde), new_vars = sch.dehomogenised_symanzik_polynomials()
        n = len(sp.Poly(u_tilde, *new_vars).terms())
        m = len(sp.Poly(f_tilde, *new_vars).terms())
        A = sch.get_A_matrix()
        assert A.shape == (2 + len(new_vars), n + m)
        assert list(A[0, :]) == [1] * n + [0] * m
        assert list(A[1, :]) == [0] * n + [1] * m
        assert all(x.is_integer for x in A)

    def test_entries_are_integers(self, massless_bubble: FeynmanIntegral) -> None:
        A = _schwinger(massless_bubble).get_A_matrix()
        assert all(isinstance(x, sp.Integer) for x in A)
