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
    def test_returns_one_variable_fewer_than_propagators(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        _, new_vars = _schwinger(massless_triangle).dehomogenised_symanzik_polynomials()
        assert len(new_vars) == 2
        assert all(isinstance(v, sp.Symbol) for v in new_vars)

    def test_bubble_polynomials_set_last_parameter_to_one(
        self, massless_bubble: FeynmanIntegral
    ) -> None:
        (u_tilde, f_tilde), (u1,) = _schwinger(massless_bubble).dehomogenised_symanzik_polynomials()
        sym = massless_bubble.symanzik
        a1, a2 = sym.schwinger_parameters
        assert sp.expand(u_tilde - sym.u.subs({a1: u1, a2: 1})) == 0
        assert sp.expand(f_tilde - sym.f.subs({a1: u1, a2: 1})) == 0

    def test_polynomials_contain_only_new_variables(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        sch = _schwinger(massless_triangle)
        (u_tilde, f_tilde), new_vars = sch.dehomogenised_symanzik_polynomials()
        old = set(massless_triangle.symanzik.schwinger_parameters)
        assert not (u_tilde.free_symbols & old)
        assert not (f_tilde.free_symbols & old)
        assert set(new_vars) <= u_tilde.free_symbols | f_tilde.free_symbols

    def test_homogeneity_recovers_original_polynomials(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        """U(t u_1, ..., t u_{N-1}, t) = t^L U~(u) and F(...) = t^{L+1} F~(u)."""
        sch = _schwinger(massless_triangle)
        (u_tilde, f_tilde), new_vars = sch.dehomogenised_symanzik_polynomials()
        sym = massless_triangle.symanzik
        t = sp.Symbol("t", positive=True)
        alphas = sym.schwinger_parameters
        subs = {a: t * v for a, v in zip(alphas[:-1], new_vars, strict=True)}
        subs[alphas[-1]] = t
        L = massless_triangle.loop_count
        assert sp.expand(sym.u.subs(subs) - t**L * u_tilde) == 0
        assert sp.expand(sym.f.subs(subs) - t ** (L + 1) * f_tilde) == 0


class TestSchwingerAMatrix:
    def test_bubble_a_matrix_columns(self, massless_bubble: FeynmanIntegral) -> None:
        """U~ = u1 + 1 and F~ = c u1 give the 3 x 3 block matrix with two homogenising rows."""
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


class TestFacade:
    def test_schwinger_gkz_is_cached_and_typed(self, massless_triangle: FeynmanIntegral) -> None:
        from feynkit import CayleyGKZSystem

        sys_ = massless_triangle.schwinger_gkz
        assert isinstance(sys_, CayleyGKZSystem)
        assert massless_triangle.schwinger_gkz is sys_

    def test_matrix_agrees_with_get_a_matrix(self, massless_triangle: FeynmanIntegral) -> None:
        assert (
            massless_triangle.schwinger_gkz.a_matrix == _schwinger(massless_triangle).get_A_matrix()
        )

    def test_exponents_and_dimension_are_the_integral_s(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        sys_ = massless_triangle.schwinger_gkz
        edges = massless_triangle.graph.get_internal_edges()
        assert sys_.propagator_exponents == [
            massless_triangle.propagator_exponents[e.idx] for e in edges
        ]
        assert sys_.dimension == massless_triangle.dimension
        assert sys_.loop_count == massless_triangle.loop_count

    def test_prefactor_is_schwinger_prefactor_times_gamma(
        self, massless_triangle: FeynmanIntegral
    ) -> None:
        sys_ = massless_triangle.schwinger_gkz
        nu = sum(sys_.propagator_exponents)
        expected = massless_triangle.schwinger.prefactor * sp.gamma(
            nu - massless_triangle.dimension / 2
        )
        assert sp.simplify(sys_.prefactor - expected) == 0

    def test_schwinger_gkz_does_not_populate_cached_schwinger(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        _ = fi.schwinger_gkz
        assert "schwinger" not in fi.__dict__

    def test_sunrise_uses_loop_count_two(self) -> None:
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        sys_ = fi.schwinger_gkz
        nu = sum(sys_.propagator_exponents)
        half_d = fi.dimension / 2
        assert sp.simplify(sys_.beta_parameters[0] - (nu - 3 * half_d)) == 0
        assert sp.simplify(sys_.beta_parameters[1] - (2 * half_d - nu)) == 0
        assert sp.simplify(sys_.prefactor / fi.schwinger.prefactor - sp.gamma(nu - 2 * half_d)) == 0


class TestEquivalenceWithLeePomeransky:
    @pytest.mark.parametrize("cnickel", ["12e|2e|e|:zzz", "111e|e|:nnn"])
    def test_cayley_matrix_is_unimodular_image_of_lp_matrix(self, cnickel: str) -> None:
        """Klausen (2023, section 3.4): the UF Cayley form is unimodularly equivalent to A_LP.

        The map T from (ones, alpha_N, alpha_1, ..., alpha_{N-1}) to
        (r0, r1, alpha_1, ..., alpha_{N-1}) is block upper triangular: rows
        r0 and r1 take the 2 x 2 block [[L + 1, -1], [-L, 1]] on the (ones,
        alpha_N) columns and -1 (resp. 1) on every alpha_i column with
        i < N, since r0 and r1 depend on the full degree alpha_N + sum_{i <
        N} alpha_i, not on alpha_N alone; the remaining rows are the
        identity below that.
        """
        fi = FeynmanIntegral.from_cnickel(cnickel)
        lp, cay = fi.gkz, fi.schwinger_gkz
        n_lp = len(fi.symanzik.lp_parameters)
        L = fi.loop_count

        # Reorder LP columns to match the Cayley column order by monomial and total degree
        # (U~ columns have degree L, F~ columns have degree L + 1, so degree disambiguates
        # LP monomials that agree on their first N - 1 exponents).
        lp_index = {alpha: j for j, (alpha, _) in enumerate(lp.support)}
        blocks = [(e, L) for e, _ in cay.u_support] + [(e, L + 1) for e, _ in cay.f_support]
        lp_cols = []
        for e, degree in blocks:
            matches = [alpha for alpha in lp_index if alpha[:-1] == e and sum(alpha) == degree]
            assert len(matches) == 1, (e, degree, matches)
            lp_cols.append(lp_index[matches[0]])
        a_lp = lp.a_matrix.extract(list(range(lp.a_matrix.rows)), lp_cols)

        # Rows of a_lp: ones, alpha_1, ..., alpha_N. Build T on (ones, alpha_N, alpha_1..alpha_{N-1}).
        perm = [0, n_lp] + list(range(1, n_lp))
        a_perm = a_lp.extract(perm, list(range(a_lp.cols)))
        T = sp.eye(n_lp + 1)
        T[0, 0], T[0, 1], T[1, 0], T[1, 1] = L + 1, -1, -L, 1
        for j in range(2, n_lp + 1):
            T[0, j] = -1
            T[1, j] = 1
        assert T.det() == 1
        assert T * a_perm == cay.a_matrix

        # lp.beta_parameters is (-D/2, -nu_1, ..., -nu_N) in internal-edge order, so
        # index n_lp is -nu_N; permute it the same way as the columns above.
        beta_perm = sp.Matrix(
            [lp.beta_parameters[0], lp.beta_parameters[n_lp]] + list(lp.beta_parameters[1:n_lp])
        )
        mapped = T * beta_perm
        assert all(sp.simplify(mapped[i] - cay.beta_parameters[i]) == 0 for i in range(n_lp + 1))


class TestNumericalHomogeneity:
    def test_bubble_euler_integral_scales_with_beta(self) -> None:
        """Rescaling coefficients along a row of A multiplies Phi by lambda^beta_r.

        Massive bubble at D = 3, nu = (1, 1): Phi(w, z) = int_0^inf du
        (w1 u + w2)^(-1) (z1 u^2 + z2 u + z3)^(-1/2), which converges.
        Expected beta = (nu - 2 D/2, D/2 - nu, -nu_1) = (-1, -1/2, -1).
        """
        import warnings

        import numpy as np
        from scipy import integrate

        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        sys_ = fi.schwinger_gkz
        A = np.array(sys_.a_matrix.tolist(), dtype=float)
        n = len(sys_.u_support)
        u_exps = [e[0] for e, _ in sys_.u_support]
        f_exps = [e[0] for e, _ in sys_.f_support]
        D, nu = 3.0, (1.0, 1.0)
        subs = {fi.dimension: D}
        subs.update({fi.propagator_exponents[i]: nu[i - 1] for i in (1, 2)})
        expected = [float(b.subs(subs)) for b in sys_.beta_parameters]

        def phi(c: np.ndarray) -> float:
            w, z = c[:n], c[n:]

            def integrand(u: float) -> float:
                u_tilde = sum(wj * u**e for wj, e in zip(w, u_exps, strict=True))
                f_tilde = sum(zj * u**e for zj, e in zip(z, f_exps, strict=True))
                return (
                    u ** (nu[0] - 1)
                    * u_tilde ** (nu[0] + nu[1] - D)
                    * f_tilde ** (D / 2 - nu[0] - nu[1])
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", integrate.IntegrationWarning)
                return integrate.quad(integrand, 0, np.inf, epsrel=1e-9, limit=200)[0]

        c0 = np.array([1.1, 0.9, 1.3, 2.2, 0.7])
        lam = 1.5
        base = phi(c0)
        for r in range(A.shape[0]):
            measured = np.log(phi(c0 * lam ** A[r]) / base) / np.log(lam)
            assert abs(measured - expected[r]) < 1e-4, (r, measured, expected[r])
