"""
Parity tests for the FeynmanIntegral façade.

Verifies that every cached_property on FeynmanIntegral produces the same
mathematical object as the corresponding free-function pipeline. The parity
guarantee is the contract for commit 2 (which migrates callers) and commit 3
(which privatises the free functions): if it ever fails, the migration cannot
be safe.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit import (
    Edge,
    FeynmanIntegral,
    Graph,
    NewtonPolytope,
    PolytopeEquivalence,
    SymanzikPolynomials,
    ToricIdeal,
)
from feynkit.algebra import compute_toric_ideal_generators
from feynkit.kinematics import create_momentum_products
from feynkit.parametrisations.factory import _create_parametrisations
from feynkit.polynomials.symanzik import _calculate_symanzik_polynomials
from feynkit.systems.complete import _create_gkz_system
from feynkit.systems.monomial import extract_monomial_support

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture  # type: ignore[misc]
def bubble() -> Generator[Graph, None, None]:
    m1, m2 = sp.symbols("m1 m2", nonnegative=True)
    nu1, nu2 = sp.symbols("nu1 nu2", positive=True)
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    yield Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])


@pytest.fixture  # type: ignore[misc]
def integral(bubble: Graph) -> Generator[FeynmanIntegral, None, None]:
    yield FeynmanIntegral(bubble)


@pytest.fixture  # type: ignore[misc]
def free_pipeline(bubble: Graph):
    """Run the existing free-function pipeline with the same defaults the
    façade uses. Returns a dict of intermediate results."""
    D = sp.Symbol("D", positive=True)
    nus = {e.idx: sp.Symbol(f"nu_{e.idx}", positive=True) for e in bubble.get_internal_edges()}
    p_dot = create_momentum_products(n_external=bubble.external_legs, use_mandelstam=True)

    u, f = _calculate_symanzik_polynomials(bubble, p_dot)
    all_param = _create_parametrisations(bubble, D, bubble.get_loop_count(), nus, p_dot)
    lp = all_param.lee_pomeransky
    lp_params, _, u_lp, f_lp = lp._build_lp_substitution()
    g = sp.simplify(u_lp + f_lp)
    gkz = _create_gkz_system(g, lp_params, D, list(nus.values()))
    toric = compute_toric_ideal_generators(gkz.a_matrix)
    support = extract_monomial_support(g, lp_params)

    return {
        "D": D,
        "nus": nus,
        "p_dot": p_dot,
        "u": u,
        "f": f,
        "u_lp": u_lp,
        "f_lp": f_lp,
        "g": g,
        "lp_params": lp_params,
        "all_param": all_param,
        "gkz": gkz,
        "toric": toric,
        "support": support,
    }


def _expr_eq(a: sp.Expr, b: sp.Expr) -> bool:
    """Symbolic equality: a == b iff sp.simplify(a - b) is zero."""
    return sp.simplify(a - b) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_defaults_are_filled_in(self, bubble: Graph) -> None:
        integral = FeynmanIntegral(bubble)

        assert integral.graph is bubble
        assert integral.loop_count == bubble.get_loop_count()
        assert isinstance(integral.dimension, sp.Symbol)
        assert integral.dimension.name == "D"
        assert set(integral.propagator_exponents) == {1, 2}
        for nu in integral.propagator_exponents.values():
            assert nu.is_positive

    def test_explicit_overrides_take_precedence(self, bubble: Graph) -> None:
        D = sp.Symbol("D_user", positive=True)
        nus = {e.idx: sp.Integer(1) for e in bubble.get_internal_edges()}
        integral = FeynmanIntegral(bubble, dimension=D, propagator_exponents=nus)
        assert integral.dimension is D
        assert integral.propagator_exponents == nus

    def test_loop_count_mismatch_rejected(self, bubble: Graph) -> None:
        from feynkit.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            FeynmanIntegral(bubble, loop_count=999)


# ──────────────────────────────────────────────────────────────────────────────
# Polynomials
# ──────────────────────────────────────────────────────────────────────────────


class TestSymanzikParity:
    def test_symanzik_is_value_type(self, integral: FeynmanIntegral) -> None:
        assert isinstance(integral.symanzik, SymanzikPolynomials)

    def test_u_matches_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        assert _expr_eq(integral.symanzik.u, free_pipeline["u"])

    def test_f_matches_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        assert _expr_eq(integral.symanzik.f, free_pipeline["f"])

    def test_g_matches_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        assert _expr_eq(integral.symanzik.g, free_pipeline["g"])

    def test_lp_substitution_matches_free_function(
        self, integral: FeynmanIntegral, free_pipeline
    ) -> None:
        assert _expr_eq(integral.symanzik.u_lp, free_pipeline["u_lp"])
        assert _expr_eq(integral.symanzik.f_lp, free_pipeline["f_lp"])
        assert list(integral.symanzik.lp_parameters) == list(free_pipeline["lp_params"])


# ──────────────────────────────────────────────────────────────────────────────
# Parametrisations
# ──────────────────────────────────────────────────────────────────────────────


class TestParametrisationParity:
    def test_schwinger_matches_free_function(
        self, integral: FeynmanIntegral, free_pipeline
    ) -> None:
        expected = free_pipeline["all_param"].schwinger.compute()
        assert integral.schwinger.name == expected.name
        assert _expr_eq(integral.schwinger.prefactor, expected.prefactor)
        assert _expr_eq(integral.schwinger.measure, expected.measure)
        assert _expr_eq(integral.schwinger.integrand, expected.integrand)

    def test_feynman_matches_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        expected = free_pipeline["all_param"].feynman.compute()
        assert integral.feynman.name == expected.name
        assert _expr_eq(integral.feynman.prefactor, expected.prefactor)
        assert _expr_eq(integral.feynman.measure, expected.measure)
        assert _expr_eq(integral.feynman.integrand, expected.integrand)

    def test_lee_pomeransky_matches_free_function(
        self, integral: FeynmanIntegral, free_pipeline
    ) -> None:
        expected = free_pipeline["all_param"].lee_pomeransky.compute()
        assert integral.lee_pomeransky.name == expected.name
        assert _expr_eq(integral.lee_pomeransky.prefactor, expected.prefactor)
        assert _expr_eq(integral.lee_pomeransky.measure, expected.measure)
        assert _expr_eq(integral.lee_pomeransky.integrand, expected.integrand)


# ──────────────────────────────────────────────────────────────────────────────
# Algebraic-geometry representations
# ──────────────────────────────────────────────────────────────────────────────


class TestGKZParity:
    def test_a_matrix_matches_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        def col_set(M: sp.Matrix) -> frozenset[tuple[int, ...]]:
            return frozenset(tuple(int(M[r, j]) for r in range(M.rows)) for j in range(M.cols))

        assert col_set(integral.gkz.a_matrix) == col_set(free_pipeline["gkz"].a_matrix)

    def test_beta_matches_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        for a, b in zip(
            integral.gkz.beta_parameters, free_pipeline["gkz"].beta_parameters, strict=True
        ):
            assert _expr_eq(a, b)

    def test_support_size_matches_free_function(
        self, integral: FeynmanIntegral, free_pipeline
    ) -> None:
        assert len(integral.gkz.support) == len(free_pipeline["gkz"].support)


class TestNewtonPolytope:
    def test_is_value_type(self, integral: FeynmanIntegral) -> None:
        assert isinstance(integral.newton_polytope, NewtonPolytope)

    def test_support_matches_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        from_facade = sorted([e for e, _ in integral.newton_polytope.support])
        from_free = sorted([e for e, _ in free_pipeline["support"]])
        assert from_facade == from_free

    def test_a_matrix_consistent_with_gkz(self, integral: FeynmanIntegral) -> None:
        def col_set(M: sp.Matrix) -> frozenset[tuple[int, ...]]:
            return frozenset(tuple(int(M[r, j]) for r in range(M.rows)) for j in range(M.cols))

        assert col_set(integral.newton_polytope.a_matrix) == col_set(integral.gkz.a_matrix)

    def test_points_and_coefficients_split(self, integral: FeynmanIntegral) -> None:
        np_obj = integral.newton_polytope
        assert len(np_obj.points) == len(np_obj.support)
        assert len(np_obj.coefficients) == len(np_obj.support)


class TestToricIdealParity:
    def test_is_value_type(self, integral: FeynmanIntegral) -> None:
        assert isinstance(integral.toric_ideal, ToricIdeal)

    def test_generators_match_free_function(self, integral: FeynmanIntegral, free_pipeline) -> None:
        # Generators may come back in a different SymPy ordering; compare as
        # sets of expanded expressions.
        from_facade = {sp.expand(g) for g in integral.toric_ideal.generators}
        from_free = {sp.expand(g) for g in free_pipeline["toric"]}
        assert from_facade == from_free


# ──────────────────────────────────────────────────────────────────────────────
# Caching, immutability, and derivation
# ──────────────────────────────────────────────────────────────────────────────


class TestCachingAndImmutability:
    def test_same_property_returns_same_object(self, integral: FeynmanIntegral) -> None:
        assert integral.symanzik is integral.symanzik
        assert integral.gkz is integral.gkz
        assert integral.toric_ideal is integral.toric_ideal

    def test_with_returns_new_instance(self, integral: FeynmanIntegral) -> None:
        derived = integral.with_(dimension=sp.Integer(4))
        assert derived is not integral
        assert derived.dimension == sp.Integer(4)
        # original unchanged
        assert integral.dimension.name == "D"

    def test_with_invalidates_cache(self, integral: FeynmanIntegral) -> None:
        # Force a cache fill.
        gkz_orig = integral.gkz
        derived = integral.with_(dimension=sp.Integer(4))
        # New object → new cache → new GKZ instance with the new D in beta.
        assert derived.gkz is not gkz_orig
        # The dimension field on the GKZ system should reflect the override.
        assert derived.gkz.dimension == sp.Integer(4)

    def test_propagator_exponents_view_is_a_copy(self, integral: FeynmanIntegral) -> None:
        view = integral.propagator_exponents
        view[999] = sp.Integer(0)
        assert 999 not in integral.propagator_exponents


class TestLaziness:
    def test_construction_does_not_compute(self, bubble: Graph) -> None:
        integral = FeynmanIntegral(bubble)
        # No cached_property should be populated yet.
        for name in ("_all_parametrisations", "symanzik", "gkz", "toric_ideal"):
            assert name not in integral.__dict__


# ──────────────────────────────────────────────────────────────────────────────
# Comparison verbs
# ──────────────────────────────────────────────────────────────────────────────


class TestEquivalenceVerbs:
    def test_unimodular_self_equivalent(self, integral: FeynmanIntegral, bubble: Graph) -> None:
        other = FeynmanIntegral(bubble)
        result = integral.is_unimodular_equivalent_to(other)
        assert isinstance(result, PolytopeEquivalence)
        assert result.relation == "unimodular"
        assert result.equivalent is True
        assert result.witness_map is not None
        assert result.vertex_correspondence is not None

    def test_affine_self_equivalent(self, integral: FeynmanIntegral, bubble: Graph) -> None:
        other = FeynmanIntegral(bubble)
        result = integral.is_affinely_equivalent_to(other)
        assert isinstance(result, PolytopeEquivalence)
        assert result.relation == "affine_polytope"
        assert result.equivalent is True
