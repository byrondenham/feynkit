"""
Parity tests for the FeynmanIntegral facade.

Verifies that every cached_property on FeynmanIntegral produces the same
mathematical object as the corresponding free-function pipeline. The parity
guarantee is the contract for commit 2 (which migrates callers) and commit 3
(which privatises the free functions): if it ever fails, the migration cannot
be safe.
"""

from __future__ import annotations

import itertools
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
    ValidationError,
    generate_graphs,
    polytope_data,
)
from feynkit.algebra import compute_toric_ideal_generators
from feynkit.kinematics import create_momentum_products
from feynkit.lattice_invariants import lattice_invariants
from feynkit.parametrisations.factory import _create_parametrisations
from feynkit.polynomials.symanzik import _calculate_symanzik_polynomials
from feynkit.systems.complete import _create_gkz_system
from feynkit.systems.monomial import extract_monomial_support

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


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
    facade uses. Returns a dict of intermediate results."""
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


# ------------------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Polynomials
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Parametrisations
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Algebraic-geometry representations
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Caching, immutability, and derivation
# ------------------------------------------------------------------------------


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
        # New object -> new cache -> new GKZ instance with the new D in beta.
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


# ------------------------------------------------------------------------------
# Comparison verbs
# ------------------------------------------------------------------------------


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


# -- Scaleless integrals ---------------------------------------------------------


def _special(cnickel: str, values: dict[str, sp.Expr]) -> FeynmanIntegral:
    """The integral of the string with the named invariants replaced by the given values."""
    legs = cnickel.split(":")[0].count("e")
    fi = FeynmanIntegral.from_cnickel(cnickel, use_mandelstam=legs >= 2)
    products = fi.momentum_products
    symbols = {s.name: s for v in products.values() for s in sp.sympify(v).free_symbols}
    substitution = {symbols[name]: value for name, value in values.items()}
    return fi.with_(
        momentum_products={
            k: sp.expand(sp.sympify(v).subs(substitution)) for k, v in products.items()
        }
    )


BOX_LEGS = {"p1^2": 0, "p2^2": 0, "p3^2": 0, "p4^2": 0}


class TestIsScaleless:
    """Lee's criterion: the origin does not lie in the affine hull of the support of G."""

    @pytest.mark.parametrize(
        ("cnickel", "values", "scaleless", "full_dimensional"),
        [
            ("0|:z", {}, True, False),
            ("01e|e|:zn", {}, True, False),
            ("00|:nz", {}, True, False),
            ("00|:zz", {}, True, False),
            ("012e|2e|e|:zzzz", {}, True, False),
            ("012e|2e|e|:znnn", {}, True, False),
            ("11e|e|:zz", {"s": 0}, True, False),
            ("111e|e|:zzz", {"s": 0}, True, False),
            ("12e|2e|e|:zzz", {"p1^2": 0, "p2^2": 0, "p3^2": 0}, True, False),
            ("12e|3e|3e|e|:zzzz", {**BOX_LEGS, "s12": 0, "s23": 0}, True, False),
            # Every equation of the affine hull passes through the origin: the integral
            # converges nowhere, but the rescaling does not involve D.
            ("1ee|1|:zn", {}, False, False),
            ("01e|e|:nz", {"s": 0}, False, False),
            ("0|:n", {}, False, True),
            ("01e|e|:nz", {}, False, True),
            ("00|:nn", {}, False, True),
            ("11e|e|:zz", {}, False, True),
            ("11e|e|:nz", {"s": 0}, False, True),
            ("12e|2e|e|:zzz", {"p1^2": 0, "p2^2": 0}, False, True),
            ("12e|3e|3e|e|:zzzz", BOX_LEGS, False, True),
        ],
    )
    def test_criterion(
        self, cnickel: str, values: dict[str, sp.Expr], scaleless: bool, full_dimensional: bool
    ) -> None:
        fi = _special(cnickel, values)
        data = polytope_data(fi.newton_polytope.points)
        assert fi.is_scaleless is scaleless
        assert data.is_full_dimensional is full_dimensional
        assert fi.is_scaleless == any(row[0] != 0 for row in data.affine_hull)

    def test_massive_bubble_on_shell(self) -> None:
        m_1 = FeynmanIntegral.from_cnickel("11e|e|:nn").graph.get_internal_edges()[0].get_mass()
        fi = _special("11e|e|:nn", {"s": m_1**2})
        assert not fi.is_scaleless
        assert polytope_data(fi.newton_polytope.points).is_full_dimensional

    def test_agrees_with_the_affine_hull(self) -> None:
        pool = [
            *generate_graphs(1, range(1, 5), self_loops=True),
            *generate_graphs(2, range(1, 4), edges=range(1, 6), self_loops=True),
        ]
        scaleless = 0
        for cnickel in pool:
            fi = _special(cnickel, {})
            data = polytope_data(fi.newton_polytope.points)
            assert fi.is_scaleless == any(row[0] != 0 for row in data.affine_hull), cnickel
            assert not (fi.is_scaleless and data.is_full_dimensional), cnickel
            scaleless += fi.is_scaleless
        assert 0 < scaleless < len(pool)


# -- Lattice invariants ----------------------------------------------------------


class TestLatticeInvariants:
    def test_the_module_on_the_newton_points(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zznz")
        points = [tuple(int(x) for x in p) for p in fi.newton_polytope.points]
        for lattice in ("support", "ambient"):
            expected = lattice_invariants(points, lattice=lattice, backend="python")
            assert fi.lattice_invariants(lattice) == expected
            assert fi.lattice_invariants(lattice, backend="python") == expected

    def test_the_massless_triangle_is_a_cube_without_two_corners(self) -> None:
        # U = u_1 + u_2 + u_3 and F has the three products u_i u_j: the points of {0, 1}^3
        # other than 0 and (1, 1, 1).
        corners = [p for p in itertools.product((0, 1), repeat=3) if 0 < sum(p) < 3]
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        assert fi.lattice_invariants() == lattice_invariants(corners)

    def test_a_massless_self_loop_lifts_the_polytope(self) -> None:
        # G is u_0 times the G of the triangle, so P is the polytope of the triangle, lifted
        # into one more dimension, with the same chart.
        lifted = FeynmanIntegral.from_cnickel("012e|2e|e|:zzzz")
        assert lifted.is_scaleless
        assert (
            lifted.lattice_invariants()
            == FeynmanIntegral.from_cnickel("12e|2e|e|:zzz").lattice_invariants()
        )

    def test_is_cached_per_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import feynkit.lattice_invariants as module

        calls: list[dict[str, object]] = []
        original = module.lattice_invariants

        def counting(*args: object, **kwargs: object) -> object:
            calls.append(kwargs)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(module, "lattice_invariants", counting)
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        first = fi.lattice_invariants()
        assert fi.lattice_invariants() is first
        assert len(calls) == 1
        fi.lattice_invariants("ambient")
        fi.lattice_invariants(backend="python", budget=1)
        fi.lattice_invariants(backend="python", budget=1)
        assert len(calls) == 3
        assert fi.with_(dimension=sp.Integer(4)).lattice_invariants() == first
        assert len(calls) == 4

    def test_the_budget_and_timeout_reach_the_module(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        found = fi.lattice_invariants(backend="python", budget=1, timeout=5)
        assert (found.h_star, found.idp, found.normal) == (None, None, None)
        assert found.lattice_points == fi.lattice_invariants().lattice_points


class TestReportLimits:
    def test_limits_reach_the_report(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        p1 = sp.Symbol("p1^2", real=True)
        fi = fi.with_(
            momentum_products={
                k: sp.expand(sp.sympify(v).subs(p1, 0)) for k, v in fi.momentum_products.items()
            }
        )
        marker = "each is a limit surface"
        for render in (fi.to_text, fi.to_latex):
            assert marker in " ".join(render(["landau"], limits=True).split())
            assert marker not in " ".join(render(["landau"], limits=False).split())


# -- Facet resonance ---------------------------------------------------------------

RESONANCE_GRAPHS = [
    "11e|e|:nn",
    "11e|e|:nz",
    "11e|e|:zz",
    "12e|2e|e|:nnn",
    "12e|2e|e|:zzz",
    "12e|3e|3e|e|:nnnn",
    "111e|e|:nnn",
    "111e|e|:nnz",
    "111e|e|:nzz",
    "1111e|e|:nnnn",
]


def _unit(fi: FeynmanIntegral) -> dict[int, int]:
    return dict.fromkeys(fi.propagator_exponents, 1)


class TestFacetResonance:
    @pytest.mark.parametrize("cnickel", RESONANCE_GRAPHS)
    def test_gkz_and_schwinger_agree(self, cnickel: str) -> None:
        # The Cayley facets come from the Cayley columns; T only carries beta across, and
        # carries the Lee-Pomeransky columns to the Cayley ones.
        from feynkit.systems.cayley import lp_to_cayley

        fi = FeynmanIntegral.from_cnickel(cnickel)
        n = len(fi.symanzik.lp_parameters)
        a, cayley = fi.gkz.a_matrix, fi.schwinger_gkz.a_matrix
        t = lp_to_cayley(n, fi.loop_count)
        position = {tuple(int(x) for x in cayley[:, j]): j for j in range(cayley.cols)}
        image = {j: position[tuple(int(x) for x in t * a[:, j])] for j in range(a.cols)}
        for d0, powers in ((4, _unit(fi)), (3, {e: 2 - e % 2 for e in fi.propagator_exponents})):
            gkz = fi.facet_resonance(d0, nu=powers)
            schwinger = {
                frozenset(r.facet.point_indices): r
                for r in fi.facet_resonance(d0, nu=powers, system="schwinger")
            }
            assert len(schwinger) == len(gkz)
            for record in gkz:
                other = schwinger[frozenset(image[j] for j in record.facet.point_indices)]
                assert other.form == record.form
                assert (other.resonant, other.admissible) == (record.resonant, record.admissible)
                assert (other.columns_off, other.reducible) == (
                    record.columns_off,
                    record.reducible,
                )

    def test_point_indices_are_the_gkz_columns(self) -> None:
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnz")
        support = [alpha for alpha, _ in fi.gkz.support]
        for record in fi.facet_resonance(nu=_unit(fi)):
            m, b = record.facet.normal, record.facet.offset
            on = {j for j, alpha in enumerate(support) if sum(map(int.__mul__, m, alpha)) == b}
            assert on == set(record.facet.point_indices)

    def test_the_integral_s_own_integer_exponents(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
        powers = {e: 1 + e % 2 for e in fi.propagator_exponents}
        fixed = fi.with_(propagator_exponents={e: sp.Integer(k) for e, k in powers.items()})
        assert fixed.facet_resonance() == fi.facet_resonance(nu=powers)
        assert fixed.facet_resonance(3) == fi.facet_resonance(3, nu=powers)

    def test_symbolic_exponents_need_nu(self) -> None:
        with pytest.raises(ValidationError, match="nu"):
            FeynmanIntegral.from_cnickel("11e|e|:nn").facet_resonance()

    def test_bad_arguments(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        with pytest.raises(ValidationError):
            fi.facet_resonance(nu={1: 1})
        with pytest.raises(ValidationError):
            fi.facet_resonance(nu=_unit(fi), system="feynman")
        with pytest.raises(ValidationError):
            fi.facet_resonance(4.0, nu=_unit(fi))  # type: ignore[arg-type]

    def test_f_u_is_admissible_where_the_f_block_restriction_is_silent(self) -> None:
        # F_U, the facet x_1 + ... + x_N <= L + 1 of the monomials of F, is admissible at
        # one eps, where nu = (L+1) D/2; restrict_to_f_block warns everywhere else.
        import warnings

        for cnickel in ("11e|e|:nn", "111e|e|:nnn", "12e|2e|e|:nnn"):
            fi = FeynmanIntegral.from_cnickel(cnickel)
            n = len(fi.symanzik.lp_parameters)
            f_u = next(
                r
                for r in fi.facet_resonance(nu=_unit(fi))
                if r.facet.normal == (1,) * n and r.facet.offset == fi.loop_count + 1
            )
            assert f_u.admissible.kind == "point" and f_u.admissible.offset is not None
            eps = sp.Rational(f_u.admissible.offset)
            powers = {e: sp.Integer(1) for e in fi.propagator_exponents}
            at = fi.with_(dimension=4 - 2 * eps, propagator_exponents=powers)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                at.schwinger_gkz.restrict_to_f_block()
            off = fi.with_(dimension=4 - 2 * (eps + 1), propagator_exponents=powers)
            with pytest.warns(UserWarning, match="does not lie in the span"):
                off.schwinger_gkz.restrict_to_f_block()


class TestD0FromTheDimension:
    def test_six_minus_two_epsilon(self) -> None:
        # D = 6 - 2 eps: F_F of the bubble, l_F(beta) = D/2 - nu_1 - nu_2, is admissible at
        # eps = 1, where D = 4, and resonant on Z.
        eps = sp.Symbol("epsilon")
        bubble = FeynmanIntegral.from_cnickel("11e|e|:nn")
        powers = dict.fromkeys(bubble.propagator_exponents, sp.Integer(1))
        fi = bubble.with_(dimension=6 - 2 * eps, propagator_exponents=powers)
        for system in ("gkz", "schwinger"):
            records = fi.facet_resonance(system=system)
            assert records == fi.facet_resonance(6, system=system)
        f_f = next(r for r in fi.facet_resonance() if r.facet.offset == -1)
        assert f_f.admissible.offset == 1
        assert "D_0 = 6, read from the dimension of the integral" in " ".join(
            fi.to_text(["resonance"]).split()
        )

    def test_falls_back_to_four(self) -> None:
        bubble = FeynmanIntegral.from_cnickel("11e|e|:nn")
        unit = dict.fromkeys(bubble.propagator_exponents, 1)
        assert bubble.facet_resonance(nu=unit) == bubble.facet_resonance(4, nu=unit)
        other = bubble.with_(dimension=4 - 2 * sp.Symbol("delta"))
        assert other.facet_resonance(nu=unit) == bubble.facet_resonance(4, nu=unit)
        assert "D_0 = 4 by default" in " ".join(other.to_text(["resonance"]).split())


class TestReportD0:
    def test_d0_reaches_the_report(self) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        for render in (fi.to_text, fi.to_latex):
            assert "D_0 = 3" in " ".join(render(["resonance"], d0=3).split())


class TestFaceIdentification:
    def test_the_accessor_caches_identify_faces(self) -> None:
        from feynkit import identify_faces

        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        first = fi.face_identification()
        assert first == identify_faces(fi)
        assert fi.face_identification() is first
        every = fi.face_identification(None)
        assert every == identify_faces(fi, max_codimension=None)
        assert len(every) > len(first)
        assert fi.with_(dimension=sp.Integer(6)).face_identification() is not first

    @pytest.mark.parametrize("bad", [-1, [2], 2.0])
    def test_a_bad_codimension_is_rejected(self, bad: object) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:nn")
        with pytest.raises(ValidationError, match="max_codimension"):
            fi.face_identification(bad)  # type: ignore[arg-type]
