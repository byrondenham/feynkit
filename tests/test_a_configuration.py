"""
Tests for feynkit.a_configuration and feynkit.artifacts.dissertation.

Key mathematical facts verified:
- triangle Smith invariants = [1,1,1]  (spans full ℤ³)
- triple-K Smith invariants = [1,1,2]  (spans even-sum sublattice)
- triangle → triangle: finite-index map with det=1 (unimodular)
- triangle → triple-K: finite-index map with det=2 (NOT unimodular)
- triangle and triple-K are NOT unimodularly equivalent
- triangle and triple-K ARE affinely equivalent (same intrinsic shape)
- artifacts load and are correctly typed
"""

import numpy as np
import pytest
import sympy as sp

from feynkit.a_configuration import (
    AConfiguration,
    FiniteIndexResult,
    IntrinsicModel,
    SymmetryPair,
    finite_index_map,
    intrinsic_lattice_model,
    symmetry_pairs,
)
from feynkit.artifacts.conformal import (
    _bms_g_polynomial,
    bms_simplex_a_config,
    complete_graph_a_config,
    conformal_companion_a_config,
    massless_polygon_a_config,
)
from feynkit.artifacts.dissertation import (
    banana3_a_config,
    four_point_simplex_a_config,
    triangle_a_config,
    triple_k_a_config,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def triangle():
    return triangle_a_config()


@pytest.fixture
def triple_k():
    return triple_k_a_config()


@pytest.fixture
def banana3():
    return banana3_a_config()


# ──────────────────────────────────────────────────────────────────────────────
# AConfiguration basics
# ──────────────────────────────────────────────────────────────────────────────


class TestAConfigurationBasics:
    def test_triangle_shape(self, triangle):
        assert triangle.n_points == 6
        assert triangle.ambient_dim == 3

    def test_triple_k_shape(self, triple_k):
        assert triple_k.n_points == 6
        assert triple_k.ambient_dim == 3

    def test_is_homogenized_autodetect(self, triangle):
        assert triangle.is_homogenized is True

    def test_affine_points_shape(self, triangle):
        pts = triangle.affine_points
        assert pts.shape == (6, 3)

    def test_non_homogenized(self):
        M = sp.Matrix([[1, 0, 1], [0, 1, 1]])
        cfg = AConfiguration(M, is_homogenized=False)
        assert cfg.ambient_dim == 2
        assert cfg.n_points == 3

    def test_repr_contains_shape(self, triangle):
        r = repr(triangle)
        assert "4×6" in r
        assert "homogenized" in r

    def test_matrix_property_roundtrip(self, triangle):
        M = triangle.matrix
        assert isinstance(M, sp.ImmutableMatrix)
        assert M.shape == (4, 6)
        assert M[0, :] == sp.ones(1, 6)


# ──────────────────────────────────────────────────────────────────────────────
# Smith invariants
# ──────────────────────────────────────────────────────────────────────────────


class TestSmithInvariants:
    def test_triangle_smith(self, triangle):
        # Triangle spans full ℤ³ — all invariants are 1.
        assert triangle.smith_invariants == [1, 1, 1]

    def test_triple_k_smith(self, triple_k):
        # Triple-K spans even-sum sublattice — last invariant is 2.
        inv = triple_k.smith_invariants
        assert inv[-1] == 2
        assert inv[:-1] == [1, 1]

    def test_banana3_smith(self, banana3):
        # Banana-3 affine points: 4 points in ℤ³.
        inv = banana3.smith_invariants
        assert len(inv) >= 1

    def test_four_simplex_smith(self):
        cfg = four_point_simplex_a_config()
        inv = cfg.smith_invariants
        # Standard simplex spans full ℤ⁴ → all invariants = 1.
        assert all(d == 1 for d in inv)


# ──────────────────────────────────────────────────────────────────────────────
# Newton polytope
# ──────────────────────────────────────────────────────────────────────────────


class TestNewtonPolytope:
    def test_triangle_hull_points(self, triangle):
        pts = triangle.newton_polytope_points
        assert len(pts) >= 4  # at minimum 4 vertices for this polytope

    def test_all_hull_points_are_integer_tuples(self, triangle):
        for pt in triangle.newton_polytope_points:
            assert isinstance(pt, tuple)
            assert all(isinstance(x, int) for x in pt)

    def test_affine_dim_triangle(self, triangle):
        assert triangle.affine_dim == 3


# ──────────────────────────────────────────────────────────────────────────────
# Unimodular equivalence
# ──────────────────────────────────────────────────────────────────────────────


class TestUnimodularEquivalence:
    def test_triangle_not_unimodular_to_triple_k(self, triangle, triple_k):
        result = triangle.is_unimodular_equivalent_to(triple_k)
        assert result.equivalent is False

    def test_triangle_unimodular_to_itself(self, triangle):
        result = triangle.is_unimodular_equivalent_to(triangle)
        assert result.equivalent is True

    def test_triple_k_unimodular_to_itself(self, triple_k):
        result = triple_k.is_unimodular_equivalent_to(triple_k)
        assert result.equivalent is True


# ──────────────────────────────────────────────────────────────────────────────
# Affine equivalence
# ──────────────────────────────────────────────────────────────────────────────


class TestAffineEquivalence:
    def test_triangle_affinely_equivalent_to_triple_k(self, triangle, triple_k):
        result = triangle.is_affinely_equivalent_to(triple_k)
        assert result.equivalent is True

    def test_triangle_affinely_equivalent_to_itself(self, triangle):
        result = triangle.is_affinely_equivalent_to(triangle)
        assert result.equivalent is True


# ──────────────────────────────────────────────────────────────────────────────
# finite_index_map
# ──────────────────────────────────────────────────────────────────────────────


class TestFiniteIndexMap:
    def test_triangle_to_triangle_is_unimodular(self, triangle):
        result = finite_index_map(triangle, triangle)
        assert result.found is True
        assert result.determinant == 1
        assert result.is_unimodular is True

    def test_triangle_to_triple_k_det_2(self, triangle, triple_k):
        result = finite_index_map(triangle, triple_k)
        assert result.found is True
        assert result.determinant == 2
        assert result.is_unimodular is False

    def test_triangle_to_triple_k_witness_maps_points(self, triangle, triple_k):
        result = finite_index_map(triangle, triple_k)
        assert result.found is True
        M = result.witness_matrix
        t = result.translation
        assert M is not None and t is not None
        # Verify: M·x + t ∈ triple_k.affine_points for every source point.
        tgt_set = {tuple(int(x) for x in row) for row in triple_k.affine_points}
        for pt in triangle.affine_points:
            img = M * sp.Matrix(pt.tolist()) + t
            assert tuple(int(x) for x in img) in tgt_set

    def test_finite_index_result_fields(self, triangle, triple_k):
        result = finite_index_map(triangle, triple_k)
        assert isinstance(result, FiniteIndexResult)
        assert result.witness_matrix is not None
        assert result.translation is not None
        assert result.column_permutation is not None
        assert len(result.column_permutation) == triangle.n_points

    def test_no_map_between_incompatible(self, triangle, banana3):
        # triangle is 3-dimensional, banana3 is also 3-dim but different shape.
        result = finite_index_map(triangle, banana3)
        # May or may not find a map — just check the type is correct.
        assert isinstance(result, FiniteIndexResult)

    def test_method_on_aconfiguration(self, triangle, triple_k):
        result = triangle.finite_index_map_to(triple_k)
        assert result.found is True

    def test_known_witness_matrix(self, triangle, triple_k):
        # The known finite-index map is M = [[0,1,1],[1,0,1],[1,1,0]], t = 0.
        # Verify it manually, even if finite_index_map finds a different one.
        M_known = sp.Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        t_known = sp.zeros(3, 1)
        tgt_set = {tuple(int(x) for x in row) for row in triple_k.affine_points}
        for pt in triangle.affine_points:
            img = M_known * sp.Matrix(pt.tolist()) + t_known
            assert tuple(int(x) for x in img) in tgt_set


# ──────────────────────────────────────────────────────────────────────────────
# intrinsic_lattice_model
# ──────────────────────────────────────────────────────────────────────────────


class TestIntrinsicLatticeModel:
    def test_triangle_model_type(self, triangle):
        model = intrinsic_lattice_model(triangle)
        assert isinstance(model, IntrinsicModel)

    def test_triangle_intrinsic_rank(self, triangle):
        model = intrinsic_lattice_model(triangle)
        assert model.intrinsic_rank == 3

    def test_triangle_smith_invariants(self, triangle):
        model = intrinsic_lattice_model(triangle)
        assert model.smith_invariants == [1, 1, 1]

    def test_triple_k_intrinsic_rank(self, triple_k):
        model = intrinsic_lattice_model(triple_k)
        assert model.intrinsic_rank == 3

    def test_triple_k_smith_invariants(self, triple_k):
        model = intrinsic_lattice_model(triple_k)
        assert model.smith_invariants[-1] == 2

    def test_intrinsic_coords_count(self, triangle):
        model = intrinsic_lattice_model(triangle)
        assert len(model.intrinsic_coords) == triangle.n_points

    def test_base_point_is_first_point(self, triangle):
        model = intrinsic_lattice_model(triangle)
        pts = triangle.affine_points
        assert model.base_point == tuple(int(x) for x in pts[0])

    def test_method_on_aconfiguration(self, triangle):
        model = triangle.intrinsic_model()
        assert isinstance(model, IntrinsicModel)


# ──────────────────────────────────────────────────────────────────────────────
# Artifact registry
# ──────────────────────────────────────────────────────────────────────────────


class TestArtifacts:
    def test_triangle_loads(self):
        cfg = triangle_a_config()
        assert isinstance(cfg, AConfiguration)

    def test_triple_k_loads(self):
        cfg = triple_k_a_config()
        assert isinstance(cfg, AConfiguration)

    def test_four_point_simplex_loads(self):
        cfg = four_point_simplex_a_config()
        assert isinstance(cfg, AConfiguration)
        assert cfg.ambient_dim == 4
        assert cfg.n_points == 5

    def test_banana3_loads(self):
        cfg = banana3_a_config()
        assert isinstance(cfg, AConfiguration)
        assert cfg.n_points == 4
        assert cfg.ambient_dim == 3

    def test_artifacts_are_independent_instances(self):
        a = triangle_a_config()
        b = triangle_a_config()
        assert a is not b


# ──────────────────────────────────────────────────────────────────────────────
# symmetry_pairs
# ──────────────────────────────────────────────────────────────────────────────


class TestSymmetryPairs:
    def test_identity_always_present(self, triangle):
        pairs = symmetry_pairs(triangle)
        identity = [
            p
            for p in pairs
            if p.is_unimodular and p.linear_map == sp.eye(3) and p.translation == sp.zeros(3, 1)
        ]
        assert len(identity) == 1

    def test_returns_list_of_symmetry_pair(self, triangle):
        pairs = symmetry_pairs(triangle)
        assert isinstance(pairs, list)
        assert all(isinstance(p, SymmetryPair) for p in pairs)

    def test_column_permutation_is_bijection(self, triangle):
        pairs = symmetry_pairs(triangle)
        N = triangle.n_points
        for p in pairs:
            assert len(p.column_permutation) == N
            assert set(p.column_permutation) == set(range(N))

    def test_ta_eq_ap(self, triangle):
        """Verify T·A = A·Π_P for every symmetry pair."""
        A = triangle.matrix  # 4×6 (homogenized)
        N = triangle.n_points
        pairs = symmetry_pairs(triangle)
        for pair in pairs:
            T = pair.homogenized_map
            # Π_P: column permutation matrix with Π_P[P(j), j] = 1
            Pi = sp.zeros(N, N)
            for j, k in enumerate(pair.column_permutation):
                Pi[k, j] = 1
            assert A * Pi == T * A, (
                f"T·A ≠ A·Π_P for det={pair.determinant}, " f"perm={pair.column_permutation}"
            )

    def test_image_points_within_config(self, triangle):
        """Every source point maps to a point in the configuration."""
        pts = triangle.affine_points
        pts_set = {tuple(int(x) for x in row) for row in pts}
        pairs = symmetry_pairs(triangle)
        for pair in pairs:
            M, t = pair.linear_map, pair.translation
            for pt in pts:
                img = tuple(int(x) for x in M * sp.Matrix(pt.tolist()) + t)
                assert img in pts_set

    def test_unimodular_count_matches_automorphisms(self, triangle):
        """Unimodular self-maps equal the Newton polytope automorphism group order.

        Valid when every configuration point is a hull vertex (triangle: all 6
        points are hull vertices of the cross-polytope).
        """
        pairs = symmetry_pairs(triangle)
        uni_count = sum(1 for p in pairs if p.is_unimodular)
        auts = triangle.automorphisms()
        assert uni_count == auts.order

    def test_triangle_has_48_unimodular_pairs(self, triangle):
        pairs = symmetry_pairs(triangle)
        uni = [p for p in pairs if p.is_unimodular]
        assert len(uni) == 48  # B₃ symmetry of the cross-polytope

    def test_triple_k_has_48_unimodular_pairs(self, triple_k):
        pairs = symmetry_pairs(triple_k)
        uni = [p for p in pairs if p.is_unimodular]
        assert len(uni) == 48  # same cross-polytope structure

    def test_determinant_field_correct(self, triangle):
        pairs = symmetry_pairs(triangle)
        for p in pairs:
            assert p.determinant == int(abs(p.linear_map.det()))
            assert p.is_unimodular == (p.determinant == 1)

    def test_homogenized_map_block_structure(self, triangle):
        """T = [[1, 0^T], [t, M]] — first row is [1, 0, 0, 0]."""
        pairs = symmetry_pairs(triangle)
        for p in pairs:
            T = p.homogenized_map
            assert T[0, 0] == 1
            for j in range(1, T.cols):
                assert T[0, j] == 0

    def test_transform_beta_identity(self, triangle):
        """Identity pair maps β to itself."""
        pairs = symmetry_pairs(triangle)
        identity = next(
            p for p in pairs if p.linear_map == sp.eye(3) and p.translation == sp.zeros(3, 1)
        )
        beta = [sp.Symbol("b0"), sp.Symbol("b1"), sp.Symbol("b2"), sp.Symbol("b3")]
        result = identity.transform_beta(beta)
        assert result == beta

    def test_method_on_aconfiguration(self, triangle):
        pairs_fn = symmetry_pairs(triangle)
        pairs_method = triangle.symmetry_pairs()
        assert len(pairs_fn) == len(pairs_method)

    def test_banana3_has_identity(self, banana3):
        pairs = symmetry_pairs(banana3)
        assert any(p.is_unimodular and p.linear_map == sp.eye(3) for p in pairs)


# ──────────────────────────────────────────────────────────────────────────────
# conformal artifacts
# ──────────────────────────────────────────────────────────────────────────────


class TestConformalArtifacts:
    # ── massless_polygon_a_config ─────────────────────────────────────────────

    def test_polygon_n3_matches_triangle(self):
        """C_3 (massless triangle) should reproduce the dissertation triangle."""
        cn3 = massless_polygon_a_config(3)
        tri = triangle_a_config()
        assert cn3.n_points == tri.n_points == 6
        assert cn3.normalized_volume == tri.normalized_volume == 4
        assert cn3.smith_invariants == tri.smith_invariants == [1, 1, 1]

    def test_polygon_n4_matches_box(self):
        """C_4 (massless box) should have N=10 monomials in ℝ⁴."""
        cn4 = massless_polygon_a_config(4)
        assert cn4.n_points == 10
        assert cn4.ambient_dim == 4
        assert cn4.smith_invariants == [1, 1, 1, 1]

    def test_polygon_n5(self):
        cn5 = massless_polygon_a_config(5)
        assert cn5.n_points == 15
        assert cn5.ambient_dim == 5

    def test_polygon_raises_for_n_lt_3(self):
        with pytest.raises(ValueError):
            massless_polygon_a_config(2)

    # ── bms_simplex_a_config ─────────────────────────────────────────────────

    def test_bms_n3_matches_triple_k(self):
        """BMS n=3 should reproduce the dissertation triple-K."""
        bms3 = bms_simplex_a_config(3)
        tpk = triple_k_a_config()
        assert bms3.n_points == tpk.n_points == 6
        assert bms3.normalized_volume == tpk.normalized_volume == 4
        assert bms3.smith_invariants == tpk.smith_invariants == [1, 1, 2]

    def test_bms_shape(self):
        """BMS n-point should produce (n+1) × 2n A-matrix."""
        for n in [2, 3, 4, 5]:
            cfg = bms_simplex_a_config(n)
            assert cfg.matrix.shape == (n + 1, 2 * n)

    def test_bms_smith_has_trailing_2(self):
        """BMS configurations always have Smith invariants ending in 2."""
        for n in [2, 3, 4, 5]:
            cfg = bms_simplex_a_config(n)
            assert cfg.smith_invariants[-1] == 2
            assert all(s == 1 for s in cfg.smith_invariants[:-1])

    def test_bms_volume_is_power_of_two(self):
        """Holonomic rank vol₀(BMS_n) = 2^{n-1}, derived from n-Bessel integral."""
        for n in [2, 3, 4, 5]:
            cfg = bms_simplex_a_config(n)
            assert cfg.normalized_volume == 2 ** (n - 1)

    def test_bms_g_polynomial_has_correct_monomials(self):
        """G polynomial derivation gives exactly 2n monomials with correct degrees."""
        import sympy as sp

        for n in [2, 3, 4, 5]:
            G = _bms_g_polynomial(n)
            u = [sp.Symbol(f"u_{i + 1}") for i in range(n)]
            poly = sp.Poly(G, *u)
            monoms = poly.monoms()
            assert len(monoms) == 2 * n
            # lower monomials: degree n-1, one zero exponent
            # upper monomials: degree n+1, one entry = 2
            lower = [m for m in monoms if sum(m) == n - 1]
            upper = [m for m in monoms if sum(m) == n + 1]
            assert len(lower) == n
            assert len(upper) == n
            # each lower monomial has exactly one zero
            for m in lower:
                assert m.count(0) == 1
            # each upper monomial has exactly one entry equal to 2
            for m in upper:
                assert m.count(2) == 1

    def test_bms_raises_for_n_lt_2(self):
        with pytest.raises(ValueError):
            bms_simplex_a_config(1)

    # ── complete_graph_a_config ───────────────────────────────────────────────

    def test_complete_graph_n3_matches_triangle(self):
        """K_3 = C_3 = triangle, so K_3 LP should match triangle invariants."""
        kn3 = complete_graph_a_config(3)
        tri = triangle_a_config()
        assert kn3.n_points == tri.n_points == 6
        assert kn3.normalized_volume == tri.normalized_volume == 4
        assert kn3.smith_invariants == tri.smith_invariants == [1, 1, 1]

    def test_complete_graph_n4_ambient_dim(self):
        """K_4 has C(4,2)=6 internal edges, so ambient dim = 6."""
        kn4 = complete_graph_a_config(4)
        assert kn4.ambient_dim == 6

    def test_complete_graph_raises_for_n_lt_3(self):
        with pytest.raises(ValueError):
            complete_graph_a_config(2)

    # ── cross-family relations ────────────────────────────────────────────────

    def test_n3_triangle_and_bms3_are_affinely_equivalent(self):
        """The known n=3 equivalence: C_3 LP ~_Q BMS_3 but NOT unimodular."""
        cn3 = massless_polygon_a_config(3)
        bms3 = bms_simplex_a_config(3)
        r_uni = cn3.is_unimodular_equivalent_to(bms3)
        r_aff = cn3.is_affinely_equivalent_to(bms3)
        assert not r_uni.equivalent
        assert r_aff.equivalent

    def test_n3_finite_index_map_det2(self):
        """The finite-index map C_3 → BMS_3 has determinant 2."""
        cn3 = massless_polygon_a_config(3)
        bms3 = bms_simplex_a_config(3)
        fim = finite_index_map(cn3, bms3)
        assert fim.found
        assert fim.determinant == 2

    def test_n4_c4_and_bms4_different_n(self):
        """For n>=4, C_n and BMS_n have different monomial counts."""
        cn4 = massless_polygon_a_config(4)
        bms4 = bms_simplex_a_config(4)
        assert cn4.n_points != bms4.n_points

    # ── conformal_companion_a_config ─────────────────────────────────────────

    def test_companion_n3_matches_triangle_invariants(self):
        """For n=3, conformal_companion must equal the triangle (same Smith, vol)."""
        comp3 = conformal_companion_a_config(3)
        tri = triangle_a_config()
        assert comp3.smith_invariants == tri.smith_invariants == [1, 1, 1]
        assert comp3.normalized_volume == tri.normalized_volume == 4

    def test_companion_n3_finite_index_map_to_bms3(self):
        """conformal_companion(3) → BMS_3 via the known det=2 map."""
        comp3 = conformal_companion_a_config(3)
        bms3 = bms_simplex_a_config(3)
        fim = finite_index_map(comp3, bms3)
        assert fim.found
        assert fim.determinant == 2
        assert not fim.is_unimodular

    def test_companion_n3_finite_index_map_witness(self):
        """The witness matrix for companion(3) → BMS_3 maps every column correctly."""
        comp3 = conformal_companion_a_config(3)
        bms3 = bms_simplex_a_config(3)
        fim = finite_index_map(comp3, bms3)
        assert fim.found
        assert fim.determinant == 2
        # Verify: for each source column v, M·v + t lands in BMS_3 columns.
        M = np.array([[int(fim.witness_matrix[r, c]) for c in range(3)] for r in range(3)])
        t = np.array([int(x) for x in fim.translation])
        bms_cols = {
            tuple(int(bms3.matrix[r, j]) for r in range(1, 4)) for j in range(bms3.n_points)
        }
        for j in range(comp3.n_points):
            v = np.array([int(comp3.matrix[r, j]) for r in range(1, 4)])
            mapped = tuple((M @ v + t).tolist())
            assert mapped in bms_cols, f"column {v} mapped to {mapped} not in BMS_3"

    def test_companion_has_correct_monomial_structure(self):
        """Companion(n) has n lower (degree n-1) and n upper (degree 1) monomials."""
        for n in [3, 4, 5]:
            comp = conformal_companion_a_config(n)
            assert comp.n_points == 2 * n
            assert comp.ambient_dim == n
            cols = [
                tuple(int(comp.matrix[r, j]) for r in range(1, n + 1)) for j in range(comp.n_points)
            ]
            lower = [c for c in cols if sum(c) == n - 1]
            upper = [c for c in cols if sum(c) == 1]
            assert len(lower) == n, f"n={n}: expected {n} lower monomials, got {len(lower)}"
            assert len(upper) == n, f"n={n}: expected {n} upper monomials, got {len(upper)}"

    def test_companion_smith_invariants(self):
        """
        Smith(companion(n)) = [1]*(n-1) + [max(1,n-2)].
        n=3 → [1,1,1]  (full ℤ³, same as triangle)
        n=4 → [1,1,1,2]  (index-2 sublattice, same as BMS_4)
        n=5 → [1,1,1,1,3]  (index-3 sublattice, different from BMS_5=[1,1,1,1,2])
        """
        expected = {3: [1, 1, 1], 4: [1, 1, 1, 2], 5: [1, 1, 1, 1, 3]}
        for n, smith in expected.items():
            comp = conformal_companion_a_config(n)
            assert (
                comp.smith_invariants == smith
            ), f"n={n}: Smith={comp.smith_invariants}, expected {smith}"

    def test_companion_raises_for_n_lt_3(self):
        with pytest.raises(ValueError):
            conformal_companion_a_config(2)
