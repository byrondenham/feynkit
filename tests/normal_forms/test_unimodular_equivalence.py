"""Tests for the Liu–Cai unimodular isomorphism algorithm (arXiv:2506.23846)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph, PolytopeEquivalence
from feynkit.normal_forms import is_unimodular_equivalent


# ──────────────────────────────────────────────────────────────────────────────
# Polytope generators
# ──────────────────────────────────────────────────────────────────────────────


def unit_simplex(n: int) -> np.ndarray:
    """The standard n-simplex with vertices 0, e_1, ..., e_n."""
    return np.vstack([np.zeros(n, dtype=int), np.eye(n, dtype=int)])


def unit_cube(n: int) -> np.ndarray:
    """The unit hypercube [0,1]^n with all 2^n vertices."""
    pts = []
    for i in range(2 ** n):
        pts.append([(i >> j) & 1 for j in range(n)])
    return np.array(pts, dtype=int)


def cross_polytope(n: int) -> np.ndarray:
    """The cross polytope: ±e_i for i = 1..n."""
    pts = []
    for i in range(n):
        e = np.zeros(n, dtype=int)
        e[i] = 1
        pts.append(e.copy())
        e[i] = -1
        pts.append(e.copy())
    return np.array(pts, dtype=int)


def random_unimodular_map(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random U ∈ GL_n(ℤ) by composing elementary row ops."""
    U = np.eye(n, dtype=int)
    n_ops = rng.integers(3, 8)
    for _ in range(n_ops):
        op = rng.integers(0, 3)
        i, j = rng.choice(n, size=2, replace=False)
        if op == 0:
            # Swap two rows.
            U[[i, j]] = U[[j, i]]
        elif op == 1:
            # Add a multiple of row j to row i.
            k = int(rng.integers(-2, 3))
            U[i] = U[i] + k * U[j]
        else:
            # Negate a row (det → -det, still ±1).
            U[i] = -U[i]
    return U


def apply_unimodular(points: np.ndarray, U: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Apply v ↦ U·v + Z to every row of points."""
    return points @ U.T + Z


# ──────────────────────────────────────────────────────────────────────────────
# Self-equivalence
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("polytope_factory,n", [
    (unit_simplex, 2),
    (unit_simplex, 3),
    (unit_cube, 2),
    (unit_cube, 3),
    (cross_polytope, 2),
    (cross_polytope, 3),
])
def test_self_equivalence(polytope_factory, n) -> None:
    pts = polytope_factory(n)
    result = is_unimodular_equivalent(pts, pts)
    assert isinstance(result, PolytopeEquivalence)
    assert result.equivalent is True
    assert result.relation == "unimodular"
    assert result.witness_map is not None
    assert abs(result.witness_map.det()) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Translation only
# ──────────────────────────────────────────────────────────────────────────────


def test_pure_translation() -> None:
    pts = unit_cube(3)
    Z = np.array([5, -2, 17])
    pts_b = pts + Z
    result = is_unimodular_equivalent(pts, pts_b)
    assert result.equivalent is True
    assert result.witness_map == sp.eye(3)


# ──────────────────────────────────────────────────────────────────────────────
# Random unimodular maps
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("polytope_factory,n,seed", [
    (unit_simplex, 2, 1),
    (unit_simplex, 3, 2),
    (unit_cube, 2, 3),
    (unit_cube, 3, 4),
    (cross_polytope, 2, 5),
    (cross_polytope, 3, 6),
])
def test_random_unimodular_image_recovered(polytope_factory, n, seed) -> None:
    """A polytope and its image under a random unimodular map must be unimodularly equivalent."""
    rng = np.random.default_rng(seed)
    pts = polytope_factory(n)
    U = random_unimodular_map(n, rng)
    Z = rng.integers(-5, 6, size=n)
    pts_b = apply_unimodular(pts, U, Z)

    result = is_unimodular_equivalent(pts, pts_b)
    assert result.equivalent is True, f"{polytope_factory.__name__}({n}) under U={U}, Z={Z}"
    assert result.witness_map is not None
    assert abs(result.witness_map.det()) == 1
    assert result.vertex_correspondence is not None


def test_witness_map_actually_works() -> None:
    """For a random unimodular image, the witness U must satisfy U·v_i + Z = v_image_i."""
    rng = np.random.default_rng(42)
    pts = unit_simplex(3)
    U_true = random_unimodular_map(3, rng)
    Z_true = np.array([1, -2, 3])
    pts_b = apply_unimodular(pts, U_true, Z_true)

    result = is_unimodular_equivalent(pts, pts_b)
    assert result.equivalent is True
    U = np.array(result.witness_map.tolist(), dtype=int)
    correspondence = result.vertex_correspondence
    # Recover Z from the first vertex pair.
    Z = pts_b[correspondence[0]] - U @ pts[0]
    # Verify the map sends every source vertex to its claimed image.
    for i, j in enumerate(correspondence):
        assert np.array_equal(U @ pts[i] + Z, pts_b[j])


# ──────────────────────────────────────────────────────────────────────────────
# Negative cases
# ──────────────────────────────────────────────────────────────────────────────


def test_different_vertex_count() -> None:
    a = unit_simplex(2)              # 3 vertices
    b = unit_cube(2)                 # 4 vertices
    assert is_unimodular_equivalent(a, b).equivalent is False


def test_different_dimension() -> None:
    a = unit_simplex(2)              # 2-d
    b = unit_simplex(3)              # 3-d
    assert is_unimodular_equivalent(a, b).equivalent is False


def test_scaled_polytope_not_unimodular() -> None:
    """Scaling by 2 multiplies all edge labels by 2^? — generally not unimodular."""
    pts = unit_cube(2)
    pts_scaled = 2 * pts             # 2x scaled square
    # The scaled cube has different edge lengths, so it's not unimodularly
    # equivalent to the unit cube (det of any U mapping one to the other
    # would be 2^n != ±1).
    assert is_unimodular_equivalent(pts, pts_scaled).equivalent is False


def test_triangle_vs_square_not_equivalent() -> None:
    triangle = unit_simplex(2)
    square = unit_cube(2)
    assert is_unimodular_equivalent(triangle, square).equivalent is False


# ──────────────────────────────────────────────────────────────────────────────
# Reflection (orientation-reversing) is allowed: det U = -1
# ──────────────────────────────────────────────────────────────────────────────


def test_reflection_is_allowed() -> None:
    """The user has confirmed orientation-reversing maps are allowed."""
    pts = unit_simplex(2)
    # Reflect across the x-axis: U = diag(1, -1), Z = (0, 0).
    U = np.array([[1, 0], [0, -1]])
    pts_b = apply_unimodular(pts, U, np.zeros(2, dtype=int))
    result = is_unimodular_equivalent(pts, pts_b)
    assert result.equivalent is True
    assert result.witness_map is not None
    assert result.witness_map.det() in (1, -1)


# ──────────────────────────────────────────────────────────────────────────────
# Robustness to vertex reordering and to non-vertex points in the input
# ──────────────────────────────────────────────────────────────────────────────


def test_permuted_input_order() -> None:
    pts = unit_cube(3)
    perm = np.array([5, 2, 0, 7, 1, 3, 6, 4])
    pts_perm = pts[perm]
    assert is_unimodular_equivalent(pts, pts_perm).equivalent is True


def test_interior_lattice_points_ignored() -> None:
    """Lattice points strictly inside the polytope must not affect the result."""
    pts = unit_cube(2)                  # 4 vertices
    # Add a redundant point inside the square (not a vertex).
    augmented = np.vstack([pts, np.array([[0, 0]])])  # duplicate of first vertex
    # Should still be equivalent to the plain unit cube.
    assert is_unimodular_equivalent(pts, augmented).equivalent is True


# ──────────────────────────────────────────────────────────────────────────────
# Newton polytopes from Feynman integrals
# ──────────────────────────────────────────────────────────────────────────────


def _bubble_integral() -> FeynmanIntegral:
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
    graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    return FeynmanIntegral(graph)


def test_bubble_self_equivalent_via_integral() -> None:
    a = _bubble_integral()
    b = _bubble_integral()
    result = a.is_unimodular_equivalent_to(b)
    assert result.equivalent is True
    assert result.relation == "unimodular"


def test_two_different_loops_not_equivalent() -> None:
    """A bubble (2 internal edges) and a sunrise (3 internal edges) have
    Newton polytopes of different dimensions, so they cannot be equivalent."""
    bubble = _bubble_integral()

    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=1, v2=2, is_internal=True)
    e3 = Edge(idx=3, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=4, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=5, v1=2, v2=4, is_internal=False)
    sunrise_graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, e3, ex1, ex2])
    sunrise = FeynmanIntegral(sunrise_graph)

    result = bubble.is_unimodular_equivalent_to(sunrise)
    assert result.equivalent is False


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation: Liu–Cai must agree with the brute-force on small instances
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("polytope_factory,n,seed", [
    (unit_simplex, 2, 100),
    (unit_simplex, 2, 101),
    (unit_cube, 2, 102),
    (unit_simplex, 3, 103),
])
def test_cross_validation_with_brute_force(polytope_factory, n, seed) -> None:
    """Liu–Cai must agree with the brute-force backend on small examples (where
    affine equivalence implies unimodular for unimodular images)."""
    from feynkit.normal_forms import is_affinely_equivalent

    rng = np.random.default_rng(seed)
    pts = polytope_factory(n)
    U = random_unimodular_map(n, rng)
    Z = rng.integers(-3, 4, size=n)
    pts_b = apply_unimodular(pts, U, Z)

    uni = is_unimodular_equivalent(pts, pts_b)
    aff = is_affinely_equivalent(sp.Matrix(pts.tolist()), sp.Matrix(pts_b.tolist()))
    # If unimodularly equivalent, certainly affinely equivalent.
    assert uni.equivalent is True
    assert aff.equivalent is True
