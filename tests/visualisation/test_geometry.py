"""Tests for the 3D geometry helpers behind the Newton-polytope renderer."""

from __future__ import annotations

import numpy as np
import pytest

from feynkit.core.exceptions import ComputationError
from feynkit.visualisation.geometry import (
    _is_planar,
    classify_edges_by_visibility,
    compute_convex_hull_3d,
    determine_label_position,
    extract_edges_from_hull,
    format_coordinate,
    project_to_3d_pca,
)

CUBE = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
SQUARE_IN_3D = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
TETRA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)


class TestProjection:
    def test_2d_points_are_padded_to_3d(self) -> None:
        pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        out = project_to_3d_pca(pts)
        assert out.shape == (3, 3)
        assert np.allclose(out[:, :2], pts)
        assert np.allclose(out[:, 2], 0)

    def test_3d_points_are_returned_unchanged(self) -> None:
        assert np.allclose(project_to_3d_pca(TETRA), TETRA)

    def test_high_dimensional_points_are_reduced_to_3d(self) -> None:
        rng = np.random.default_rng(0)
        pts = rng.random((10, 5))
        out = project_to_3d_pca(pts)
        assert out.shape == (10, 3)

    def test_pca_preserves_pairwise_distances_for_embedded_3d_data(self) -> None:
        # A 3-simplex embedded in 5D via a zero-padded orthonormal map.
        pts5 = np.hstack([TETRA, np.zeros((4, 2))])
        out = project_to_3d_pca(pts5)
        d_in = np.linalg.norm(pts5[:, None] - pts5[None], axis=-1)
        d_out = np.linalg.norm(out[:, None] - out[None], axis=-1)
        assert np.allclose(d_in, d_out)


class TestPlanarity:
    def test_square_is_planar(self) -> None:
        assert _is_planar(SQUARE_IN_3D)

    def test_tetrahedron_is_not_planar(self) -> None:
        assert not _is_planar(TETRA)


class TestEdgeExtraction:
    def test_cube_has_twelve_edges(self) -> None:
        hull = compute_convex_hull_3d(CUBE)
        edges = extract_edges_from_hull(hull, CUBE)
        assert len(edges) == 12
        assert all(a < b for a, b in edges)
        # Every cube edge joins points differing in exactly one coordinate.
        assert all(np.sum(CUBE[a] != CUBE[b]) == 1 for a, b in edges)

    def test_tetrahedron_has_six_edges(self) -> None:
        hull = compute_convex_hull_3d(TETRA)
        assert len(extract_edges_from_hull(hull, TETRA)) == 6

    def test_coplanar_points_are_rejected_as_a_3d_hull(self) -> None:
        # Exactly coplanar input has no 3D hull; the renderer falls back to a
        # points-only picture in that case.
        with pytest.raises(ComputationError):
            compute_convex_hull_3d(SQUARE_IN_3D)

    def test_interior_point_is_not_a_vertex(self) -> None:
        pts = np.vstack([CUBE, [[0.5, 0.5, 0.5]]])
        hull = compute_convex_hull_3d(pts)
        edges = extract_edges_from_hull(hull, pts)
        assert all(8 not in e for e in edges)


class TestVisibility:
    def test_cube_partitions_edges_and_hides_some(self) -> None:
        hull = compute_convex_hull_3d(CUBE)
        edges = extract_edges_from_hull(hull, CUBE)
        visible, hidden = classify_edges_by_visibility(CUBE, hull, edges)
        assert sorted(visible + hidden) == sorted(edges)
        assert set(visible).isdisjoint(hidden)
        assert visible and hidden

    def test_camera_direction_changes_visibility(self) -> None:
        hull = compute_convex_hull_3d(CUBE)
        edges = extract_edges_from_hull(hull, CUBE)
        top, _ = classify_edges_by_visibility(CUBE, hull, edges, np.array([0, 0, 1]))
        bottom, _ = classify_edges_by_visibility(CUBE, hull, edges, np.array([0, 0, -1]))
        assert set(top) != set(bottom)


class TestFormatting:
    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            ([2, 0, 0], "right"),
            ([-2, 0, 0], "left"),
            ([0, 2, 0], "above"),
            ([0, -2, 0], "below"),
            ([0, 0, 2], "above right"),
            ([0, 0, -2], "below left"),
        ],
    )
    def test_label_position_follows_dominant_axis(self, point: list[int], expected: str) -> None:
        assert determine_label_position(np.array(point, dtype=float)) == expected

    def test_whole_numbers_print_without_decimals(self) -> None:
        assert format_coordinate(3.0) == "3"
        assert format_coordinate(-0.0) == "0"

    def test_fractions_respect_precision(self) -> None:
        assert format_coordinate(1.23456) == "1.23"
        assert format_coordinate(1.23456, precision=4) == "1.2346"
