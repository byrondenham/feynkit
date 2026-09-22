from __future__ import annotations

import numpy as np
import pytest

from feynkit import FeynmanIntegral
from feynkit.polytope import Facet, PolytopeData, faces, lattice_coordinates, polytope_data


class TestMasslessBubble:
    """Newt(G) for G = u_1 + u_2 - s u_1 u_2 / mu^2: the triangle (1,0), (0,1), (1,1)."""

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls) -> PolytopeData:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        return polytope_data(fi.newton_polytope.points)

    def test_dimensions_and_vertices(self, data: PolytopeData) -> None:
        assert data.ambient_dimension == 2
        assert data.dimension == 2
        assert data.is_full_dimensional
        assert set(data.vertices) == {(1, 0), (0, 1), (1, 1)}

    def test_facet_inequalities(self, data: PolytopeData) -> None:
        # x <= 1, y <= 1, -x - y <= -1, each with a primitive outward normal.
        got = {(f.normal, f.offset) for f in data.facets}
        assert got == {((1, 0), 1), ((0, 1), 1), ((-1, -1), -1)}

    def test_every_point_satisfies_every_facet(self, data: PolytopeData) -> None:
        for f in data.facets:
            for p in data.points:
                assert sum(m * x for m, x in zip(f.normal, p, strict=True)) <= f.offset

    def test_f_vector_and_volume(self, data: PolytopeData) -> None:
        assert data.f_vector == (3, 3, 1)
        assert data.normalized_volume == 1

    def test_facet_points_lie_on_their_hyperplane(self, data: PolytopeData) -> None:
        for f in data.facets:
            assert isinstance(f, Facet)
            for i in f.point_indices:
                p = data.points[i]
                assert sum(m * x for m, x in zip(f.normal, p, strict=True)) == f.offset


class TestOneMassTriangle:
    def test_counts_match_landau_faces(self) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        data = polytope_data(fi.newton_polytope.points)
        assert data.dimension == 3
        assert len(data.faces) == 27  # the count the Landau tests report
        assert len(data.facets) == data.f_vector[2]
        assert data.normalized_volume == 5  # A = 4 x 7, one-mass triangle


class TestDegenerate:
    def test_segment_is_not_full_dimensional(self) -> None:
        data = polytope_data([(0, 0), (1, 1), (2, 2)])
        assert data.dimension == 1
        assert not data.is_full_dimensional
        assert data.facets == ()
        assert set(data.vertex_indices) == {0, 2}
        assert data.normalized_volume == 2

    def test_moved_helpers_still_work(self) -> None:
        pts = np.array([[0, 0], [1, 0], [0, 1]])
        assert (2, (0, 1, 2)) in faces(pts)
        assert lattice_coordinates(np.array([[0, 0], [2, 2], [4, 4]])) == [(0,), (1,), (2,)]
