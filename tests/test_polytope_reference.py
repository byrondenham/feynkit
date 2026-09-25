"""
The exact polyhedral core against a floating-point reference, and properties.

The _float_ functions follow the code the exact core replaced: faces from
Qhull on an SVD projection with a tolerance of 1e-7, facet normals from SymPy
nullspaces, and volumes from Qhull divided by the product of SymPy's Smith
invariants. They share one routine with the code under test: below full
dimension the volume is measured in the coordinates lattice_coordinates
gives, which come from the exact lattice_chart where the old code used SymPy.
tests/test_polytope.py checks those coordinates against the SymPy
construction. Nothing else in the reference calls feynkit.
"""

from __future__ import annotations

import functools
import math
import random
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import numpy as np
import pytest
import sympy as sp
from scipy.spatial import ConvexHull
from sympy.matrices.normalforms import smith_normal_form
from sympy.polys.domains import ZZ

from feynkit import _exact, polytope
from feynkit.a_configuration import AConfiguration
from feynkit.polytope import faces, lattice_coordinates, polytope_data
from tests.test_polytope import STEEP, assert_lattice_forms, diagram_points, requires_normaliz

CNICKEL: list[Any] = [
    "0|:n",
    "0|:z",
    "11e|e|:zz",
    "11e|e|:nn",
    "e11|e|:zzz|z|",
    "e11|e|:n00|n|",
    "12e|2e|e|:zzz",
    "12e|2e|e|:nzz",
    "12e|2e|e|:znz",
    "12e|2e|e|:sss",
    "12e|2e|e|:aab",
    "111e|e|:zzz",
    "111e|e|:nnn",
    "12e|3e|3e|e|:zzzz",
    "12e|3e|3e|e|:nnnn",
    "13e|2e|3e|e|:zzzz",
    "12e|23|3|e|:zzzzz",
    "12e|23|3|e|:nnnnn",
    "12ee|22e|e|:zzzz",
    "12ee|22e|e|:nnnn",
    "15e|24|3e|4e|5|e|:zzzzzzz",
    pytest.param("145|26|3e|4e|e|6e|e|:zzzzzzzz", id="pentagon-box", marks=pytest.mark.slow),
]


# --- the floating reference --------------------------------------------------


def _float_affine_rank(pts: np.ndarray) -> int:
    if len(pts) <= 1:
        return 0
    diffs = (pts[1:] - pts[0]).astype(float)
    return int(np.linalg.matrix_rank(diffs, tol=1e-9))


def _float_faces(pts: np.ndarray) -> list[tuple[int, tuple[int, ...]]]:
    n_pts = len(pts)
    rank = _float_affine_rank(pts)
    if rank == 0:
        return [(0, tuple(range(n_pts)))]
    diffs = (pts - pts[0]).astype(float)
    _, _, vt = np.linalg.svd(diffs, full_matrices=False)
    coords = diffs @ vt[:rank].T
    if rank == 1:
        order = np.argsort(coords[:, 0])
        lo, hi = int(order[0]), int(order[-1])
        return [(0, (lo,)), (0, (hi,)), (1, tuple(range(n_pts)))]
    hull = ConvexHull(coords)
    facets: set[frozenset[int]] = set()
    for eq in hull.equations:
        normal, offset = eq[:-1], eq[-1]
        facets.add(frozenset(i for i in range(n_pts) if abs(coords[i] @ normal + offset) < 1e-7))
    all_faces: set[frozenset[int]] = set(facets)
    frontier = set(facets)
    while frontier:
        new: set[frozenset[int]] = set()
        for a in frontier:
            for b in all_faces:
                c = a & b
                if c and c not in all_faces and c not in new:
                    new.add(c)
        all_faces |= new
        frontier = new
    all_faces.add(frozenset(range(n_pts)))
    return [(_float_affine_rank(pts[sorted(face)]), tuple(sorted(face))) for face in all_faces]


def _float_facet(pts: np.ndarray, idx: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    base = pts[idx[0]]
    ambient = int(pts.shape[1])
    diffs = sp.Matrix(
        len(idx) - 1, ambient, [int(x) for i in idx[1:] for x in (pts[i] - base).tolist()]
    )
    null = diffs.nullspace()
    if len(null) != 1:
        raise ValueError("facet does not span a hyperplane")
    vec = null[0]
    lcm = sp.ilcm(*[sp.Rational(x).q for x in vec], 1)
    ints = [int(x * lcm) for x in vec]
    g = math.gcd(*ints)
    normal = [x // g for x in ints]
    centroid = pts.mean(axis=0)
    offset = int(sum(m * int(x) for m, x in zip(normal, base, strict=True)))
    if sum(m * c for m, c in zip(normal, centroid, strict=True)) > offset:
        normal = [-m for m in normal]
        offset = -offset
    return tuple(normal), offset


def _float_hull_volume(pts: np.ndarray) -> int:
    """AConfiguration.normalized_volume before the exact core, on full-dimensional input."""
    diag = smith_normal_form(sp.Matrix((pts[1:] - pts[0]).tolist()), domain=ZZ)
    index = math.prod(abs(int(diag[i, i])) for i in range(min(diag.shape)) if diag[i, i] != 0)
    return int(round(math.factorial(pts.shape[1]) * ConvexHull(pts.astype(float)).volume / index))


def _float_volume(pts: np.ndarray, dimension: int) -> int:
    """The volume as polytope_data computed it before the exact core.

    Below full dimension it is measured in the coordinates of
    lattice_coordinates, the exact lattice chart of the code under test.
    """
    if dimension == 0:
        return 1
    if dimension == 1:
        along = [c[0] for c in lattice_coordinates(pts)]
        return max(along) - min(along)
    if dimension < pts.shape[1]:
        pts = np.asarray(lattice_coordinates(pts), dtype=np.int64)
    return _float_hull_volume(pts)


class _Reference(NamedTuple):
    faces: list[tuple[int, tuple[int, ...]]]
    facets: set[tuple[tuple[int, ...], int]]
    volume: int


def _float_reference(points: list[tuple[int, ...]]) -> _Reference | None:
    """The floating results, or None where the floating code fails."""
    pts = np.asarray(points, dtype=np.int64)
    try:
        all_faces = sorted(_float_faces(pts))
        dimension = max(d for d, _ in all_faces)
        facets: set[tuple[tuple[int, ...], int]] = set()
        if dimension == pts.shape[1]:
            facets = {_float_facet(pts, idx) for d, idx in all_faces if d == dimension - 1}
        return _Reference(all_faces, facets, _float_volume(pts, dimension))
    except Exception:  # the comparison runs wherever the floating code runs
        return None


# --- samples -----------------------------------------------------------------


def _embed(
    rng: random.Random, points: list[tuple[int, ...]], ambient: int
) -> list[tuple[int, ...]]:
    """The image of the points under x -> M x + t, with M an injective integer matrix."""
    d = len(points[0])
    while True:
        m = [[rng.randint(-2, 2) for _ in range(d)] for _ in range(ambient)]
        if sp.Matrix(m).rank() == d:
            break
    shift = [rng.randint(-3, 3) for _ in range(ambient)]
    return [tuple(s + _exact.dot(row, p) for row, s in zip(m, shift, strict=True)) for p in points]


def _configurations(seed: int, count: int) -> list[list[tuple[int, ...]]]:
    """Distinct random lattice points of dimension 2 to 5.

    Every third set is embedded in Z^(d + 1) or Z^(d + 2). The points are
    distinct because the floating reference keeps one index per endpoint of a
    segment where the exact code keeps every repeated point.
    """
    rng = random.Random(seed)
    out = []
    for i in range(count):
        d = rng.randint(2, 5)
        size = rng.randint(d + 1, d + 6)
        points = sorted({tuple(rng.randint(-2, 2) for _ in range(d)) for _ in range(size)})
        if i % 3 == 0:
            points = _embed(rng, points, d + rng.randint(1, 2))
        out.append(points)
    return out


RANDOM = _configurations(2026, 200)


@functools.cache
def _diagrams() -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(diagram_points(c)) for c in CNICKEL if isinstance(c, str))


def _with_repeats(
    seed: int, samples: Iterable[Sequence[tuple[int, ...]]]
) -> list[list[tuple[int, ...]]]:
    """Each sample with a copy of its least point and of one or two random points, shuffled.

    The least point in lexicographic order is a vertex, so every sample has a
    repeated vertex.
    """
    rng = random.Random(seed)
    out = []
    for sample in samples:
        points = list(sample)
        points += [min(points)] + [rng.choice(points) for _ in range(rng.randint(1, 2))]
        rng.shuffle(points)
        out.append(points)
    return out


@functools.cache
def _repeated() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Samples with repeated points, which the floating reference cannot take."""
    by_hand = [
        [(2, 2), (0, 0), (1, 1), (2, 2), (0, 0)],
        [(3,), (1,), (3,), (2,)],
        [(1, 2, 3), (1, 2, 3)],
    ]
    return tuple(
        tuple(points)
        for points in by_hand + _with_repeats(7, RANDOM[::10] + list(_diagrams()[:15]))
    )


def _distinct_samples() -> list[list[tuple[int, ...]]]:
    return RANDOM + [list(points) for points in _diagrams()]


def _samples() -> list[list[tuple[int, ...]]]:
    return _distinct_samples() + [list(points) for points in _repeated()]


# Sets on which Qhull's tolerance of 1e-7 misplaces points, so that its tight sets are wrong: the
# steep pentagon of tests/test_polytope.py, the pyramid over it, and the cube [0, 2]^3 with its
# centre and the centres of two facets, sheared by x -> (x_1 + 10^9 x_2, x_2 + 10^9 x_3, x_3).
_CORNERS = [(a, b, c) for a in (0, 2) for b in (0, 2) for c in (0, 2)]
_CUBE = [*_CORNERS, (1, 1, 0), (1, 1, 1), (0, 1, 1)]
STEEP_SETS = [
    STEEP,
    [(x, y, 0) for x, y in STEEP] + [(0, 0, 1)],
    [(a + 10**9 * b, b + 10**9 * c, c) for a, b, c in _CUBE],
]


def _unimodular(rng: random.Random, n: int) -> list[list[int]]:
    """A random matrix in GL_n(Z), built from elementary row operations and swaps."""
    if n == 1:
        return [[rng.choice((-1, 1))]]
    m = [[int(i == j) for j in range(n)] for i in range(n)]
    for _ in range(3 * n):
        i, j = rng.sample(range(n), 2)
        c = rng.choice((-2, -1, 1, 2))
        m[i] = [a + c * b for a, b in zip(m[i], m[j], strict=True)]
        if rng.random() < 0.3:
            m[i], m[j] = m[j], m[i]
    return m


# --- exact against floating --------------------------------------------------


def _agree(points: list[tuple[int, ...]]) -> bool:
    reference = _float_reference(points)
    if reference is None:
        return False
    data = polytope_data(points)
    assert list(data.faces) == reference.faces
    assert {(f.normal, f.offset) for f in data.facets} == reference.facets
    assert data.normalized_volume == reference.volume
    return True


@pytest.mark.parametrize("cnickel", CNICKEL)
def test_diagrams_agree_with_the_floating_reference(cnickel: str) -> None:
    assert _agree(diagram_points(cnickel))


def test_random_configurations_agree_with_the_floating_reference() -> None:
    assert sum(_agree(points) for points in RANDOM) >= 180


def test_aconfiguration_volume_agrees_on_full_dimensional_input() -> None:
    compared = 0
    for points in _distinct_samples():
        if _exact.affine_rank(points) != len(points[0]) or len(points[0]) < 2:
            continue
        cfg = AConfiguration(np.array(points).T, is_homogenized=False)
        assert cfg.normalized_volume == _float_hull_volume(np.asarray(points, dtype=np.int64))
        compared += 1
    assert compared >= 90


# --- properties --------------------------------------------------------------


def test_euler_poincare() -> None:
    for points in _samples():
        f = polytope_data(points).f_vector
        assert sum((-1) ** k * n for k, n in enumerate(f)) == 1


def test_volume_and_faces_are_invariant_under_unimodular_maps() -> None:
    rng = random.Random(4)
    for points in _samples():
        n = len(points[0])
        m = _unimodular(rng, n)
        shift = [rng.randint(-5, 5) for _ in range(n)]
        image = [
            tuple(s + _exact.dot(row, p) for row, s in zip(m, shift, strict=True)) for p in points
        ]
        before, after = polytope_data(points), polytope_data(image)
        assert after.normalized_volume == before.normalized_volume
        assert after.faces == before.faces


def test_volume_and_faces_do_not_depend_on_the_embedding() -> None:
    rng = random.Random(5)
    for points in _configurations(6, 60) + [list(points) for points in _repeated()]:
        image = _embed(rng, points, len(points[0]) + rng.randint(1, 2))
        before, after = polytope_data(points), polytope_data(image)
        assert after.faces == before.faces
        assert after.normalized_volume == before.normalized_volume


def test_every_copy_of_a_point_lies_on_its_faces() -> None:
    """The faces of a sample with repeats are those of its distinct points, listing every copy."""
    for points in _repeated():
        distinct = sorted(set(points))
        where = [distinct.index(p) for p in points]
        expected = sorted(
            (k, tuple(j for j, i in enumerate(where) if i in idx)) for k, idx in faces(distinct)
        )
        assert faces(points) == expected


def test_triangulation_determinants_and_euclidean_volume() -> None:
    checked = 0
    for points in _samples():
        d = _exact.affine_rank(points)
        if d < 2 or d != len(points[0]):
            continue
        data = polytope_data(points)
        simplices = _exact.pulling_simplices(polytope._hull(points, "python").lattice, points)
        total = sum(
            abs(
                _exact.determinant(
                    [[a - b for a, b in zip(points[i], points[s[0]], strict=True)] for i in s[1:]]
                )
            )
            for s in simplices
        )
        assert total == data.sublattice_index * data.normalized_volume
        assert len(simplices) <= data.normalized_volume
        euclidean = ConvexHull(np.asarray(points, dtype=float)).volume
        scaled = data.normalized_volume * data.sublattice_index / math.factorial(d)
        assert math.isclose(scaled, euclidean, rel_tol=1e-9)
        checked += 1
    assert checked >= 100


def test_python_and_qhull_backends_agree() -> None:
    """Qhull's candidates certify on every sample, so python is not compared with itself."""
    for points in _samples():
        assert polytope._hull(points, "qhull").generator in ("qhull", "closed form")
        assert polytope_data(points, backend="qhull") == polytope_data(points, backend="python")


def test_qhull_on_steep_sets_is_exact() -> None:
    """Qhull's candidates are verified, and replaced by beneath-beyond where they do not certify."""
    for points in STEEP_SETS:
        assert polytope_data(points, backend="qhull") == polytope_data(points, backend="python")


@requires_normaliz
def test_python_and_normaliz_backends_agree() -> None:
    """With PyNormaliz installed, no sample falls back to beneath-beyond, steep sets included."""
    for points in _samples() + STEEP_SETS:
        assert polytope._hull(points, "normaliz").generator in ("normaliz", "closed form")
        assert polytope_data(points, backend="normaliz") == polytope_data(points, backend="python")


def test_lattice_forms_on_every_sample() -> None:
    for points in _samples():
        assert_lattice_forms(polytope_data(points))
