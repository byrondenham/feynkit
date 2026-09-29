"""
Lattice invariants of Newton polytopes against Normaliz and Sage.

Both run at test time on the Newton polytopes of generated graphs: the
one-loop graphs with two to six legs, and under the slow marker a sample of
the two-loop graphs with two to four legs. Normaliz is called through
PyNormaliz, directly rather than through feynkit's backend, and Sage as a
program. Each test skips when its program is missing. The polytopes go to both in the coordinates of invariant_chart,
where the support lattice is Z^d, so that their lattice is the one feynkit
uses by default.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from feynkit import FeynmanIntegral, _exact, generate_graphs
from feynkit.lattice_invariants import (
    count_lattice_points,
    ehrhart_polynomial,
    gorenstein_index,
    h_star_vector,
    invariant_chart,
    is_idp,
    lattice_invariants,
    polar_dual,
)

SAGE = shutil.which("sage")

requires_normaliz = pytest.mark.skipif(
    importlib.util.find_spec("PyNormaliz") is None, reason="PyNormaliz not installed"
)
requires_sage = pytest.mark.skipif(SAGE is None, reason="Sage not installed")

ONE_LOOP = list(generate_graphs(1, range(2, 7)))
TWO_LOOP_SAMPLE = list(generate_graphs(2, range(2, 5)))[::15]
GRAPHS = [
    *ONE_LOOP,
    *(pytest.param(cnickel, marks=pytest.mark.slow) for cnickel in TWO_LOOP_SAMPLE),
]

Points = list[tuple[int, ...]]


def newton_points(cnickel: str) -> Points:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    return [tuple(int(x) for x in p) for p in fi.newton_polytope.points]


def chart_points(points: Points) -> Points:
    return sorted(set(invariant_chart(points).coordinates))


def run_normaliz(points: Points) -> dict[str, Any]:
    """What PyNormaliz gives for the polytope of the points, with the keys of a Normaliz .inv file."""
    import PyNormaliz

    cone = PyNormaliz.Cone(polytope=[list(p) for p in points])
    numerator, denominator, shift = cone.HilbertSeries()
    *quasi, quasi_denominator = cone.HilbertQuasiPolynomial()
    gorenstein = cone.IsGorenstein()
    assert shift == 0
    return {
        "hilbert_basis_elements": len(cone.HilbertBasis()),
        "degree_1_elements": len(cone.Deg1Elements()),
        "hilbert_series_num": numerator,
        "hilbert_series_denom": denominator,
        "hilbert_quasipolynomial": quasi,
        "hilbert_quasipolynomial_denom": quasi_denominator,
        "Gorenstein": gorenstein,
        "generator_of_interior": cone.GeneratorOfInterior() if gorenstein else None,
        "integrally_closed": cone.IsIntegrallyClosed(),
    }


def normaliz_ehrhart(values: dict[str, Any]) -> tuple[Fraction, ...]:
    (quasi,) = values["hilbert_quasipolynomial"]
    denominator = values["hilbert_quasipolynomial_denom"]
    return tuple(Fraction(c, denominator) for c in quasi)


def check_with_normaliz(points: Points) -> None:
    coordinates = chart_points(points)
    d = len(coordinates[0])
    values = run_normaliz(coordinates)

    ehrhart = normaliz_ehrhart(values)
    poly = ehrhart_polynomial(points)
    assert tuple(Fraction(int(c.p), int(c.q)) for c in reversed(poly.all_coeffs())) == ehrhart

    # Normaliz may write the Hilbert series over another denominator; over (1 - t)^(d + 1) its
    # numerator is the h*-vector.
    h_star = h_star_vector(points)
    if values["hilbert_series_denom"] == [1] * (d + 1):
        numerator = values["hilbert_series_num"]
        assert list(h_star) == numerator + [0] * (d + 1 - len(numerator))

    assert count_lattice_points(points) == values["degree_1_elements"]
    for k in (1, 2):
        interior = count_lattice_points(points, k, interior=True)
        assert interior == (-1) ** d * sum(c * (-k) ** i for i, c in enumerate(ehrhart))

    if values["Gorenstein"]:
        assert gorenstein_index(points) == values["generator_of_interior"][-1]
    else:
        assert gorenstein_index(points) is None

    # P has IDP exactly when its Hilbert basis lies in degree 1, and N A is normal exactly when
    # the monoid the points generate is integrally closed.
    idp = values["hilbert_basis_elements"] == values["degree_1_elements"]
    assert is_idp(points, backend="python") == idp
    found = lattice_invariants(points, backend="python")
    assert found.idp == idp
    assert found.normal == values["integrally_closed"]


@requires_normaliz
@pytest.mark.parametrize("cnickel", GRAPHS)
def test_normaliz_on_graphs(cnickel: str) -> None:
    check_with_normaliz(newton_points(cnickel))


@requires_normaliz
@pytest.mark.parametrize("seed", range(40))
def test_normaliz_on_random_polytopes(seed: int) -> None:
    # Random supports in a small box: some miss lattice points, some lack IDP.
    rng = random.Random(12000 + seed)
    d = 3 + seed % 2
    while True:
        points = [
            tuple(rng.randint(0, 2) for _ in range(d)) for _ in range(rng.randint(d + 1, d + 5))
        ]
        if _exact.affine_rank(points) == d:
            break
    check_with_normaliz(points)


SAGE_SCRIPT = """
import json, sys
results = []
for points in json.load(open(sys.argv[1])):
    P = LatticePolytope(points)
    interior = P.interior_points()
    reflexive = False
    polar = None
    if len(interior) == 1:
        Q = LatticePolytope([vector(p) - vector(interior[0]) for p in points])
        reflexive = Q.is_reflexive()
        if reflexive:
            polar = sorted(tuple(int(x) for x in v) for v in Q.polar().vertices())
    results.append({"reflexive": bool(reflexive), "polar": polar})
json.dump(results, open(sys.argv[2], "w"))
"""


def run_sage(polytopes: list[Points], folder: Path) -> list[dict[str, Any]]:
    """Sage's is_reflexive and polar vertices for each polytope, translated to its interior point."""
    assert SAGE is not None
    source, target, script = folder / "in.json", folder / "out.json", folder / "check.sage"
    source.write_text(json.dumps(polytopes))
    script.write_text(SAGE_SCRIPT)
    subprocess.run([SAGE, str(script), str(source), str(target)], check=True, capture_output=True)
    results: list[dict[str, Any]] = json.loads(target.read_text())
    return results


def check_with_sage(polytopes: list[Points], folder: Path) -> None:
    # Sage's LatticePolytope calls PALP, built for dimension at most 6.
    polytopes = [chart_points(points) for points in polytopes]
    polytopes = [points for points in polytopes if len(points[0]) <= 6]
    for points, found in zip(polytopes, run_sage(polytopes, folder), strict=True):
        assert (gorenstein_index(points) == 1) == found["reflexive"], points
        expected = None if found["polar"] is None else tuple(tuple(v) for v in found["polar"])
        assert polar_dual(points) == expected, points


@requires_sage
def test_sage_on_one_loop_graphs(tmp_path: Path) -> None:
    check_with_sage([newton_points(c) for c in ONE_LOOP], tmp_path)


@requires_sage
@pytest.mark.slow
def test_sage_on_two_loop_graphs(tmp_path: Path) -> None:
    check_with_sage([newton_points(c) for c in TWO_LOOP_SAMPLE], tmp_path)


@requires_sage
def test_sage_on_random_polytopes(tmp_path: Path) -> None:
    # No Newton polytope above is reflexive; these polytopes about the origin often are.
    rng = random.Random(11)
    polytopes = []
    for d in (2, 3, 4):
        box = list(itertools.product((-1, 0, 1), repeat=d))
        while sum(1 for p in polytopes if len(p[0]) == d) < 15:
            points = [(0,) * d, *rng.sample(box, rng.randint(d + 1, 2 * d + 3))]
            if _exact.affine_rank(points) == d:
                polytopes.append(points)
    assert sum(gorenstein_index(p) == 1 for p in polytopes) >= 10
    check_with_sage(polytopes, tmp_path)
