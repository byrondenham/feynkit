"""Tests for feynkit.strata.torus_euler_characteristic.

Oracles: Kouchnirenko's theorem on random full-support polynomials (Invent. Math. 32 (1976)
1-31, Thm IV, p. 30), the conics of Telen and Weinstein, "Euler stratification of hypersurface
families", Ex. 1.2, p. 2, hand computations on curves and point sets whose pieces are read off
a parametrisation, both backends against each other, and stubbed runs for the error paths.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pytest
import sympy as sp

from feynkit import strata as strata_module
from feynkit.landau import _singular_binary
from feynkit.point_count import _msolve_binary
from feynkit.polytope import polytope_data
from feynkit.strata import torus_euler_characteristic

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")
requires_msolve = pytest.mark.skipif(_msolve_binary() is None, reason="msolve not installed")

BACKENDS = [
    "singular",
    pytest.param("msolve", marks=requires_msolve),
]

X, Y, Z, W = sp.symbols("x y z w")


def _chi(generators: list[sp.Expr], variables: list[sp.Symbol], backend: str, **kw: Any) -> int:
    return torus_euler_characteristic(generators, variables, backend=backend, **kw)


@pytest.fixture
def count_calls(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], list[int]]:
    """Counts the calls of a method of the per-prime run, by name."""
    calls: dict[str, list[int]] = {}

    def spy(name: str) -> list[int]:
        original = getattr(strata_module._Run, name)
        log = calls.setdefault(name, [])

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            log.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(strata_module._Run, name, wrapped)
        return log

    return spy


@requires_singular
@pytest.mark.parametrize("backend", BACKENDS)
class TestKouchnirenko:
    """chi(V(f) cap T) = (-1)^(d-1) d! Vol(Newt f) for random f with the given support."""

    SUPPORTS: list[list[tuple[int, ...]]] = [
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)],
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (0, 1), (2, 2), (3, 1)],
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)],
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)],
        [
            (0, 0, 0),
            (2, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
        ],
    ]

    @pytest.mark.parametrize("index", range(len(SUPPORTS)))
    def test_volume(self, backend: str, index: int) -> None:
        support = self.SUPPORTS[index]
        variables = [X, Y, Z][: len(support[0])]
        rng = random.Random(index)
        f = sum(
            rng.randint(1, 30) * sp.Mul(*(v**e for v, e in zip(variables, point, strict=True)))
            for point in support
        )
        data = polytope_data(support)
        assert data.sublattice_index == 1
        expected = (-1) ** (len(variables) - 1) * data.normalized_volume
        assert _chi([f], variables, backend) == expected


@requires_singular
@pytest.mark.parametrize("backend", BACKENDS)
class TestConics:
    """chi(V(f) cap T) for f = z0 + z1 x + z2 y + z3 x^2 + z4 x y + z5 y^2 (TW24, Ex. 1.2, p. 2)."""

    MONOMIALS = [1, X, Y, X**2, X * Y, Y**2]

    @pytest.mark.parametrize(
        ("z", "value"),
        [
            # -4 off the principal A-determinant E_A of Eq. (5), -3 at a generic point of each
            # of its seven components: z0, z3 and z5, z1^2 = 4 z0 z3, z2^2 = 4 z0 z5,
            # z4^2 = 4 z3 z5 and the cubic Delta of Eq. (3).
            ((3, 5, 7, 11, 13, 17), -4),
            ((0, 5, 7, 11, 13, 17), -3),
            ((3, 5, 7, 0, 13, 17), -3),
            ((3, 5, 7, 11, 13, 0), -3),
            ((1, 4, 7, 4, 13, 17), -3),
            ((1, 5, 6, 11, 13, 9), -3),
            ((3, 5, 7, 1, 6, 9), -3),
            ((11, 11, 22, 33, 55, 27), -3),
            # Deeper points, read off by hand, which together with the above give the values
            # 0, -1, -2, -3, -4 that Ex. 1.2 lists. 5x + 7y + 13xy + 17y^2 is a line in x
            # over y less three points; 5x + 7y + 13xy is the graph of a function of y on a
            # line less two; the next two are one or two lines through the origin, or the
            # line x + 2y + 3 = 0 doubled; and 13xy has no zero.
            ((0, 5, 7, 0, 13, 17), -2),
            ((0, 5, 7, 0, 13, 0), -1),
            ((0, 0, 0, 11, 13, 17), 0),
            ((0, 0, 0, 0, 13, 17), 0),
            ((9, 6, 12, 1, 4, 4), -1),
            ((0, 0, 0, 0, 13, 0), 0),
        ],
    )
    def test_value(self, backend: str, z: tuple[int, ...], value: int) -> None:
        f = sum(c * m for c, m in zip(z, self.MONOMIALS, strict=True))
        assert _chi([f], [X, Y], backend) == value
