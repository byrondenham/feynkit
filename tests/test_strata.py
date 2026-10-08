"""Tests for feynkit.strata.torus_euler_characteristic.

Oracles: Kouchnirenko's theorem on random full-support polynomials (Invent. Math. 32 (1976)
1-31, Thm IV, p. 30), the conics of Telen and Weinstein, "Euler stratification of hypersurface
families", Ex. 1.2, p. 2, hand computations on curves and point sets whose pieces are read off
a parametrisation, both backends against each other, and stubbed runs for the error paths.
"""

from __future__ import annotations

import random
import re
import subprocess
from collections.abc import Callable
from typing import Any

import pytest
import sympy as sp

from feynkit import strata as strata_module
from feynkit.core.exceptions import ComputationError, ValidationError
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


@requires_singular
@pytest.mark.parametrize("backend", BACKENDS)
class TestExamples:
    """Hand computations; each piece is read off a parametrisation."""

    @pytest.mark.parametrize("generators", [[], [0], [sp.Integer(0), sp.Integer(0)]])
    def test_torus(self, backend: str, generators: list[sp.Expr]) -> None:
        assert _chi(generators, [X, Y], backend) == 0
        assert _chi(generators, [X, Y, Z], backend) == 0

    def test_torus_less_hypersurface(self, backend: str) -> None:
        # The line x + y = 1 meets the torus in the line less two points, chi = -1, so the
        # complement has chi = 0 - (-1).
        assert _chi([], [X, Y], backend, remove=X + Y - 1) == 1

    def test_hypersurface_less_hypersurface(self, backend: str) -> None:
        # The line x + y = 1 less the axes is C less two points; x = y removes one more.
        assert _chi([X + Y - 1], [X, Y], backend) == -1
        assert _chi([X + Y - 1], [X, Y], backend, remove=X - Y) == -2

    def test_removed_set_contains_the_variety(self, backend: str) -> None:
        assert _chi([X + Y - 1], [X, Y], backend, remove=(X + Y - 1) * (X - 2)) == 0
        assert _chi([X + Y - 1], [X, Y], backend, remove=sp.Integer(0)) == 0

    def test_constant_remove_removes_nothing(self, backend: str) -> None:
        assert _chi([X + Y - 1], [X, Y], backend, remove=sp.Integer(7)) == -1

    def test_nodal_curve(self, backend: str, count_calls: Callable[[str], list[int]]) -> None:
        # (y - 1)^2 = (x - 1)^2 (x + 2) has its node at (1, 1) in the torus. With s = (y - 1)
        # / (x - 1) it is x = s^2 - 2, y = 1 + s (s^2 - 3), with s = +-sqrt(3) the node, so
        # the curve has chi = 1 - 1 = 0 in C^2. The torus removes two points on x = 0 and three
        # on y = 0.
        closed = count_calls("closed")
        f = (Y - 1) ** 2 - (X - 1) ** 2 * (X + 2)
        assert _chi([f], [X, Y], backend) == -5
        # Once for the curve, once for its singular locus, at each of two primes.
        assert len(closed) >= 4

    def test_singular_surface(self, backend: str, count_calls: Callable[[str], list[int]]) -> None:
        # (x - 1)^2 = z (y - 1)^2 is singular along the curve x = y = 1. Off y = 1 it is the
        # graph z = ((x - 1)/(y - 1))^2 over x, y not in {0, 1}, chi = 1; on y = 1 it is the
        # curve x = 1, z in C^*, chi = 0.
        closed = count_calls("closed")
        assert _chi([(X - 1) ** 2 - Z * (Y - 1) ** 2], [X, Y, Z], backend) == 1
        assert len(closed) >= 4

    def test_reducible(self, backend: str, count_calls: Callable[[str], list[int]]) -> None:
        # Two lines, each C less two points, meeting at (-1, 2) in the torus.
        opened = count_calls("open")
        assert _chi([(X + Y - 1) * (X - Y + 3)], [X, Y], backend) == -3
        assert len(opened) >= 6

    def test_three_lines(self, backend: str) -> None:
        # Three lines, each C less two points, meeting pairwise in three distinct points of
        # the torus: -3 - 3.
        lines = (X + Y - 1) * (X - Y + 3) * (2 * X + Y - 5)
        assert _chi([lines], [X, Y], backend) == -6

    def test_points(self, backend: str) -> None:
        assert _chi([X**2 - 1, Y - 1], [X, Y], backend) == 2
        assert _chi([X**2 - 1, Y - 1], [X, Y], backend, remove=X - 1) == 1
        # The point with x = 0 is not in the torus.
        assert _chi([X * (X - 1), Y - 1], [X, Y], backend) == 1

    def test_empty(self, backend: str) -> None:
        assert _chi([X * Y], [X, Y], backend) == 0
        assert _chi([sp.Integer(1)], [X, Y], backend) == 0
        assert _chi([X, Y - 1], [X, Y], backend) == 0

    def test_line_in_space(self, backend: str) -> None:
        # Two general planes meet in a line, P^1 less four points.
        assert _chi([X + 2 * Y + 3 * Z - 1, 3 * X - Y + Z - 2], [X, Y, Z], backend) == -2

    def test_twisted_cubic(self, backend: str) -> None:
        # (s, s^2, s^3), s in C^*, with a redundant generator.
        gens = [Y - X**2, Z - X * Y, X * Z - Y**2]
        assert _chi(gens, [X, Y, Z], backend) == 0
        assert _chi(gens, [X, Y, Z], backend, remove=X - 1) == -1

    def test_rank_minors(self, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # The rank 1 matrices ((x, y, z), (y, z, w)) are (x, lambda x, lambda^2 x, lambda^3 x),
        # chi = 0 on the torus. h = x + y + z + w - 1 becomes x q(lambda) = 1 for
        # q = (1 + lambda)(1 + lambda^2), a curve that is C less the four points lambda = 0 and
        # the roots of q, chi = -3.
        modes = []
        original = strata_module._Run.open

        def spy(self: Any, comp: Any) -> int:
            modes.append(comp.count == self.d - comp.dim)
            return original(self, comp)

        monkeypatch.setattr(strata_module._Run, "open", spy)
        gens = [X * Z - Y**2, X * W - Y * Z, Y * W - Z**2]
        assert _chi(gens, [X, Y, Z, W], backend) == 0
        assert _chi(gens, [X, Y, Z, W], backend, remove=X + Y + Z + W - 1) == 3
        # These components are not complete intersections, so the system uses rank minors.
        assert modes
        assert not any(modes)

    def test_seed(self, backend: str) -> None:
        f = (Y - 1) ** 2 - (X - 1) ** 2 * (X + 2)
        assert [_chi([f], [X, Y], backend, seed=s) for s in (1, 2, 3)] == [-5, -5, -5]


@requires_singular
class TestBackendsAgree:
    @requires_msolve
    def test_minors_system(self) -> None:
        gens = [X * Z - Y**2, X * W - Y * Z, Y * W - Z**2]
        remove = X + 2 * Y + 3 * Z + 5 * W - 1
        assert _chi(gens, [X, Y, Z, W], "msolve", remove=remove) == _chi(
            gens, [X, Y, Z, W], "singular", remove=remove
        )


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"backend": "magma"},
            {"seed": 1.5},
            {"seed": True},
            {"timeout": 0},
            {"timeout": -1},
            {"timeout": float("nan")},
            {"timeout": 3_000_000},
            {"timeout": "3"},
            {"remove": sp.sin(X)},
            {"remove": Z},
            {"remove": 1 / X},
        ],
    )
    def test_bad_keywords(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            torus_euler_characteristic([X + Y - 1], [X, Y], **kwargs)

    @pytest.mark.parametrize("variables", [[], [X, X], [X, "y"], "xy", None, [X, X + 1]])
    def test_bad_variables(self, variables: Any) -> None:
        with pytest.raises(ValidationError):
            torus_euler_characteristic([X + Y - 1], variables)

    @pytest.mark.parametrize(
        "generators", [X + Y - 1, "x + y - 1", [Z], [sp.sin(X)], [1 / X], [X ** sp.Rational(1, 2)]]
    )
    def test_bad_generators(self, generators: Any) -> None:
        with pytest.raises(ValidationError):
            torus_euler_characteristic(generators, [X, Y])


def _no_binary() -> None:
    return None


class TestMissingTools:
    def test_singular(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strata_module, "_singular_binary", _no_binary)
        with pytest.raises(RuntimeError, match="needs Singular"):
            torus_euler_characteristic([X + Y - 1], [X, Y])

    def test_msolve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strata_module, "_msolve_binary", _no_binary)
        monkeypatch.setattr(strata_module, "_singular_binary", lambda: "Singular")
        with pytest.raises(RuntimeError, match="needs msolve"):
            torus_euler_characteristic([X + Y - 1], [X, Y], backend="msolve")


def _stub_run(monkeypatch: pytest.MonkeyPatch, tool: str, make: Callable[[list[str]], Any]) -> None:
    """Replaces subprocess.run for runs of ``tool`` by ``make(command)``, the rest as before."""
    real = subprocess.run

    def run(command: list[str], *args: Any, **kwargs: Any) -> Any:
        if tool in command[0]:
            return make(command)
        return real(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)


def _expired(command: list[str]) -> Any:
    raise subprocess.TimeoutExpired(command, 1)


@requires_singular
class TestUndecided:
    """Every ComputationError for an input the computation cannot decide says "undecided: "."""

    def test_singular_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_run(monkeypatch, "ingular", _expired)
        with pytest.raises(ComputationError, match=r"^undecided: Singular ran past timeout=5"):
            torus_euler_characteristic([X + Y - 1], [X, Y], timeout=5)

    def test_singular_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(command: list[str]) -> Any:
            return subprocess.CompletedProcess(command, 0, "? error occurred\n", "")

        _stub_run(monkeypatch, "ingular", broken)
        with pytest.raises(ComputationError, match=r"^undecided: Singular failed: \? error"):
            torus_euler_characteristic([X + Y - 1], [X, Y])

    def test_singular_exit_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(command: list[str]) -> Any:
            return subprocess.CompletedProcess(command, 3, "", "boom")

        _stub_run(monkeypatch, "ingular", broken)
        with pytest.raises(ComputationError, match=r"^undecided: Singular failed: boom"):
            torus_euler_characteristic([X + Y - 1], [X, Y])

    def test_singular_prints_nothing_useful(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def silent(command: list[str]) -> Any:
            return subprocess.CompletedProcess(command, 0, "nothing here\n", "")

        _stub_run(monkeypatch, "ingular", silent)
        with pytest.raises(ComputationError, match=r"^undecided: Singular printed"):
            torus_euler_characteristic([X + Y - 1], [X, Y])

    def test_primes_disagree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        values = iter([1, 2])
        monkeypatch.setattr(strata_module._Run, "closed", lambda self, gens: next(values))
        with pytest.raises(ComputationError, match=r"^undecided: .* 1 modulo \d+ and 2 modulo"):
            torus_euler_characteristic([X + Y - 1], [X, Y])

    @pytest.mark.parametrize(
        ("pattern", "replacement", "message"),
        [
            (r"CRIT -?\d+", "CRIT 1", "do not form a finite set"),
            (r"AUDIM -?\d+", "AUDIM 5", "vanished on a component"),
            (r"CRIT -?\d+ \d+", "", "instead of a count"),
        ],
    )
    def test_bad_critical_output(
        self, monkeypatch: pytest.MonkeyPatch, pattern: str, replacement: str, message: str
    ) -> None:
        original = strata_module._Run.singular_output

        def altered(self: Any, script: str) -> str:
            return re.sub(pattern, replacement, original(self, script))

        monkeypatch.setattr(strata_module._Run, "singular_output", altered)
        with pytest.raises(ComputationError, match=f"^undecided: .*{message}"):
            torus_euler_characteristic([X + Y - 1], [X, Y])

    def test_u_vanishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = strata_module._Run.singular_output

        def altered(self: Any, script: str) -> str:
            out = original(self, script)
            return out.replace("AU ", "UZERO\nAU ", 1)

        monkeypatch.setattr(strata_module._Run, "singular_output", altered)
        with pytest.raises(ComputationError, match="^undecided: .*vanished on a component"):
            torus_euler_characteristic([X + Y - 1], [X, Y])


@requires_singular
@requires_msolve
class TestMsolveUndecided:
    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_run(monkeypatch, "msolve", _expired)
        with pytest.raises(ComputationError, match=r"^undecided: msolve ran past timeout=5"):
            torus_euler_characteristic([X + Y - 1], [X, Y], timeout=5, backend="msolve")

    def test_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(command: list[str]) -> Any:
            return subprocess.CompletedProcess(command, 1, "", "no memory")

        _stub_run(monkeypatch, "msolve", broken)
        with pytest.raises(ComputationError, match=r"^undecided: msolve failed: no memory"):
            torus_euler_characteristic([X + Y - 1], [X, Y], backend="msolve")

    def test_not_finite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(strata_module, "_msolve_count", lambda output: -1)
        with pytest.raises(ComputationError, match=r"^undecided: .*do not form a finite set"):
            torus_euler_characteristic([X + Y - 1], [X, Y], backend="msolve")

    def test_unexpected_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def garbled(output: str) -> int:
            raise RuntimeError("msolve wrote 'junk'")

        monkeypatch.setattr(strata_module, "_msolve_count", garbled)
        with pytest.raises(ComputationError, match=r"^undecided: msolve printed"):
            torus_euler_characteristic([X + Y - 1], [X, Y], backend="msolve")
