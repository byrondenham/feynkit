"""Tests for backend="msolve" of feynkit.point_count.critical_point_count.

Oracles: the numbers of master integrals of the banana graphs, 2^E - 1 for E
lines (Fevola, Mizera and Telen, arXiv:2311.16219, Sec. 4.2), the Singular
backend on the same systems, and stubbed runs for the error paths.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit import point_count as point_count_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.landau import _singular_binary
from feynkit.point_count import _msolve_binary, critical_point_count
from feynkit.sectors import fixed_point_euler_characteristic, sector_hierarchy

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")
requires_msolve = pytest.mark.skipif(_msolve_binary() is None, reason="msolve not installed")

X, Y = sp.symbols("x y")
PRIMES = [2, 3, 5, 7, 11, 13, 17]


def _system(code: str) -> tuple[sp.Expr, list[sp.Symbol], dict[sp.Expr, int]]:
    """G of the graph, its variables, and the point with s = 13 and squared masses 2, 3, 5, ..."""
    fi = FeynmanIntegral.from_cnickel(code)
    g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
    masses = sorted((x for x in g.free_symbols if x.name.startswith("m_")), key=str)
    point: dict[sp.Expr, int] = {m**2: PRIMES[i] for i, m in enumerate(masses)}
    for x in g.free_symbols - set(masses) - set(fi.symanzik.lp_parameters):
        point[x] = 13
    return g, list(fi.symanzik.lp_parameters), point


def _banana(edges: int) -> str:
    return "1" * edges + "e|e|:" + "n" * edges


@requires_msolve
class TestCounts:
    @pytest.mark.parametrize("edges", [2, 3, 4, 5])
    def test_bananas(self, edges: int) -> None:
        g, variables, point = _system(_banana(edges))
        assert critical_point_count(g, variables, point, backend="msolve") == 2**edges - 1

    @requires_singular
    @pytest.mark.parametrize("edges", [2, 3, 4])
    def test_agrees_with_singular(self, edges: int) -> None:
        g, variables, point = _system(_banana(edges))
        assert critical_point_count(g, variables, point, backend="msolve") == critical_point_count(
            g, variables, point, backend="singular"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("code", "count"),
        [("123e|23|3e|e|:nnnnnn", 181), ("123e|24|3|4|e|:nnnnnnn", 208)],
    )
    def test_larger_systems(self, code: str, count: int) -> None:
        g, variables, point = _system(code)
        assert critical_point_count(g, variables, point, backend="msolve") == count

    @requires_singular
    def test_fixed_point_euler_characteristic(self) -> None:
        g, variables, point = _system(_banana(2))
        numeric = g.subs(point)
        identity = [0, 1]
        assert fixed_point_euler_characteristic(
            numeric, variables, identity, backend="msolve"
        ) == fixed_point_euler_characteristic(numeric, variables, identity)

    @requires_singular
    def test_sector_hierarchy_matches_default(self) -> None:
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        a = sector_hierarchy(fi, counts="critical", timeout=120)
        b = sector_hierarchy(fi, counts="critical", timeout=120, backend="msolve")
        assert a.sectors == b.sectors
        assert fi.sectors(counts="critical", backend="msolve").sectors == b.sectors


class TestErrors:
    def test_unknown_backend(self) -> None:
        with pytest.raises(ValidationError, match="backend"):
            critical_point_count(X**2 + 1, [X], {}, backend="magma")
        fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
        with pytest.raises(ValidationError, match="backend"):
            sector_hierarchy(fi, counts=None, backend="magma")
        with pytest.raises(ValidationError, match="backend"):
            fixed_point_euler_characteristic(X + 1, [X], [0], backend="magma")

    def test_missing_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(point_count_module, "_msolve_binary", lambda: None)
        with pytest.raises(RuntimeError, match="msolve"):
            critical_point_count(X + 1, [X], {}, backend="msolve")

    @staticmethod
    def _stub(
        monkeypatch: pytest.MonkeyPatch, write: Callable[[Path], str | None], code: int = 0
    ) -> list[list[str]]:
        calls: list[list[str]] = []

        def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            text = write(Path(cmd[cmd.index("-o") + 1]))
            return subprocess.CompletedProcess(cmd, code, "", text or "")

        monkeypatch.setattr(point_count_module, "_msolve_binary", lambda: "msolve")
        monkeypatch.setattr(point_count_module.subprocess, "run", run)
        return calls

    @staticmethod
    def _writes(text: str) -> Callable[[Path], None]:
        return lambda path: path.write_text(text) and None

    def test_input_files_and_parsed_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []

        def write(path: Path) -> None:
            seen.append(path.with_suffix(".ms").read_text())
            path.write_text("[0, [1073741789, \n2, \n5, \n['x', 'y'],\n[0, 1]]]:\n")

        calls = self._stub(monkeypatch, write)
        assert critical_point_count(X**2 + 3, [X], {}, backend="msolve") == 5
        assert len(calls) == 2
        for text, call in zip(seen, calls, strict=True):
            names, prime, *generators = text.split("\n")
            assert names == "v0,v1"
            assert int(prime) < 2**30
            assert len(generators) == 3  # two generators and the empty last line
            assert call[1:3] == ["-t", "1"]
        assert seen[0].split("\n")[1] != seen[1].split("\n")[1]

    def test_no_solutions_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub(monkeypatch, self._writes("[-1]:\n"))
        assert critical_point_count(X + 1, [X], {}, backend="msolve") == 0

    def test_positive_dimension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub(monkeypatch, self._writes("[1, 2, -1, []]:\n"))
        with pytest.raises(ComputationError, match="finite"):
            critical_point_count(X + 1, [X], {}, backend="msolve")

    @pytest.mark.parametrize("text", ["", "garbage", "[0, [5, 2]]:", "[2, 1, 3, []]:"])
    def test_malformed_output(self, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
        self._stub(monkeypatch, self._writes(text))
        with pytest.raises(RuntimeError, match="msolve"):
            critical_point_count(X + 1, [X], {}, backend="msolve")

    def test_malformed_output_is_quoted_briefly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub(monkeypatch, self._writes("z" * 5000))
        with pytest.raises(RuntimeError) as info:
            critical_point_count(X + 1, [X], {}, backend="msolve")
        assert len(str(info.value)) < 700

    def test_failure_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub(monkeypatch, lambda path: "boom", code=1)
        with pytest.raises(RuntimeError, match="msolve failed: boom"):
            critical_point_count(X + 1, [X], {}, backend="msolve")

    def test_primes_disagree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answers = iter(["[0, [5, 1, 3, ['x']]]:", "[0, [5, 1, 4, ['x']]]:"])
        self._stub(monkeypatch, lambda path: path.write_text(next(answers)) and None)
        with pytest.raises(ComputationError, match="modulo"):
            critical_point_count(X + 1, [X], {}, backend="msolve")

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def run(cmd: list[str], **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        monkeypatch.setattr(point_count_module, "_msolve_binary", lambda: "msolve")
        monkeypatch.setattr(point_count_module.subprocess, "run", run)
        with pytest.raises(ComputationError, match="timeout"):
            critical_point_count(X + 1, [X], {}, backend="msolve", timeout=5)

    def test_timeout_covers_both_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        budgets: list[float] = []
        clock = iter([0.0, 0.0, 4.0, 4.0, 4.0, 4.0])

        def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            budgets.append(kwargs["timeout"])
            Path(cmd[cmd.index("-o") + 1]).write_text("[-1]:")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(point_count_module, "_msolve_binary", lambda: "msolve")
        monkeypatch.setattr(point_count_module.subprocess, "run", run)
        monkeypatch.setattr(point_count_module.time, "monotonic", lambda: next(clock))
        critical_point_count(X + 1, [X], {}, backend="msolve", timeout=10)
        assert budgets == [10.0, 6.0]
