"""Tests for feynkit.strata.stratum_sum and stratum_sum_from_polynomial.

The identity is the toric, torus-cut form of the formula of Parusinski and Pragacz (J. Algebraic
Geom. 4 (1995) 337-351, Prop. 7): vol(P) - |chi| equals the sum over the strata S of the singular
loci of mu^T_S chi(S minus V(H)). The oracles are hand examples and published numbers:
- the massless one-loop box on shell, G = U + s x1 x3 + t x2 x4: every face polynomial is smooth in
  its orbit, so the sum is empty and vol = |chi| = 3;
- the vacuum sunrise with one massive line, G = (1 + m x1)(x1 x2 + x1 x3 + x2 x3): one curve of
  singular points with mu^T = -1 and chi(C minus V(H)) = -2, so the drop 3 - 1 = 2 is (-1)(-2);
- the sunsets with one, two and three massive lines, whose volumes 3, 6 and 10 and numbers of
  master integrals 2, 4 and 7 are in Table 1, p. 34 of the arXiv version, of R. P. Klausen,
  JHEP 04 (2020) 121 (arXiv:1910.08651), so the drop is 1, 2 and 3;
- three lines through a point and a non-reduced surface, where the identity itself is the oracle:
  the volume and the critical points are counted independently of the strata.
"""

from __future__ import annotations

import dataclasses
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit import strata as strata_module
from feynkit.core.exceptions import ComputationError, ValidationError
from feynkit.landau import _singular_binary
from feynkit.point_count import _msolve_binary
from feynkit.strata import (
    StratumSum,
    singular_strata_from_polynomial,
    stratum_sum,
    stratum_sum_from_polynomial,
)

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")
requires_msolve = pytest.mark.skipif(_msolve_binary() is None, reason="msolve not installed")

X1, X2, X3, X4 = sp.symbols("x1:5")

SUNRISE = sp.expand((1 + 5306 * X1) * (X1 * X2 + X1 * X3 + X2 * X3))


def sunset(cnickel: str) -> tuple[FeynmanIntegral, dict[sp.Expr, Fraction]]:
    """A sunset and a rational kinematic point with distinct masses."""
    fi = FeynmanIntegral.from_cnickel(cnickel)
    symbols = {s.name: s for s in fi.symanzik.g.free_symbols}
    point: dict[sp.Expr, Fraction] = {symbols["s"]: Fraction(7)}
    for k, name in enumerate(sorted(n for n in symbols if n.startswith("m_"))):
        point[symbols[name] ** 2] = Fraction(3 + 3 * k + k * k)
    return fi, point


# --- the identity on published and hand examples --------------------------------------------


@requires_singular
def test_the_massless_box_on_shell_has_no_strata_and_zero_drop() -> None:
    """0 = 3 - 3: every face polynomial of the box on shell is smooth in its orbit."""
    box = sp.expand(X1 + X2 + X3 + X4 + 3 * X1 * X3 + 7 * X2 * X4)
    r = stratum_sum_from_polynomial(box, [X1, X2, X3, X4], timeout=60)
    assert (r.volume, r.master_count, r.drop) == (3, 3, 0)
    assert r.strata == () and r.total == 0 and r.agrees is True
    assert r.transverse is True and r.complete and r.reason is None


@requires_singular
def test_the_vacuum_sunrise_with_one_massive_line_is_a_curve() -> None:
    """2 = 3 - 1 = (-1)(-2): the curve of singular points has mu^T = -1 and
    chi(C minus V(H)) = -2; the three points on facets have mu^T = 0."""
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert (r.volume, r.master_count, r.drop, r.total) == (3, 1, 2, 2)
    assert r.agrees is True and r.transverse is True and r.reason is None
    (curve,) = [s for s in r.strata if s.dimension == 1]
    assert (curve.mu_t, curve.euler) == (-1, -2)
    assert all(s.euler is None for s in r.strata if s is not curve)
    assert sum((s.mu_t or 0) * s.euler for s in r.strata if s.euler is not None) == 2
    assert r.seed == 0 and r.primes and list(r.primes) == sorted(r.primes)
    assert all(p < 2**29 for p in r.primes)


@requires_singular
@pytest.mark.parametrize(
    ("cnickel", "volume", "masters", "total"),
    [("111e|e|:nzz", 3, 2, 1), ("111e|e|:nnz", 6, 4, 2), ("111e|e|:nnn", 10, 7, 3)],
)
def test_klausens_sunsets(cnickel: str, volume: int, masters: int, total: int) -> None:
    """Volumes 3, 6, 10 and 2, 4, 7 master integrals, Table 1 of Klausen (2020); one point per massive
    line has mu^T = 1, so the totals are 1, 2, 3."""
    fi, point = sunset(cnickel)
    r = stratum_sum(fi, point)
    assert (r.volume, r.master_count, r.total) == (volume, masters, total)
    assert r.drop == total and r.agrees is True and r.transverse is True
    assert [s.mu_t for s in r.strata if s.euler is not None] == [1] * total


@requires_singular
def test_lines_and_a_surface_with_pieces_cut_out_of_each_other() -> None:
    """G = (x - 1)(y - 1)(1 + z): three lines through a point, so the pieces are three lines with
    the point cut out and the point; and G = (x - 1)^2 (y + z + 1), a plane with a curve in it."""
    lines = stratum_sum_from_polynomial((X1 - 1) * (X2 - 1) * (1 + X3), [X1, X2, X3], timeout=60)
    assert lines.agrees is True and lines.total == lines.drop == 5
    assert sorted(s.euler for s in lines.strata if s.euler is not None) == [-2, -2, -2, 1]
    surface = stratum_sum_from_polynomial((X1 - 1) ** 2 * (X2 + X3 + 1), [X1, X2, X3], timeout=60)
    assert surface.agrees is True and surface.total == surface.drop
    assert {s.dimension for s in surface.strata if s.euler is not None} >= {2, 1}


@requires_singular
def test_a_lattice_of_index_two_scales_every_term() -> None:
    """Substituting x1 -> x1^2 doubles the volume, |chi| and the sum, since the torus covers the
    torus of the lattice spanned by the exponents twice."""
    plain = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    doubled = stratum_sum_from_polynomial(SUNRISE.subs(X1, X1**2), [X1, X2, X3], timeout=60)
    assert (doubled.volume, doubled.master_count, doubled.total) == (6, 2, 4)
    assert doubled.agrees is True
    assert doubled.drop == 2 * plain.drop


@requires_singular
def test_the_seed_changes_h_and_not_the_result() -> None:
    a = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60, seed=3)
    assert a.seed == 3 and a.total == 2 and a.agrees is True
    (curve,) = [s for s in a.strata if s.dimension == 1]
    assert curve.euler == -2


@requires_singular
@requires_msolve
def test_the_msolve_backend_gives_the_same_sum() -> None:
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60, backend="msolve")
    assert (r.volume, r.master_count, r.total, r.agrees) == (3, 1, 2, True)


# --- when the sum cannot be trusted: agrees is None, never True ----------------------------


def spy_analysis(patch: pytest.MonkeyPatch, change: Any) -> None:
    """Make _analyse return the analysis changed by ``change``."""
    original = strata_module._analyse

    def changed(*args: Any, **kw: Any) -> Any:
        return change(original(*args, **kw))

    patch.setattr(strata_module, "_analyse", changed)


@requires_singular
def test_an_undecided_piece_leaves_agrees_undecided(monkeypatch: pytest.MonkeyPatch) -> None:
    """The curve is undecided, so the sum is partial: total is None and agrees is None, with the
    reason of the piece."""

    def undecide(a: Any) -> Any:
        strata = tuple(
            (
                dataclasses.replace(s, mu_t=None, reason="Singular ran past timeout=60 s")
                if s.dimension == 1
                else s
            )
            for s in a.strata
        )
        return dataclasses.replace(a, strata=strata, complete=False)

    spy_analysis(monkeypatch, undecide)
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert r.agrees is None and r.total is None and not r.complete
    assert r.reason is not None and "Singular ran past timeout=60 s" in r.reason
    assert (r.volume, r.master_count) == (3, 1)


@requires_singular
def test_a_piece_with_a_reason_blocks_agreement_even_when_the_rest_adds_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undecided piece of mu^T = 0 would not change the total, but it may hide a jump:
    agrees stays None although the decided part already equals the drop."""
    box = sp.expand(X1 + X2 + X3 + X4 + 3 * X1 * X3 + 7 * X2 * X4)

    def add_piece(a: Any) -> Any:
        extra = strata_module.Stratum(
            (0, 1, 2), None, (), None, (), None, None, None, "Singular ran past timeout=60 s"
        )
        return dataclasses.replace(a, strata=(*a.strata, extra), complete=False)

    spy_analysis(monkeypatch, add_piece)
    r = stratum_sum_from_polynomial(box, [X1, X2, X3, X4], timeout=60)
    assert r.agrees is None and r.total is None and r.reason is not None


@requires_singular
def test_incomplete_strata_leave_agrees_undecided_with_the_total_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """complete=False without any undecided piece (a failed condition, or a subdivided face): the
    sum is computed and equals the drop, but it is not proved to be the whole sum."""
    spy_analysis(monkeypatch, lambda a: dataclasses.replace(a, complete=False))
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert r.total == r.drop == 2 and r.agrees is None and r.complete is False
    assert r.reason is not None and "not known to be complete" in r.reason


@requires_singular
def test_a_missing_or_wrong_piece_is_a_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    """With complete strata a total that differs from the drop is False, the check that finds a
    jump of mu^T that the candidate set missed."""
    spy_analysis(
        monkeypatch,
        lambda a: dataclasses.replace(a, strata=tuple(s for s in a.strata if s.dimension != 1)),
    )
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert r.total == 0 and r.drop == 2 and r.agrees is False and r.reason is None


@requires_singular
def test_an_undecided_euler_characteristic_is_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kw: Any) -> Any:
        raise strata_module._undecided("Singular ran past timeout=1 s")

    monkeypatch.setattr(strata_module, "_piece", refuse)
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert r.total is None and r.agrees is None and r.transverse is None
    assert r.reason is not None and "Singular ran past timeout=1 s" in r.reason


@requires_singular
def test_a_non_transverse_h_is_reported_and_blocks_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(strata_module._Run, "transverse", lambda *args, **kw: False)
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert r.transverse is False and r.agrees is None
    assert r.reason is not None and "transverse" in r.reason


@requires_singular
def test_a_non_transverse_h_is_drawn_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """H fails the first draw, a derived seed gives a new H, and the second draw agrees."""
    draws: list[int] = []
    original_draw = strata_module._random_h
    original_check = strata_module._Run.transverse

    def draw(data: Any, seed: int, attempt: int = 0) -> Any:
        draws.append(attempt)
        return original_draw(data, seed, attempt)

    def check(self: Any, *args: Any, **kw: Any) -> bool:
        return len(draws) >= 2 and original_check(self, *args, **kw)

    monkeypatch.setattr(strata_module, "_random_h", draw)
    monkeypatch.setattr(strata_module._Run, "transverse", check)
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert draws == [0, 1]
    assert r.transverse is True and r.agrees is True and r.reason is None


@requires_singular
def test_every_draw_failing_leaves_agrees_undecided_never_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws: list[int] = []
    original_draw = strata_module._random_h
    monkeypatch.setattr(
        strata_module,
        "_random_h",
        lambda data, seed, attempt=0: (draws.append(attempt), original_draw(data, seed, attempt))[
            1
        ],
    )
    monkeypatch.setattr(strata_module._Run, "transverse", lambda *args, **kw: False)
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert draws == [0, 1, 2, 3]
    assert r.agrees is None and r.transverse is False
    assert r.reason is not None and "4 draws" in r.reason


@requires_singular
def test_h_through_the_end_of_a_counted_piece_is_not_a_disagreement() -> None:
    """At seed 79, H vanishes at a point with mu^T = 0 that closes a counted line. That is a
    non-generic H, not a wrong stratum: it must be drawn again, never reported as False."""
    x, y, z = sp.symbols("x y z")
    r = stratum_sum_from_polynomial((x - 1) * (y - 1) * (1 + z), [x, y, z], seed=79, timeout=60)
    assert r.agrees is not False
    assert r.agrees is True and r.total == r.drop == 5 and r.reason is None


@requires_singular
def test_transversality_fails_for_h_through_a_point_stratum() -> None:
    """A point stratum is transverse to V(h) only if it is off V(h)."""
    run = strata_module._Run(
        prime=536870909,
        dimension=2,
        removed="v0-1",
        seed=0,
        backend="singular",
        timeout=60,
        singular=_singular_binary() or "",
        msolve=None,
    )
    assert run.transverse("v0-1,v1-2", 0, []) is False
    assert run.transverse("v0-2,v1-2", 0, []) is True
    # a curve v1 = 2 against h = v0 - 1: transverse at v0 = 1; tangent for h = (v1 - 2) + v0 - 1
    assert run.transverse("v1-2", 1, []) is True
    tangent = strata_module._Run(
        prime=536870909,
        dimension=2,
        removed="(v1-2)*(v0-3)",
        seed=0,
        backend="singular",
        timeout=60,
        singular=_singular_binary() or "",
        msolve=None,
    )
    assert tangent.transverse("v1-2", 1, []) is False


@requires_singular
def test_counts_that_disagree_between_seeds_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = iter([3, 4])
    monkeypatch.setattr(strata_module, "critical_point_count", lambda *a, **k: next(counts))
    with pytest.raises(ComputationError, match="critical points"):
        stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)


# --- the arguments --------------------------------------------------------------------------


def test_the_result_is_a_frozen_dataclass_with_a_drop() -> None:
    r = StratumSum(5, 2, (), 3, True, True, (), 0)
    assert r.drop == 3 and r.complete is False and r.reason is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.total = 1  # type: ignore[misc]


@requires_singular
def test_bad_arguments_raise() -> None:
    with pytest.raises(ValidationError, match="backend"):
        stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], backend="gp")
    with pytest.raises(ValidationError, match="seed"):
        stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], seed=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=0)
    with pytest.raises(ValidationError, match="full-dimensional"):
        stratum_sum_from_polynomial(1 + X1 + X2, [X1, X2, X3])


def test_kinematic_constraints_are_refused() -> None:
    s = sp.Symbol("s")
    fi = FeynmanIntegral.from_cnickel("11e|e|:zz", kinematic_constraints=[s - 1])
    with pytest.raises(ValidationError, match="kinematic_constraints"):
        stratum_sum(fi)


@requires_singular
def test_the_strata_are_those_of_singular_strata() -> None:
    """The sum reports the pieces of singular_strata_from_polynomial, with euler filled in."""
    analysis = singular_strata_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    r = stratum_sum_from_polynomial(SUNRISE, [X1, X2, X3], timeout=60)
    assert [dataclasses.replace(s, euler=None) for s in r.strata] == list(analysis.strata)


GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide.md"


@requires_singular
def test_guide_examples_print_what_the_guide_says(capsys: pytest.CaptureFixture[str]) -> None:
    section = GUIDE.read_text(encoding="utf-8").split("\n## Strata and the stratum sum\n", 1)[1]
    section = section.split("\n## ", 1)[0]
    first, second = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    first_printed = section.split("prints\n\n```\n", 1)[1].split("```", 1)[0]
    second_printed = section.split("prints `", 1)[1].split("`", 1)[0]
    exec(compile(first, "docs/guide.md", "exec"), {})
    assert capsys.readouterr().out == first_printed
    exec(compile(second, "docs/guide.md", "exec"), {})
    assert capsys.readouterr().out.strip() == second_printed
