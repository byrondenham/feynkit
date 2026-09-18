"""Tests for syzygy-module computation."""

from __future__ import annotations

import sympy as sp

from feynkit.algebra.syzygy import compute_syzygy_module, trivial_syzygy

x, y, z = sp.symbols("x y z")


def _is_syzygy(coeffs: list[sp.Expr], gens: list[sp.Expr]) -> bool:
    return sp.expand(sum(c * f for c, f in zip(coeffs, gens))) == 0


def _proportional(a: list[sp.Expr], b: list[sp.Expr]) -> bool:
    ratios = {sp.cancel(p / q) for p, q in zip(a, b) if q != 0}
    zeros_match = all((p == 0) == (q == 0) for p, q in zip(a, b))
    return zeros_match and len(ratios) == 1 and next(iter(ratios)).is_number


class TestComputeSyzygyModule:
    def test_every_returned_relation_is_a_syzygy(self) -> None:
        gens = [x**2 - y * z, x * z - y**2, x * y - z**2]
        for s in compute_syzygy_module(gens, [x, y, z]):
            assert len(s) == len(gens)
            assert _is_syzygy(s, gens)

    def test_koszul_syzygy_of_two_variables(self) -> None:
        syz = compute_syzygy_module([x, y], [x, y])
        assert any(_proportional(s, [y, -x]) for s in syz)

    def test_single_generator_has_no_nonzero_syzygies(self) -> None:
        assert compute_syzygy_module([x**2 + y], [x, y]) == []

    def test_no_zero_rows_returned(self) -> None:
        syz = compute_syzygy_module([x * y, x * z, y * z], [x, y, z])
        assert syz
        assert all(any(c != 0 for c in s) for s in syz)

    def test_dependent_generators_yield_polynomial_coefficients(self) -> None:
        syz = compute_syzygy_module([x**2 - 1, x - 1], [x])
        assert any(_proportional(s, [1, -(x + 1)]) for s in syz)

    def test_toric_generators_have_syzygies(self) -> None:
        # Twisted cubic: three quadrics with two independent linear syzygies.
        gens = [x**2 - y * z, x * z - y**2, x * y - z**2]
        syz = compute_syzygy_module(gens, [x, y, z])
        assert len(syz) >= 2


class TestTrivialSyzygy:
    def test_layout(self) -> None:
        f0, f1 = sp.symbols("f_0 f_1")
        assert trivial_syzygy(0, 1, 3) == [f1, -f0, 0]
