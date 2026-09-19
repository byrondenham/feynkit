"""Tests for the standard kinematic invariants used when use_mandelstam=True."""

from __future__ import annotations

import sympy as sp

from feynkit.kinematics.mandelstam import standard_invariants
from feynkit.kinematics.momentum import create_momentum_products


def _pair_products_from_symbols(n: int) -> dict[tuple[int, int], sp.Expr]:
    return create_momentum_products(n, use_mandelstam=True)


class TestTwoPoint:
    def test_single_invariant_is_p_squared(self) -> None:
        inv = standard_invariants(2)
        s = sp.Symbol("s", real=True)
        assert inv.mandelstam == {}
        assert inv.external_masses == [s, s]
        assert _pair_products_from_symbols(2) == {(1, 2): -s}


class TestThreePoint:
    def test_only_external_masses(self) -> None:
        inv = standard_invariants(3)
        assert inv.mandelstam == {}
        assert [str(m) for m in inv.external_masses] == ["p1^2", "p2^2", "p3^2"]

    def test_products_follow_from_conservation(self) -> None:
        p1, p2, p3 = standard_invariants(3).external_masses
        prods = _pair_products_from_symbols(3)
        assert sp.expand(prods[(1, 2)] - (p3 - p1 - p2) / 2) == 0
        assert sp.expand(prods[(1, 3)] - (p2 - p1 - p3) / 2) == 0
        assert sp.expand(prods[(2, 3)] - (p1 - p2 - p3) / 2) == 0


class TestFourPoint:
    def test_invariants_are_s_t_and_masses(self) -> None:
        inv = standard_invariants(4)
        assert [str(v) for v in inv.mandelstam.values()] == ["s12", "s23"]
        assert list(inv.mandelstam) == [(1, 3), (2, 4)]
        assert len(inv.external_masses) == 4

    def test_two_particle_invariants_reproduce_s_and_t(self) -> None:
        inv = standard_invariants(4)
        m = inv.external_masses
        prods = _pair_products_from_symbols(4)
        s12 = m[0] + m[1] + 2 * prods[(1, 2)]
        s23 = m[1] + m[2] + 2 * prods[(2, 3)]
        s13 = m[0] + m[2] + 2 * prods[(1, 3)]
        assert sp.expand(s12 - inv.mandelstam[(1, 3)]) == 0
        assert sp.expand(s23 - inv.mandelstam[(2, 4)]) == 0
        # u = sum of masses - s - t
        assert sp.expand(s13 - (sum(m) - inv.mandelstam[(1, 3)] - inv.mandelstam[(2, 4)])) == 0


class TestGeneral:
    def test_invariant_count_is_n_choose_2(self) -> None:
        for n in (3, 4, 5, 6):
            inv = standard_invariants(n)
            assert len(inv.external_masses) + len(inv.mandelstam) == n * (n - 1) // 2

    def test_momentum_conservation_holds_for_every_leg(self) -> None:
        for n in (3, 4, 5, 6):
            inv = standard_invariants(n)
            prods = _pair_products_from_symbols(n)
            for i in range(1, n + 1):
                total = inv.external_masses[i - 1] + sum(
                    prods[(min(i, j), max(i, j))] for j in range(1, n + 1) if j != i
                )
                assert sp.expand(total) == 0, (n, i)

    def test_five_point_uses_cyclic_two_particle_invariants(self) -> None:
        inv = standard_invariants(5)
        assert [str(v) for v in inv.mandelstam.values()] == ["s12", "s123", "s23", "s234", "s34"]
