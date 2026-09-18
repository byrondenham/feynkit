"""Tests for Gröbner-basis helpers, in particular the ideal quotient."""

from __future__ import annotations

import sympy as sp

from feynkit.algebra.groebner import ideal_quotient, is_in_ideal

x, y, z = sp.symbols("x y z")
XYZ = [x, y, z]


def _same_ideal(gens_a: list[sp.Expr], gens_b: list[sp.Expr]) -> bool:
    ga = sp.groebner(gens_a, *XYZ, order="grevlex") if gens_a else None
    gb = sp.groebner(gens_b, *XYZ, order="grevlex") if gens_b else None
    if ga is None or gb is None:
        return ga is gb
    return list(ga.exprs) == list(gb.exprs)


class TestIdealQuotient:
    def test_principal_quotient_cancels_common_factor(self) -> None:
        assert _same_ideal(ideal_quotient([x * y], [x], XYZ), [y])

    def test_quotient_of_monomial_ideal(self) -> None:
        assert _same_ideal(ideal_quotient([x**2, x * y], [x], XYZ), [x, y])

    def test_quotient_by_unit_ideal_is_identity(self) -> None:
        assert _same_ideal(ideal_quotient([x * y, y * z], [sp.Integer(1)], XYZ), [x * y, y * z])

    def test_quotient_intersects_over_generators(self) -> None:
        # <xy, yz> : <y> = <x, z>
        assert _same_ideal(ideal_quotient([x * y, y * z], [y], XYZ), [x, z])

    def test_quotient_by_two_generators(self) -> None:
        # <x^2 y, x y^2> : <x, y> = <xy>
        assert _same_ideal(ideal_quotient([x**2 * y, x * y**2], [x, y], XYZ), [x * y])

    def test_quotient_by_zero_ideal_is_whole_ring(self) -> None:
        assert _same_ideal(ideal_quotient([x], [], XYZ), [sp.Integer(1)])

    def test_zero_ideal_quotient_is_zero(self) -> None:
        assert ideal_quotient([], [x], XYZ) == []

    def test_every_result_times_divisor_lies_in_ideal(self) -> None:
        ideal = [x**2 - y * z, x * z - y**2]
        divisor = [x * y - z**2, x]
        quotient = ideal_quotient(ideal, divisor, XYZ)
        basis = list(sp.groebner(ideal, *XYZ, order="grevlex").exprs)
        for q in quotient:
            for g in divisor:
                assert is_in_ideal(sp.expand(q * g), basis, XYZ, order="grevlex")
