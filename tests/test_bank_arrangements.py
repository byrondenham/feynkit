"""
Hyperplane arrangements of the benchmark bank (T6, slow).

The Euler characteristic of the complement of a cosmological chain's arrangement against the
published dimension, its drop on the printed singular factors, and the two published sizes of
the three-site system side by side: 25, the dimension of the full twisted cohomology, and 16,
the size of the closed system of the wavefunction.
"""

from __future__ import annotations

import pytest
import sympy as sp

from feynkit import bank
from feynkit.strata import torus_euler_characteristic
from tests.bank_support import TIMEOUT, requires_singular

pytestmark = requires_singular


def chains() -> list[str]:
    return [
        i
        for i in bank.list_families()
        if i.startswith("CS") and bank.load(i).data_of("cohomology_dimension")
    ]


def arrangement_chi(family: bank.BankFamily, point: str) -> int:
    """|chi| of the complement in C^n of the coordinate hyperplanes and the arrangement."""
    g, variables = family.polynomial_at(point)
    chi = torus_euler_characteristic([], variables, remove=g, timeout=TIMEOUT)
    return abs(chi)


@pytest.mark.slow
@pytest.mark.parametrize("family_id", chains())
def test_cohomology_dimension_is_the_euler_characteristic_of_the_complement(
    family_id: str,
) -> None:
    family = bank.load(family_id)
    for datum in family.data_of("cohomology_dimension"):
        (published,) = datum.numbers
        assert arrangement_chi(family, datum.point) == published


def point_on(family: bank.BankFamily, letter: str, base: str) -> dict[str, int]:
    """A point on the hyperplane {letter = 0}, found by solving for one parameter of the base."""
    parameters = [sp.Symbol(p) for p in family.parameters]
    expression = sp.sympify(letter, locals={str(p): p for p in parameters})
    values = {
        str(p): sp.Rational(v.numerator, v.denominator)
        for p, v in zip(
            parameters, (family.point(base).values[str(q)] for q in parameters), strict=True
        )
    }
    solved = next(p for p in parameters if expression.coeff(p) != 0)
    others = {p: values[str(p)] for p in parameters if p != solved}
    values[str(solved)] = sp.solve(expression.subs(others), solved)[0]
    return values


@pytest.mark.slow
@pytest.mark.parametrize("family_id", chains())
def test_the_characteristic_drops_on_every_printed_factor(family_id: str) -> None:
    # The published factors lie in the Euler discriminant: the characteristic is smaller on
    # each. This does not show that there are no other factors.
    family = bank.load(family_id)
    (dimension,) = family.datum(family.data_of("cohomology_dimension")[0].id).numbers
    expression, variables, parameters = family.polynomial()
    checked = 0
    for datum in family.data_of("singular_locus") + family.data_of("alphabet"):
        for letter in datum.letters:
            values = point_on(family, letter, family.points[0].name)
            others = [
                sp.sympify(other, locals={str(p): p for p in parameters})
                for other in datum.letters
                if other != letter
            ]
            assert all(
                o.subs({sp.Symbol(k): v for k, v in values.items()}) != 0 for o in others
            ), f"the point on {letter} lies on another factor"
            polynomial = sp.expand(expression.subs({sp.Symbol(k): v for k, v in values.items()}))
            chi = abs(torus_euler_characteristic([], variables, remove=polynomial, timeout=TIMEOUT))
            assert chi < dimension, letter
            checked += 1
    assert checked >= 6


@pytest.mark.slow
def test_published_ranks_of_the_chains() -> None:
    # Two sites: the rank of the restricted GKZ system is the characteristic, 4. Three
    # sites: 25 is the dimension of the full twisted cohomology and 16 the size of the closed
    # system of the wavefunction; the characteristic is the first, and the two differ.
    two = bank.load("CS1-two-site-chain")
    assert arrangement_chi(two, "generic") == two.datum("holonomic-rank").numbers[0]
    three = bank.load("CS2-three-site-chain")
    full = three.datum("twisted-cohomology-dimension").numbers[0]
    system = three.datum("wavefunction-system-size").numbers[0]
    assert arrangement_chi(three, "generic") == full
    assert system < full
