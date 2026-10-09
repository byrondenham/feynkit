"""
Hyperplane arrangements of the benchmark bank (T6, slow).

The Euler characteristic of the complement of a cosmological chain's arrangement against the
published dimension, its drop on the printed singular factors, and the two published sizes of
the three-site system side by side: 25, the dimension of the full twisted cohomology, and 16,
the size of the closed system of the wavefunction.
"""

from __future__ import annotations

import itertools

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
    # Every printed factor lies in the Euler discriminant: the characteristic is smaller on
    # each. That the list is complete is tested in test_the_minors_give_the_printed_locus.
    family = bank.load(family_id)
    (dimension,) = family.datum(family.data_of("cohomology_dimension")[0].id).numbers
    expression, variables, parameters = family.polynomial()
    checked = 0
    expected = 0
    for datum in family.data_of("singular_locus") + family.data_of("alphabet"):
        expected += len(datum.letters)
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
    assert checked == expected
    assert checked >= 6


def normalised(factor: sp.Expr, parameters: list[sp.Symbol]) -> sp.Expr:
    """A factor with integer coefficients, no common divisor and a positive leading term."""
    poly = sp.Poly(sp.expand(factor), *parameters)
    _, primitive = poly.primitive()
    if primitive.LC() < 0:
        primitive = -primitive
    return primitive.as_expr()


def minor_factors(family: bank.BankFamily) -> set[sp.Expr]:
    """The irreducible factors of the non-zero maximal minors of the matrix M_G.

    The columns of M_G are the hyperplanes of the arrangement, the factors of the polynomial
    and the coordinate hyperplanes, each as the coefficients of the variables and the constant
    term (Fevola and Matsubara-Heo; FPSW24, Sec. 3.2, p. 17). This is exact and uses no solver.
    """
    expression, variables, parameters = family.polynomial()
    forms = [f for f, _ in sp.factor_list(expression)[1]] + list(variables)
    columns = [
        [sp.Poly(f, *variables).coeff_monomial(v) for v in variables]
        + [sp.expand(f.subs(dict.fromkeys(variables, 0)))]
        for f in forms
    ]
    n = len(variables) + 1
    found: set[sp.Expr] = set()
    for chosen in itertools.combinations(columns, n):
        minor = sp.Matrix(chosen).T.det()
        if minor == 0:
            continue
        for factor, _ in sp.factor_list(sp.expand(minor), *parameters)[1]:
            if factor.free_symbols:
                found.add(normalised(factor, list(parameters)))
    return found


def printed(family: bank.BankFamily, datum: bank.Datum) -> set[sp.Expr]:
    parameters = [sp.Symbol(p) for p in family.parameters]
    names = {str(p): p for p in parameters}
    return {normalised(sp.sympify(letter, locals=names), parameters) for letter in datum.letters}


@pytest.mark.parametrize("family_id", chains())
def test_the_minors_give_the_printed_locus(family_id: str) -> None:
    # The factors of the maximal minors of M_G are exactly the printed singular locus, not
    # only a set that contains it; the physical singularities are among them.
    family = bank.load(family_id)
    parameters = [sp.Symbol(p) for p in family.parameters]
    found = {normalised(f, parameters) for f in minor_factors(family)}
    (locus,) = family.data_of("singular_locus")
    letters = printed(family, locus)
    assert len(letters) == len(locus.letters) == locus.numbers[0]
    assert found == letters
    for alphabet in family.data_of("alphabet"):
        assert printed(family, alphabet) <= found


def test_the_system_size_is_not_a_rank() -> None:
    three = bank.load("CS2-three-site-chain")
    assert three.datum("wavefunction-system-size").type == "system_size"
    assert not three.data_of("rank")


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
