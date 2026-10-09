"""
Published letters of the benchmark bank against the principal Landau determinant (T4, slow).

Each published letter of a double box is a ratio of polynomials in the kinematic symbols of
its source. Every irreducible factor of its numerator and denominator must be a component of
the principal Landau determinant of the family at the point; extra components are allowed. A
factor that is not found while faces were skipped is undecided, not a pass.
"""

from __future__ import annotations

import copy

import pytest
import sympy as sp

from feynkit import bank
from feynkit.landau import landau_analysis
from tests.bank_support import TIMEOUT, requires_singular

pytestmark = requires_singular


def landau_families() -> list[str]:
    return [
        i
        for i in bank.list_families()
        if bank.load(i).kind == "graph" and bank.load(i).data_of("alphabet")
    ]


def divides_a_surface(factor: sp.Expr, surfaces: list[sp.Expr], symbols: list[sp.Symbol]) -> bool:
    """Whether the factor divides one of the surfaces; a symbol outside theirs is an error."""
    foreign = factor.free_symbols - set(symbols)
    assert not foreign, f"{factor} uses {foreign}, which is no kinematic symbol of the surfaces"
    return any(sp.rem(sp.expand(s), factor, *symbols) == 0 for s in surfaces)


@pytest.mark.slow
@pytest.mark.parametrize("family_id", landau_families())
def test_published_letters_divide_landau_components(family_id: str) -> None:
    # A letter is a ratio of polynomials in the kinematic symbols of the source. Each
    # irreducible factor of its numerator and denominator must divide, hence be, a component
    # of the principal Landau determinant of the family at the point. Extra components are
    # allowed. A factor that is not found while faces were skipped is undecided.
    family = bank.load(family_id)
    for datum in family.data_of("alphabet"):
        point = family.point(datum.point)
        analysis = landau_analysis(
            family.integral(datum.point),
            timeout=TIMEOUT,
            large_face_timeout=TIMEOUT,
            total_timeout=5 * TIMEOUT,
        )
        names = {k: sp.Symbol(v, real=True) for k, v in family.letter_symbols.items()}
        surfaces = list(analysis.landau_surfaces)
        symbols = sorted(set().union(*(s.free_symbols for s in surfaces)), key=str)
        missing = []
        for letter in datum.letters:
            numerator, denominator = sp.fraction(sp.together(sp.sympify(letter, locals=names)))
            for part in (numerator, denominator):
                for factor, _ in sp.factor_list(part)[1]:
                    if not divides_a_surface(factor, surfaces, symbols):
                        missing.append((letter, factor))
        if missing and analysis.skipped_faces:
            pytest.skip(f"undecided: faces were skipped, and {missing} were not found")
        assert not missing, point.name
        assert len(datum.letters) >= 1


def test_a_misspelt_letter_symbol_is_caught() -> None:
    # With t spelt "s32", the factor of t/s has a symbol no surface contains, and the test
    # must fail on it instead of dividing by a coefficient. The loader rejects the same.
    s12, s23, s32 = sp.symbols("s12 s23 s32", real=True)
    assert sp.rem(s12, s32, s12, s23) == 0  # the false pass without the guard
    with pytest.raises(AssertionError):
        divides_a_surface(s32, [s12], [s12, s23])
    assert divides_a_surface(s12, [s12], [s12, s23])
    from tests.test_bank_schema import raw

    table = copy.deepcopy(raw("CO1-planar-dbox-onshell"))
    sources = dict(bank.load_sources())
    table["family"]["letter_symbols"] = {"s": "s12", "t": "s32"}
    with pytest.raises(bank.BankError):
        bank.parse_family(table, sources)
    table["family"]["letter_symbols"] = {"s": "s12"}
    with pytest.raises(bank.BankError):
        bank.parse_family(table, sources)
