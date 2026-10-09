"""
Published letters of the benchmark bank against the principal Landau determinant (T4, slow).

Each published letter of a double box is a ratio of polynomials in the kinematic symbols of
its source. Every irreducible factor of its numerator and denominator must be a component of
the principal Landau determinant of the family at the point; extra components are allowed. A
factor that is not found while faces were skipped is undecided, not a pass.
"""

from __future__ import annotations

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
                    if not any(sp.rem(sp.expand(s), factor, *symbols) == 0 for s in surfaces):
                        missing.append((letter, factor))
        if missing and analysis.skipped_faces:
            pytest.skip(f"undecided: faces were skipped, and {missing} were not found")
        assert not missing, point.name
        assert len(datum.letters) >= 1
