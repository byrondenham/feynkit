"""
Counts of the benchmark bank against the numbers published for the same families.

A count of the Euler characteristic, |chi| of the complement of {G = 0} in the torus, is the
number of master integrals with subsectors included and symmetries unused (Bitoun, Bogner,
Klausen and Panzer, arXiv:1712.09215, Cor. 37). The sources count differently: the top
sector only, with symmetries between lines, or for a whole family. tests/bank_support.py
holds the adapter that puts the computed count in the convention of each datum. Where a
source used symmetries, the computed count must be at least the published one.

The first tier is in the default suite: the two-loop sunrise, and the double box against the
pair (|chi|, volume) of Fevola, Mizera and Telen. The second is marked slow and records the
count of every other published master count it can reach.
"""

from __future__ import annotations

import pytest

from feynkit import bank
from tests.bank_support import (
    adapt,
    count_at,
    graph_data,
    newton_volume,
    report_dir,
    requires_singular,
    write_report,
)

pytestmark = requires_singular

# The two-loop sunrise at three different masses and at equal masses is cheap, and in the
# default suite; the other families run under -m slow.
FAST = {
    ("FT1-sunrise2-generic", "master-count-generic"),
    ("FT1-sunrise2-generic", "master-count-equal-mass"),
}


def tiered(pairs: list[tuple[str, str]]) -> list[pytest.ParameterSet]:
    return [
        pytest.param(f, d, id=f"{f}-{d}", marks=() if (f, d) in FAST else pytest.mark.slow)
        for f, d in pairs
    ]


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    return report_dir(tmp_path_factory)


@pytest.mark.parametrize(
    ("family_id", "datum_id"), tiered(graph_data(("master_count",), ("published",)))
)
def test_master_count(family_id: str, datum_id: str, reports) -> None:  # noqa: ANN001
    family = bank.load(family_id)
    datum = family.datum(datum_id)
    (published,) = datum.numbers
    computed = count_at(family, datum.point)
    value, relation = adapt(family, datum, computed)
    write_report(
        reports,
        f"{family_id}.{datum_id}",
        {
            "computed_family_count": computed,
            "in_convention": value,
            "relation": relation,
            "published": published,
            "source": datum.source,
            "location": datum.location,
        },
    )
    if relation == "==":
        assert value == published
    else:
        assert value >= published


def test_double_box_matches_the_pair_of_table_1() -> None:
    # Fevola, Mizera and Telen, Tab. 1: (|chi|, volume) = (12, 238) for the planar double box
    # with massless propagators and legs. |chi| below the volume is the drop of the count.
    family = bank.load("CO1-planar-dbox-onshell")
    (chi,) = family.datum("euler-characteristic").numbers
    (volume,) = family.datum("volume").numbers
    assert count_at(family, "on-shell") == chi
    assert newton_volume(family, "on-shell") == volume
    assert chi < volume
