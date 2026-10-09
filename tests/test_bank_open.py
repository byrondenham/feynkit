"""
Open and disputed data of the benchmark bank (T7).

Such data is run and reported; a mismatch with the published numbers never fails a test, a
crash or a malformed result does. The report goes to FEYNKIT_BANK_REPORT_DIR when that is
set, else to a temporary directory.
"""

from __future__ import annotations

import pytest

from feynkit import bank
from tests.bank_support import (
    count_at,
    graph_data,
    newton_volume,
    report_dir,
    requires_singular,
    tadpole_products,
    write_report,
)

pytestmark = requires_singular


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    return report_dir(tmp_path_factory)


@pytest.mark.bank_open
@pytest.mark.parametrize(
    ("family_id", "datum_id"),
    graph_data(("master_count",), ("open", "disputed")),
    ids=lambda x: x,
)
def test_open_and_disputed_counts_are_reported(
    family_id: str, datum_id: str, reports  # noqa: ANN001
) -> None:
    # Whatever the numbers are, they are written down; only a crash fails.
    family = bank.load(family_id)
    datum = family.datum(datum_id)
    rows = []
    for point in family.points:
        computed = count_at(family, point.name)
        tadpoles = tadpole_products(family, point.name)
        rows.append(
            {
                "point": point.name,
                "euler_characteristic": computed,
                "tadpole_products": tadpoles,
                "top_sector": computed - tadpoles,
                "newton_volume": newton_volume(family, point.name),
            }
        )
    path = write_report(
        reports,
        f"{family_id}.{datum_id}",
        {"published": list(datum.numbers), "status": datum.status, "rows": rows},
    )
    assert path.is_file()
    assert all(isinstance(v, (int, str)) for row in rows for v in row.values())
    for row in rows:
        assert row["top_sector"] >= 0, row
        assert row["euler_characteristic"] <= row["newton_volume"], row
    assert any(row["point"] == datum.point for row in rows)
