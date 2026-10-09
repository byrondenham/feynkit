"""
The benchmark bank: the schema, the loader and the rules a datum must meet.

Every number in the bank has a paper, a place in it and a note of how it was checked, and
no number is computed. These tests read the files under feynkit/bank/data/ and also feed
the loader bad tables, which it must reject.
"""

from __future__ import annotations

import copy
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from feynkit import FeynmanIntegral, bank
from feynkit.bank.loader import DATA_DIR

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

FAMILIES = bank.list_families()
FILES = sorted(DATA_DIR.glob("*.toml"))

# Names that do not belong in a repository, written in pieces so that this file does not
# contain them either.
FORBIDDEN = [
    "".join(parts)
    for parts in (
        ("cla", "ude"),
        ("anthr", "opic"),
        ("chat", "gpt"),
        ("open", "ai"),
        ("gem", "ini"),
        ("cop", "ilot"),
        ("ki", "mi"),
    )
]
# Things only the research layer has.
PRIVATE = [r"\bR-\d{3}\b", r"\.research", r"\bcompendium\b", r"\bclaim\b"]


def raw(family_id: str) -> dict:
    with (DATA_DIR / f"{family_id}.toml").open("rb") as handle:
        return tomllib.load(handle)


def good() -> tuple[dict, dict]:
    """A table that passes, to be spoiled one way at a time, with the sources."""
    return copy.deepcopy(raw("FT1-sunrise2-generic")), dict(bank.load_sources())


def test_the_bank_is_not_empty() -> None:
    assert len(FAMILIES) >= 1
    assert set(FAMILIES) == {p.stem for p in FILES if p.name != "sources.toml"}


@pytest.mark.parametrize("family_id", FAMILIES)
def test_family_loads_and_its_file_is_named_by_its_id(family_id: str) -> None:
    family = bank.load(family_id)
    assert family.id == family_id
    assert family.area in bank.AREAS
    assert family.kind in bank.KINDS
    assert family.data


@pytest.mark.parametrize("family_id", FAMILIES)
def test_every_datum_has_a_source_a_place_and_a_check(family_id: str) -> None:
    family = bank.load(family_id)
    for datum in family.data:
        assert datum.source in family.sources, datum.id
        for key in (datum.source, *datum.also):
            source = family.sources[key]
            assert re.fullmatch(r"\d{4}\.\d{4,5}", source.arxiv)
            assert not source.doi or source.doi.startswith("10.")
        assert re.search(r"\bpp?\. ?\d+", datum.location), datum.id
        assert datum.verified in {"text", "image", "cite-only"}
        if datum.verified == "cite-only":
            assert datum.status == "published-via"


@pytest.mark.parametrize("family_id", FAMILIES)
def test_points_are_exact_rationals(family_id: str) -> None:
    table = raw(family_id)
    for point in table.get("point", []):
        for name in ("values", "on_shell"):
            for key, value in point.get(name, {}).items():
                assert isinstance(value, str), (point["name"], key)
                assert re.fullmatch(r"-?\d+(/\d+)?", value), (point["name"], key, value)
                Fraction(value)
    for point in bank.load(family_id).points:
        assert all(isinstance(v, Fraction) for v in point.values.values())


@pytest.mark.parametrize("family_id", FAMILIES)
def test_graph_family_builds_with_its_loop_and_propagator_count(family_id: str) -> None:
    family = bank.load(family_id)
    if family.kind != "graph":
        pytest.skip("not a graph family")
    fi = family.integral()
    assert fi.loop_count == family.loops
    assert len(fi.graph.get_internal_edges()) == family.propagators
    assert fi.cnickel == family.cnickel, "write the canonical CNickel string"


@pytest.mark.parametrize("family_id", FAMILIES)
def test_every_point_gives_an_exact_polynomial(family_id: str) -> None:
    family = bank.load(family_id)
    if family.kind == "reference":
        pytest.skip("a reference family has no polynomial")
    for point in family.points:
        polynomial, variables = family.polynomial_at(point.name)
        assert not polynomial.free_symbols - set(variables)
        assert polynomial.is_polynomial(*variables)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_files_are_ascii_and_free_of_forbidden_words(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.isascii()
    lowered = text.lower()
    for word in FORBIDDEN:
        assert word not in lowered
    for pattern in PRIVATE:
        assert not re.search(pattern, text), pattern


@pytest.mark.parametrize("family_id", FAMILIES)
def test_no_computed_values_are_stored(family_id: str) -> None:
    def keys(node: object) -> list[str]:
        if isinstance(node, dict):
            return [str(k) for k in node] + [k for v in node.values() for k in keys(v)]
        if isinstance(node, list):
            return [k for v in node for k in keys(v)]
        return []

    assert not [k for k in keys(raw(family_id)) if "comput" in k.lower()]


def test_list_families_filters() -> None:
    assert set(bank.list_families(area="formal")) <= set(FAMILIES)
    assert all(bank.load(i).status == "G" for i in bank.list_families(status="G"))
    assert bank.list_families(area="gw") == [i for i in FAMILIES if bank.load(i).area == "gw"]
    with pytest.raises(bank.BankError):
        bank.list_families(area="nowhere")
    with pytest.raises(bank.BankError):
        bank.load("XX0-missing")


# -------- Tables the loader must reject --------


def test_the_unspoiled_table_passes() -> None:
    table, sources = good()
    assert bank.parse_family(table, sources).id == "FT1-sunrise2-generic"


def spoil_no_source(table: dict) -> None:
    del table["datum"][0]["source"]


def spoil_unknown_source(table: dict) -> None:
    table["datum"][0]["source"] = "Nobody99"


def spoil_unknown_also(table: dict) -> None:
    table["datum"][0]["also"] = ["Nobody99"]


def spoil_cite_only_published(table: dict) -> None:
    table["datum"][0]["verified"] = "cite-only"
    table["datum"][0]["status"] = "published"


def spoil_published_via_without_cite_only(table: dict) -> None:
    table["datum"][0]["status"] = "published-via"


def spoil_location_without_page(table: dict) -> None:
    table["datum"][0]["location"] = "Sec. 4, Eq. (4.5)"


def spoil_location_without_label(table: dict) -> None:
    table["datum"][0]["location"] = "somewhere on p. 12"


def spoil_verified(table: dict) -> None:
    table["datum"][0]["verified"] = "believed"


def spoil_no_verified(table: dict) -> None:
    del table["datum"][0]["verified"]


def spoil_checked(table: dict) -> None:
    table["datum"][0]["checked"] = "yesterday"


def spoil_computed_key(table: dict) -> None:
    table["datum"][0]["computed"] = 4


def spoil_float_point(table: dict) -> None:
    table["point"][0]["values"]["s"] = "2.5"


def spoil_float_point_number(table: dict) -> None:
    table["point"][0]["values"]["s"] = 2.5


def spoil_undefined_point(table: dict) -> None:
    table["datum"][0]["point"] = "nowhere"


def spoil_duplicate_datum(table: dict) -> None:
    table["datum"][1]["id"] = table["datum"][0]["id"]


def spoil_several_values(table: dict) -> None:
    table["datum"][0]["value"] = [4, 5]


def spoil_bound(table: dict) -> None:
    table["datum"][0]["value"] = {"lo": 5, "hi": 4}


def spoil_type(table: dict) -> None:
    table["datum"][0]["type"] = "feeling"


def spoil_convention(table: dict) -> None:
    table["datum"][0]["excludes"] = ["whatever"]


def spoil_kind(table: dict) -> None:
    table["family"]["kind"] = "sketch"


def spoil_graph_without_cnickel(table: dict) -> None:
    del table["family"]["cnickel"]


def spoil_no_data(table: dict) -> None:
    table["datum"] = []


SPOILS = [
    spoil_no_source,
    spoil_unknown_source,
    spoil_unknown_also,
    spoil_cite_only_published,
    spoil_published_via_without_cite_only,
    spoil_location_without_page,
    spoil_location_without_label,
    spoil_verified,
    spoil_no_verified,
    spoil_checked,
    spoil_computed_key,
    spoil_float_point,
    spoil_float_point_number,
    spoil_undefined_point,
    spoil_duplicate_datum,
    spoil_several_values,
    spoil_bound,
    spoil_type,
    spoil_convention,
    spoil_kind,
    spoil_graph_without_cnickel,
    spoil_no_data,
]


@pytest.mark.parametrize("spoil", SPOILS, ids=lambda f: f.__name__)
def test_a_bad_table_is_rejected(spoil) -> None:  # noqa: ANN001
    table, sources = good()
    spoil(table)
    with pytest.raises(bank.BankError):
        bank.parse_family(table, sources)


def test_a_cite_only_number_is_accepted_as_published_via() -> None:
    table, sources = good()
    table["datum"][0]["verified"] = "cite-only"
    table["datum"][0]["status"] = "published-via"
    assert bank.parse_family(table, sources).data[0].status == "published-via"


def test_a_list_of_values_is_accepted_when_disputed() -> None:
    table, sources = good()
    table["datum"][0]["value"] = [2, 1]
    table["datum"][0]["status"] = "disputed"
    assert bank.parse_family(table, sources).data[0].numbers == (2, 1)


def test_a_bound_is_accepted() -> None:
    table, sources = good()
    table["datum"][0]["value"] = {"lo": 3, "hi": 4}
    datum = bank.parse_family(table, sources).data[0]
    assert datum.is_bound
    assert datum.numbers == (3, 4)


@pytest.mark.parametrize(
    "bad",
    [
        {"arxiv": "1234", "authors": "A", "title": "T", "venue": "V"},
        {"arxiv": "1612.06637", "authors": "A", "title": "T", "venue": "V", "doi": "11.1/x"},
        {"arxiv": "1612.06637", "authors": "A", "title": "T"},
        {"arxiv": "1612.06637", "authors": "A", "title": "T", "venue": "V", "mood": "x"},
    ],
    ids=["arxiv-pattern", "doi-prefix", "venue-missing", "unknown-key"],
)
def test_a_bad_source_is_rejected(bad: dict) -> None:
    with pytest.raises(bank.BankError):
        bank.parse_sources({"X": bad})


def test_the_default_marker_registry_knows_bank_open() -> None:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "bank_open:" in text


def test_integral_of_a_reference_family_is_refused() -> None:
    table, sources = good()
    table["family"].update(kind="reference", reference_data="elsewhere")
    for key in ("cnickel", "loops", "propagators"):
        table["family"].pop(key)
    family = bank.parse_family(table, sources)
    with pytest.raises(bank.BankError):
        family.integral()
    with pytest.raises(bank.BankError):
        family.polynomial_at("generic")


def test_loader_builds_the_integral_it_names() -> None:
    fi = bank.load("FT1-sunrise2-generic").integral()
    assert isinstance(fi, FeynmanIntegral)
