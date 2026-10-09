"""
The benchmark bank: integral families with the numbers published for them.

Each family is one TOML file under ``feynkit/bank/data/``, and the papers are
listed in ``sources.toml`` beside them. A file holds a family (a graph, a
polynomial or a bare reference), exact kinematic points, and published data,
each datum with the paper, the place in it (section, equation or table, and
page) and how the number was checked. No computed value is stored; computed
values belong to the tests that compare them with the published ones.

>>> from feynkit import bank
>>> bank.list_families(area="formal")
>>> family = bank.load("FT1-sunrise2-generic")
>>> family.data_of("master_count")
>>> family.integral()

See ``docs/guide.md`` for the format and the conventions a count depends on.
"""

from .loader import (
    AREAS,
    DATUM_TYPES,
    EXCLUDES,
    KINDS,
    BankError,
    BankFamily,
    Datum,
    Point,
    Source,
    list_families,
    load,
    load_sources,
    parse_family,
    parse_sources,
)

__all__ = [
    "AREAS",
    "DATUM_TYPES",
    "EXCLUDES",
    "KINDS",
    "BankError",
    "BankFamily",
    "Datum",
    "Point",
    "Source",
    "list_families",
    "load",
    "load_sources",
    "parse_family",
    "parse_sources",
]
