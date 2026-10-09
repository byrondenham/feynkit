"""Loader and schema checks of the benchmark bank; see :mod:`feynkit.bank`."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import sympy as sp

from ..core.exceptions import ValidationError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

if TYPE_CHECKING:
    from ..integral import FeynmanIntegral

DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCES_FILE = "sources.toml"

AREAS = ("collider", "gw", "cosmology", "formal")
KINDS = ("graph", "polynomial", "reference")
STATUSES = ("G", "H")
DATUM_TYPES = (
    "master_count",
    "alphabet",
    "singular_locus",
    "function_class",
    "rank",
    "geometry",
    "euler_characteristic",
    "volume",
    "cohomology_dimension",
    "system_size",
)
FUNCTION_CLASSES = ("polylog", "elliptic", "K3", "CY3", "hyperelliptic")
SCOPES = ("family", "sector", "top-sector")
SYMMETRIES = ("used", "not-used", "not-stated", "n/a")
EXCLUDES = ("tadpole-products", "subsectors", "symmetric-images", "trivial-gamma-factors")
DATUM_STATUSES = ("published", "published-via", "disputed", "open")
VERIFIED = ("text", "image", "cite-only")

# A location names a section, equation, table, figure or the like, and a page.
_LABEL = re.compile(
    r"\b(Sec\.|Secs\.|Section|Eq\.|Eqs\.|Tab\.|Table|Fig\.|Figure|Example|Prop\.|Proposition|"
    r"Cor\.|Corollary|Lemma|Thm\.|Theorem|Conclusion|Abstract|Introduction)"
)
_PAGE = re.compile(r"\bpp?\. ?\d+")
_ARXIV = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})$")
_RATIONAL = re.compile(r"^-?\d+(/\d+)?$")
_FAMILY_ID = re.compile(r"^[A-Z]{2}\d+[a-z]?(-[A-Za-z0-9]+)+$")

_FAMILY_KEYS = {
    "id",
    "area",
    "kind",
    "status",
    "tags",
    "title",
    "cnickel",
    "loops",
    "propagators",
    "polynomial",
    "variables",
    "parameters",
    "reference_data",
    "letter_symbols",
    "note",
}
_POINT_KEYS = {"name", "values", "on_shell", "kinematics", "note"}
KINEMATIC_CLASSES = ("generic", "massless_off_shell", "massless_on_shell", "equal_masses")
_DATUM_KEYS = {
    "id",
    "type",
    "point",
    "value",
    "scope",
    "sector",
    "symmetries",
    "excludes",
    "status",
    "source",
    "also",
    "location",
    "verified",
    "checked",
    "derived",
    "note",
    "letters",
    "gap",
}
_DATUM_REQUIRED = ("id", "type", "point", "value", "source", "location", "verified", "checked")
_SOURCE_KEYS = {"arxiv", "version", "authors", "title", "venue", "doi"}
_SOURCE_REQUIRED = ("arxiv", "authors", "title", "venue")


class BankError(ValidationError):
    """Raised when a bank file is malformed or asks for something it cannot give."""


@dataclass(frozen=True)
class Source:
    """A paper: where to find it, not where in it a number is (that is the datum's location)."""

    key: str
    arxiv: str
    authors: str
    title: str
    venue: str
    version: str = ""
    doi: str = ""


@dataclass(frozen=True)
class Point:
    """A named kinematic point with exact values, and the on-shell substitutions to apply."""

    name: str
    values: Mapping[str, Fraction]
    on_shell: Mapping[str, Fraction] = field(default_factory=dict)
    kinematics: str = ""
    note: str = ""


@dataclass(frozen=True)
class Datum:
    """One published number, with the convention it is stated in and where it was read."""

    id: str
    type: str
    point: str
    value: int | str | tuple[int, ...] | Mapping[str, int]
    source: str
    location: str
    verified: str
    checked: str
    scope: str = "family"
    sector: str = "n/a"
    symmetries: str = "n/a"
    excludes: tuple[str, ...] = ()
    status: str = "published"
    also: tuple[str, ...] = ()
    derived: str = ""
    note: str = ""
    gap: str = ""
    letters: tuple[str, ...] = ()

    @property
    def numbers(self) -> tuple[int, ...]:
        """The published integers: one, several that disagree, or the two ends of a bound."""
        if isinstance(self.value, int):
            return (self.value,)
        if isinstance(self.value, tuple):
            return self.value
        if isinstance(self.value, Mapping):
            return (self.value["lo"], self.value["hi"])
        raise BankError(f"datum {self.id} has the non-numeric value {self.value!r}")

    @property
    def is_bound(self) -> bool:
        """Whether the value is an interval [lo, hi] rather than a number."""
        return isinstance(self.value, Mapping)


@dataclass(frozen=True)
class BankFamily:
    """A family of the bank with its points, published data and sources."""

    id: str
    area: str
    kind: str
    status: str
    points: tuple[Point, ...]
    data: tuple[Datum, ...]
    sources: Mapping[str, Source]
    tags: tuple[str, ...] = ()
    title: str = ""
    cnickel: str = ""
    loops: int | None = None
    propagators: int | None = None
    polynomial_text: str = ""
    variables: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    reference_data: str = ""
    letter_symbols: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    def point(self, name: str) -> Point:
        for point in self.points:
            if point.name == name:
                return point
        raise BankError(f"{self.id} has no point {name!r}")

    def datum(self, datum_id: str) -> Datum:
        for datum in self.data:
            if datum.id == datum_id:
                return datum
        raise BankError(f"{self.id} has no datum {datum_id!r}")

    def data_of(self, type: str | None = None, point: str | None = None) -> tuple[Datum, ...]:
        """The data of a type and at a point; either filter may be left out."""
        return tuple(
            d
            for d in self.data
            if (type is None or d.type == type) and (point is None or d.point == point)
        )

    # -------- Building the family --------

    def integral(self, point: str | None = None) -> FeynmanIntegral:
        """The :class:`FeynmanIntegral` of a ``graph`` family, with its generic symbols.

        A point that names a ``kinematics`` class gets the integral with that class imposed,
        as :meth:`FeynmanIntegral.with_kinematics` does.
        """
        from ..integral import FeynmanIntegral

        if self.kind != "graph":
            raise BankError(f"{self.id} is a {self.kind} family, not a graph")
        kinematics = self.point(point).kinematics if point is not None else ""
        return FeynmanIntegral.from_cnickel(self.cnickel, kinematics=cast(Any, kinematics or None))

    def polynomial(self) -> tuple[sp.Expr, list[sp.Symbol], list[sp.Symbol]]:
        """The polynomial, the variables and the parameters of a ``polynomial`` family."""
        if self.kind != "polynomial":
            raise BankError(f"{self.id} is a {self.kind} family, not a polynomial")
        variables = [sp.Symbol(v) for v in self.variables]
        parameters = [sp.Symbol(p) for p in self.parameters]
        local = {str(s): s for s in variables + parameters}
        return sp.sympify(self.polynomial_text, locals=local), variables, parameters

    def polynomial_at(self, name: str) -> tuple[sp.Expr, list[sp.Symbol]]:
        """The polynomial of a family at a point, as an exact expression, and its variables.

        For a ``graph`` family it is G = U + F of Lee and Pomeransky with the energy scale
        set to 1, the point's values and its on-shell substitutions made; a key such as
        ``m_1^2`` gives the square of the mass m_1. For a ``polynomial`` family it is the
        polynomial with the parameters substituted. Every parameter must be given a value.
        """
        point = self.point(name)
        if self.kind == "polynomial":
            expr, variables, _ = self.polynomial()
            known = {str(s): s for s in expr.free_symbols}
            substitution = {_resolve(known, k): _exact(v) for k, v in point.values.items()}
        elif self.kind == "graph":
            integral = self.integral()
            variables = list(integral.symanzik.lp_parameters)
            expr = integral.symanzik.g.subs(integral.graph.energy_scale, 1)
            table = _symbols(integral)
            substitution = {}
            for key, value in {**point.values, **point.on_shell}.items():
                if key not in table and key.endswith("^2") and key[:-2] in table:
                    substitution[table[key[:-2]]] = sp.sqrt(_exact(value))
                else:
                    substitution[_resolve(table, key)] = _exact(value)
        else:
            raise BankError(f"{self.id} is a reference family; it has no polynomial")
        result = sp.expand(expr.subs(substitution))
        left = sorted(str(s) for s in result.free_symbols if s not in variables)
        if left:
            raise BankError(f"point {name!r} of {self.id} leaves {', '.join(left)} unspecified")
        return result, variables


def _exact(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


# -------- Name resolution --------


def _symbols(integral: FeynmanIntegral) -> dict[str, sp.Symbol]:
    found: set[sp.Symbol] = set(integral.symanzik.g.free_symbols)
    for value in integral.momentum_products.values():
        found |= sp.sympify(value).free_symbols
    return {str(s): s for s in found}


def _resolve(table: Mapping[str, sp.Symbol], key: str) -> sp.Symbol:
    """The symbol a name of a point or an on-shell substitution stands for."""
    if key in table:
        return table[key]
    raise BankError(f"{key!r} names no kinematic symbol; the symbols are {sorted(table)}")


# -------- Reading files --------


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise BankError(f"{path.name} is not valid TOML: {error}") from error


@cache
def load_sources() -> Mapping[str, Source]:
    """The papers of ``sources.toml`` by key."""
    return parse_sources(_read(DATA_DIR / SOURCES_FILE))


def parse_sources(raw: Mapping[str, Any]) -> dict[str, Source]:
    """Check the table of papers and build it; raises :class:`BankError` listing every problem."""
    problems: list[str] = []
    result: dict[str, Source] = {}
    for key, entry in raw.items():
        if not isinstance(entry, Mapping):
            problems.append(f"source {key}: not a table")
            continue
        unknown = set(entry) - _SOURCE_KEYS
        problems += [f"source {key}: unknown key {name!r}" for name in sorted(unknown)]
        problems += [
            f"source {key}: {name} is missing or empty"
            for name in _SOURCE_REQUIRED
            if not isinstance(entry.get(name), str) or not entry[name]
        ]
        arxiv = entry.get("arxiv")
        if isinstance(arxiv, str) and not _ARXIV.match(arxiv):
            problems.append(f"source {key}: arxiv {arxiv!r} is not an arXiv identifier")
        doi = entry.get("doi", "")
        if doi and not (isinstance(doi, str) and doi.startswith("10.")):
            problems.append(f"source {key}: doi {doi!r} does not start with 10.")
        if not problems:
            result[key] = Source(key=key, **{k: entry.get(k, "") for k in _SOURCE_KEYS})
    if problems:
        raise BankError("; ".join(problems))
    return result


def _rational(text: object) -> Fraction | None:
    if isinstance(text, bool):
        return None
    if isinstance(text, int):
        return Fraction(text)
    if isinstance(text, str) and _RATIONAL.match(text):
        return Fraction(text)
    return None


def _fraction(text: object) -> Fraction:
    value = _rational(text)
    if value is None:  # checked before; this keeps the type honest
        raise BankError(f"{text!r} is not an exact rational")
    return value


def _check_family_header(raw: Mapping[str, Any], problems: list[str]) -> None:
    head = raw.get("family")
    if not isinstance(head, Mapping):
        problems.append("there is no [family] table")
        return
    problems += [f"family: unknown key {k!r}" for k in sorted(set(head) - _FAMILY_KEYS)]
    fid = head.get("id")
    if not isinstance(fid, str) or not _FAMILY_ID.match(fid):
        problems.append(f"family: id {fid!r} does not look like 'FT1-sunrise2-generic'")
    for name, allowed in (("area", AREAS), ("kind", KINDS), ("status", STATUSES)):
        if head.get(name) not in allowed:
            problems.append(f"family: {name} {head.get(name)!r} is not one of {allowed}")
    kind = head.get("kind")
    if kind == "graph":
        if not isinstance(head.get("cnickel"), str) or not head.get("cnickel"):
            problems.append("family: a graph family needs cnickel")
        for name in ("loops", "propagators"):
            if not isinstance(head.get(name), int) or isinstance(head.get(name), bool):
                problems.append(f"family: a graph family needs the integer {name}")
    elif kind == "polynomial":
        if not isinstance(head.get("polynomial"), str) or not head.get("polynomial"):
            problems.append("family: a polynomial family needs polynomial")
        for name in ("variables", "parameters"):
            value = head.get(name)
            if not (isinstance(value, list) and value and all(isinstance(v, str) for v in value)):
                problems.append(f"family: a polynomial family needs the list {name}")
    elif kind == "reference" and not isinstance(head.get("reference_data"), str):
        problems.append("family: a reference family names its data in reference_data")
    symbols = head.get("letter_symbols", {})
    if not (
        isinstance(symbols, Mapping)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in symbols.items())
    ):
        problems.append("family: letter_symbols is a table of names")


def _check_points(
    raw: Mapping[str, Any], head: Mapping[str, Any], problems: list[str]
) -> list[str]:
    names: list[str] = []
    for index, point in enumerate(raw.get("point", [])):
        where = f"point {index}"
        if not isinstance(point, Mapping):
            problems.append(f"{where}: not a table")
            continue
        problems += [f"{where}: unknown key {k!r}" for k in sorted(set(point) - _POINT_KEYS)]
        name = point.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"{where}: name is missing")
            continue
        where = f"point {name}"
        if name in names:
            problems.append(f"{where}: the name is used twice")
        names.append(name)
        if name == "family":
            problems.append(f"{where}: 'family' is reserved for statements on the whole space")
        if point.get("kinematics", "") not in ("", *KINEMATIC_CLASSES):
            problems.append(f"{where}: kinematics {point.get('kinematics')!r} is not a class")
        for table in ("values", "on_shell"):
            entries = point.get(table, {})
            if not isinstance(entries, Mapping):
                problems.append(f"{where}: {table} is not a table")
                continue
            for key, value in entries.items():
                if _rational(value) is None:
                    problems.append(f"{where}: {table}.{key} = {value!r} is not an exact rational")
        if head.get("kind") == "polynomial":
            allowed = set(head.get("parameters", []))
            stray = sorted(set(point.get("values", {})) - allowed)
            problems += [f"{where}: values.{key} is not a parameter" for key in stray]
    return names


def _check_value(datum: Mapping[str, Any], where: str, problems: list[str]) -> None:
    value = datum.get("value")
    kind = datum.get("type")
    if kind == "function_class":
        if value not in FUNCTION_CLASSES:
            problems.append(f"{where}: function class {value!r} is not one of {FUNCTION_CLASSES}")
        return

    def integer(x: object) -> bool:
        return isinstance(x, int) and not isinstance(x, bool)

    if integer(value):
        return
    if isinstance(value, list):
        if len(value) < 2 or not all(integer(x) for x in value):
            problems.append(f"{where}: a list of values holds at least two integers")
        elif datum.get("status") != "disputed":
            problems.append(f"{where}: several values need the status 'disputed'")
        return
    if isinstance(value, Mapping):
        if set(value) != {"lo", "hi"} or not all(integer(x) for x in value.values()):
            problems.append(f"{where}: a bound is {{lo = .., hi = ..}} with integers")
        elif value["lo"] > value["hi"]:
            problems.append(f"{where}: the bound has lo above hi")
        return
    problems.append(f"{where}: value {value!r} is not an integer, a list or a bound")


def _check_datum(
    datum: object,
    index: int,
    point_names: list[str],
    sources: Mapping[str, Source],
    seen: set[str],
    problems: list[str],
) -> None:
    where = f"datum {index}"
    if not isinstance(datum, Mapping):
        problems.append(f"{where}: not a table")
        return
    problems += [f"{where}: unknown key {k!r}" for k in sorted(set(datum) - _DATUM_KEYS)]
    problems += [f"{where}: {k} is missing" for k in _DATUM_REQUIRED if k not in datum]
    did = datum.get("id")
    if isinstance(did, str):
        where = f"datum {did}"
        if did in seen:
            problems.append(f"{where}: the id is used twice")
        seen.add(did)
    if datum.get("type") not in DATUM_TYPES:
        problems.append(f"{where}: type {datum.get('type')!r} is not one of {DATUM_TYPES}")
    if "value" in datum:
        _check_value(datum, where, problems)
    if datum.get("point") not in [*point_names, "family"]:
        problems.append(f"{where}: point {datum.get('point')!r} is not defined")
    for name, allowed in (
        ("scope", SCOPES),
        ("symmetries", SYMMETRIES),
        ("status", DATUM_STATUSES),
    ):
        if name in datum and datum[name] not in allowed:
            problems.append(f"{where}: {name} {datum[name]!r} is not one of {allowed}")
    excludes = datum.get("excludes", [])
    if not isinstance(excludes, list) or any(e not in EXCLUDES for e in excludes):
        problems.append(f"{where}: excludes must be a list drawn from {EXCLUDES}")
    # A datum with no source is rejected; so is one whose source is not in sources.toml.
    for name in ("source", *(["also"] if "also" in datum else [])):
        keys = datum.get(name) if name == "also" else [datum.get(name)]
        if not isinstance(keys, list) or any(k not in sources for k in keys):
            problems.append(f"{where}: {name} {datum.get(name)!r} is not in sources.toml")
    location = datum.get("location", "")
    if not (isinstance(location, str) and _LABEL.search(location) and _PAGE.search(location)):
        problems.append(
            f"{where}: location {location!r} needs a section, equation or table and a page"
        )
    verified = datum.get("verified")
    if verified not in VERIFIED:
        problems.append(f"{where}: verified {verified!r} is not one of {VERIFIED}")
    status = datum.get("status", "published")
    if verified == "cite-only" and status == "published":
        problems.append(f"{where}: a cite-only number cannot be 'published'; use 'published-via'")
    if status == "published-via" and verified != "cite-only":
        problems.append(f"{where}: 'published-via' is for cite-only numbers")
    checked = datum.get("checked")
    try:
        date.fromisoformat(checked)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        problems.append(f"{where}: checked {checked!r} is not an ISO date")
    letters = datum.get("letters", [])
    if not isinstance(letters, list) or not all(isinstance(x, str) for x in letters):
        problems.append(f"{where}: letters must be a list of strings")


def _check_second_sources(raw: Mapping[str, Any], problems: list[str]) -> None:
    """A 'published-via' number needs a sibling datum read from text or image, or a gap note."""
    data = [d for d in raw.get("datum", []) if isinstance(d, Mapping)]
    for datum in data:
        if datum.get("status") != "published-via":
            continue
        gap = datum.get("gap")
        if isinstance(gap, str) and gap.strip():
            continue
        if "gap" in datum:
            problems.append(f"datum {datum.get('id')}: gap must say why there is no second source")
            continue
        if not any(
            other is not datum
            and other.get("type") == datum.get("type")
            and other.get("point") == datum.get("point")
            and other.get("value") == datum.get("value")
            and other.get("verified") in ("text", "image")
            for other in data
        ):
            problems.append(
                f"datum {datum.get('id')}: a 'published-via' number needs a second datum of the "
                f"same type, point and value read from text or image, or a 'gap' note"
            )


_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _check_letters(raw: Mapping[str, Any], head: Mapping[str, Any], problems: list[str]) -> None:
    """Letters use only the names the family defines, and those name kinematic symbols."""
    symbols = head.get("letter_symbols", {})
    if not isinstance(symbols, Mapping):
        return
    kinematic: set[str] = set()
    for point in raw.get("point", []):
        if isinstance(point, Mapping) and isinstance(point.get("values"), Mapping):
            kinematic |= set(point["values"])
    if head.get("kind") == "graph":
        stray = sorted(str(v) for v in symbols.values() if v not in kinematic)
        problems += [
            f"family: letter_symbols names {v!r}, which no point gives a value" for v in stray
        ]
        known = set(symbols)
    else:
        known = (
            set(head.get("parameters", [])) if isinstance(head.get("parameters"), list) else set()
        )
    for datum in raw.get("datum", []):
        letters = datum.get("letters", []) if isinstance(datum, Mapping) else []
        if not isinstance(letters, list):
            continue
        for letter in letters:
            if not isinstance(letter, str):
                continue
            used = set(_NAME.findall(letter))
            if used - known:
                problems.append(
                    f"datum {datum.get('id')}: letter {letter!r} uses {sorted(used - known)}, "
                    f"which the family does not define"
                )


def parse_family(raw: Mapping[str, Any], sources: Mapping[str, Source]) -> BankFamily:
    """Check one family's table and build it; raises :class:`BankError` listing every problem."""
    problems: list[str] = []
    unknown = set(raw) - {"family", "point", "datum"}
    problems += [f"unknown table {k!r}" for k in sorted(unknown)]
    _check_family_header(raw, problems)
    head = raw.get("family", {}) if isinstance(raw.get("family"), Mapping) else {}
    names = _check_points(raw, head, problems)
    seen: set[str] = set()
    for index, datum in enumerate(raw.get("datum", [])):
        _check_datum(datum, index, names, sources, seen, problems)
    if not raw.get("datum"):
        problems.append("a family without published data has no use")
    _check_letters(raw, head, problems)
    _check_second_sources(raw, problems)
    if problems:
        raise BankError(f"{head.get('id', '?')}: " + "; ".join(problems))
    points = tuple(
        Point(
            name=p["name"],
            values={k: _fraction(v) for k, v in p.get("values", {}).items()},
            on_shell={k: _fraction(v) for k, v in p.get("on_shell", {}).items()},
            kinematics=p.get("kinematics", ""),
            note=p.get("note", ""),
        )
        for p in raw.get("point", [])
    )
    data = tuple(_build_datum(d) for d in raw["datum"])
    return BankFamily(
        id=head["id"],
        area=head["area"],
        kind=head["kind"],
        status=head["status"],
        points=points,
        data=data,
        sources=dict(sources),
        tags=tuple(head.get("tags", ())),
        title=head.get("title", ""),
        cnickel=head.get("cnickel", ""),
        loops=head.get("loops"),
        propagators=head.get("propagators"),
        polynomial_text=head.get("polynomial", ""),
        variables=tuple(head.get("variables", ())),
        parameters=tuple(head.get("parameters", ())),
        reference_data=head.get("reference_data", ""),
        letter_symbols=dict(head.get("letter_symbols", {})),
        note=head.get("note", ""),
    )


def _build_datum(raw: Mapping[str, Any]) -> Datum:
    value = raw["value"]
    if isinstance(value, list):
        value = tuple(value)
    kwargs = {
        k: raw[k] for k in _DATUM_KEYS if k in raw and k not in ("excludes", "also", "letters")
    }
    kwargs["value"] = value
    return Datum(
        excludes=tuple(raw.get("excludes", ())),
        also=tuple(raw.get("also", ())),
        letters=tuple(raw.get("letters", ())),
        **kwargs,
    )


# -------- Public access --------


def _ids() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.toml") if p.name != SOURCES_FILE)


@cache
def load(family_id: str) -> BankFamily:
    """The family of that id, checked against ``sources.toml``."""
    path = DATA_DIR / f"{family_id}.toml"
    if family_id not in _ids():
        raise BankError(f"no family {family_id!r}; the bank has {', '.join(_ids())}")
    family = parse_family(_read(path), load_sources())
    if family.id != family_id:
        raise BankError(f"{path.name} holds the family {family.id!r}")
    return family


def list_families(area: str | None = None, status: str | None = None) -> list[str]:
    """The ids of the families, sorted, optionally of one area and one status (G or H)."""
    if area is not None and area not in AREAS:
        raise BankError(f"area {area!r} is not one of {AREAS}")
    if status is not None and status not in STATUSES:
        raise BankError(f"status {status!r} is not one of {STATUSES}")
    result = []
    for family_id in _ids():
        family = load(family_id)
        if (area is None or family.area == area) and (status is None or family.status == status):
            result.append(family_id)
    return result
