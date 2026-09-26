"""
Newton polytopes against the principal Landau determinant database.

The database of Fevola, Mizera and Telen, Principal Landau determinants,
Comput. Phys. Commun. 303 (2024) 109278, arXiv:2311.16219, is published on
MathRepo (https://mathrepo.mis.mpg.de/PLD/, PLD_database.zip, CC BY 4.0).
Each entry gives U, F and the f-vector f_0, ..., f_{d-1} of the Newton
polytope of G = U + F. Three entries are committed under tests/data/pld/ and
checked always; set FEYNKIT_PLD_DATA to the unpacked database directory to
check all 114.
"""

from __future__ import annotations

import itertools
import os
import re
from pathlib import Path
from typing import Any

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from feynkit.polytope import polytope_data
from tests.test_polytope import diagram_points

FIXTURES = Path(__file__).resolve().parent / "data" / "pld"
SLOW_ENTRIES = {"dpent_zero_zero", "npl-dpent_zero_zero", "npl-dpent2_zero_zero"}
_INDEXED = re.compile(r"(\w+)\[(\d+)\]")
_NAME = re.compile(r"\w+(?:\[\d+\])?")


def _python(text: str) -> str:
    """Julia syntax to SymPy's: x[1] -> x_1, ^ -> **, // -> /."""
    return _INDEXED.sub(r"\1_\2", text).replace("^", "**").replace("//", "/")


def read_entry(path: Path) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    """The exponent vectors of U + F and the published f-vector of one entry."""
    head = path.read_text(encoding="utf-8").split("# Component 1")[0]
    fields = dict(re.findall(r"^(\w+) = (.+?)\s*$", head, flags=re.MULTILINE))
    names = [_python(name) for name in _NAME.findall(fields["variables"])]
    symbols = {name: sp.Symbol(name) for name in names}
    for parameter in _NAME.findall(fields["parameters"]):
        symbols[_python(parameter)] = sp.Symbol(_python(parameter))
    g = parse_expr(_python(fields["U"]), local_dict=symbols) + parse_expr(
        _python(fields["F"]), local_dict=symbols
    )
    monomials = sp.Poly(sp.expand(g), *(symbols[name] for name in names)).monoms()
    f_vector = tuple(int(x) for x in re.findall(r"\d+", fields["f_vector"]))
    return [tuple(int(e) for e in m) for m in monomials], f_vector


def same_up_to_permutation(a: list[tuple[int, ...]], b: list[tuple[int, ...]]) -> bool:
    """Whether the point sets agree after permuting the coordinates of a.

    Coordinates are grouped by the multiset of their values, and only
    permutations within a group are tried.
    """
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return False
    n = len(a[0])
    target = sorted(b)
    classes: dict[tuple[int, ...], tuple[list[int], list[int]]] = {}
    for i in range(n):
        classes.setdefault(tuple(sorted(p[i] for p in a)), ([], []))[0].append(i)
    for j in range(n):
        classes.setdefault(tuple(sorted(p[j] for p in b)), ([], []))[1].append(j)
    groups = list(classes.values())
    if any(len(src) != len(dst) for src, dst in groups):
        return False
    for choice in itertools.product(*(itertools.permutations(dst) for _, dst in groups)):
        perm = [0] * n
        for (src, _), dst in zip(groups, choice, strict=True):
            for i, j in zip(src, dst, strict=True):
                perm[j] = i
        if sorted(tuple(p[perm[j]] for j in range(n)) for p in a) == target:
            return True
    return False


def _database_dir() -> Path | None:
    root = os.environ.get("FEYNKIT_PLD_DATA")
    return Path(root).expanduser() if root else None


def _skip_reason() -> str | None:
    """Why the full database cannot be checked, or None when it can."""
    base = _database_dir()
    if base is None:
        return "FEYNKIT_PLD_DATA is not set"
    if not base.is_dir():
        return f"FEYNKIT_PLD_DATA={base} is not a directory"
    return None


def _database_files() -> list[Path]:
    """The entry files under FEYNKIT_PLD_DATA, or its database subdirectory.

    Hidden files are left out, such as the AppleDouble files ._<name>.txt
    that macOS writes beside files on some file systems.
    """
    base = _database_dir()
    if base is None:
        return []
    for directory in (base, base / "database"):
        files = sorted(p for p in directory.glob("*.txt") if not p.name.startswith("."))
        if files:
            return files
    return []


def _entry(name: str) -> Path:
    """The committed copy of an entry, else the one under FEYNKIT_PLD_DATA."""
    fixture = FIXTURES / f"{name}.txt"
    if fixture.exists():
        return fixture
    for path in _database_files():
        if path.stem == name:
            return path
    pytest.skip(f"{name} is not committed; set FEYNKIT_PLD_DATA to check it")


TABLE = [
    pytest.param("12e|2e|e|:zzz", None, (6, 12, 8, 1), 4, id="massless-triangle"),
    pytest.param(
        "12e|3e|3e|e|:zzzz", "A4_zero_generic", (10, 30, 30, 10, 1), 11, id="massless-box"
    ),
    pytest.param(
        "12e|3e|3e|e|:nnnn", "A4_generic_generic", (8, 16, 14, 6, 1), 15, id="massive-box"
    ),
    pytest.param("111e|e|:zzz", None, (4, 6, 4, 1), 1, id="massless-sunrise"),
    pytest.param("111e|e|:nnn", None, (9, 15, 8, 1), 10, id="massive-sunrise"),
    pytest.param("12e|23|3|e|:zzzzz", None, (16, 54, 78, 54, 16, 1), 42, id="massless-kite"),
    pytest.param(
        "12e|23|3|e|:nnnnn",
        "kite_generic_generic",
        (24, 66, 73, 39, 10, 1),
        136,
        id="massive-kite",
    ),
    pytest.param("12ee|22e|e|:zzzz", "par_zero_generic", (9, 24, 24, 9, 1), 8, id="par-massless"),
    pytest.param(
        "12ee|22e|e|:nnnn", "par_generic_generic", (15, 33, 27, 9, 1), 35, id="par-massive"
    ),
    pytest.param(
        "15e|24|3e|4e|5|e|:zzzzzzz",
        "dbox_zero_generic",
        (46, 282, 636, 706, 421, 133, 20, 1),
        903,
        id="planar-double-box",
    ),
    pytest.param(
        "145|26|3e|4e|e|6e|e|:zzzzzzzz",
        "pentb_zero_generic",
        (68, 504, 1327, 1778, 1373, 630, 166, 22, 1),
        3148,
        id="pentagon-box",
        marks=pytest.mark.slow,
    ),
]


@pytest.mark.parametrize(("cnickel", "entry", "f_vector", "volume"), TABLE)
def test_table(cnickel: str, entry: str | None, f_vector: tuple[int, ...], volume: int) -> None:
    data = polytope_data(diagram_points(cnickel))
    assert data.f_vector == f_vector
    assert data.normalized_volume == volume
    assert sum((-1) ** k * f for k, f in enumerate(data.f_vector)) == 1


@pytest.mark.parametrize(("cnickel", "entry", "f_vector", "volume"), TABLE)
def test_table_row_is_the_database_entry(
    cnickel: str, entry: str | None, f_vector: tuple[int, ...], volume: int
) -> None:
    if entry is None:
        pytest.skip("no database entry matches this diagram")
    exponents, published = read_entry(_entry(entry))
    assert same_up_to_permutation(diagram_points(cnickel), exponents)
    assert published + (1,) == f_vector


@pytest.mark.parametrize(
    ("name", "f_vector"),
    [
        ("A4_zero_generic", (10, 30, 30, 10)),
        ("par_zero_generic", (9, 24, 24, 9)),
        ("kite_generic_generic", (24, 66, 73, 39, 10)),
    ],
)
def test_committed_entry(name: str, f_vector: tuple[int, ...]) -> None:
    exponents, published = read_entry(FIXTURES / f"{name}.txt")
    assert published == f_vector
    assert polytope_data(exponents).f_vector == f_vector + (1,)


def test_same_up_to_permutation() -> None:
    square = [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert same_up_to_permutation([(0, 2), (1, 0)], [(2, 0), (0, 1)])
    assert same_up_to_permutation(square, square)
    assert not same_up_to_permutation([(0, 2), (1, 0)], [(2, 0), (1, 0)])
    assert not same_up_to_permutation([(0, 0), (1, 1)], [(0, 1), (1, 0)])
    assert not same_up_to_permutation(square, square[:3])
    assert not same_up_to_permutation([(0, 0)], [(0, 0, 0)])


def _entry_params() -> list[Any]:
    reason = _skip_reason()
    if reason is not None:
        return [pytest.param(None, id="no-database", marks=pytest.mark.skip(reason=reason))]
    params = [
        pytest.param(
            path, id=path.stem, marks=[pytest.mark.slow] if path.stem in SLOW_ENTRIES else []
        )
        for path in _database_files()
    ]
    return params or [pytest.param(None, id="no-entries")]


@pytest.mark.parametrize("path", _entry_params())
def test_database_entry(path: Path | None) -> None:
    if path is None:
        pytest.fail(
            f"FEYNKIT_PLD_DATA={os.environ.get('FEYNKIT_PLD_DATA')} holds no entry files; "
            "point it at the unpacked database directory"
        )
    exponents, f_vector = read_entry(path)
    assert polytope_data(exponents).f_vector == f_vector + (1,)


def test_database_has_every_entry() -> None:
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    assert len(_database_files()) == 114
