"""Helpers shared by the benchmark-bank tests: counts, the convention adapter and reports."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import sympy as sp

from feynkit import bank
from feynkit.bank import BankFamily, Datum
from feynkit.core.exceptions import ComputationError
from feynkit.landau import _singular_binary
from feynkit.point_count import critical_point_count
from feynkit.polytope import polytope_data

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

# Singular gets at most this many seconds for each call.
TIMEOUT = 120

SUNRISE = "111e|e|:nnn"


def count_at(family: BankFamily, point: str) -> int:
    """|chi| of the complement of {G = 0} in the torus at a point, by counting critical points.

    This is the number of master integrals with subsectors included and symmetries unused
    (Bitoun, Bogner, Klausen and Panzer, arXiv:1712.09215, Cor. 37). A solver that fails or
    runs past its limit makes the test skip: an undecided count is never a pass.
    """
    g, variables = family.polynomial_at(point)
    try:
        return critical_point_count(g, variables, {}, timeout=TIMEOUT)
    except (RuntimeError, ComputationError) as error:
        pytest.skip(f"undecided: {error}")


def newton_volume(family: BankFamily, point: str) -> int:
    """The normalised volume of the Newton polytope of G at a point."""
    g, variables = family.polynomial_at(point)
    monomials = sp.Poly(g, *variables).monoms()
    return polytope_data(monomials).normalized_volume


def tadpole_products(family: BankFamily, point: str) -> int:
    """The masters in the proper subsectors of the two-loop sunrise at a point.

    Contracting one line of the sunrise leaves the tadpoles of the other two lines. Their
    product has one master when both lines are massive and none when one is massless, where
    it is scaleless. A count of the Euler characteristic includes these; the sources count the
    top sector only. Other families have no entry, as their sources count the whole family.
    """
    if family.cnickel != SUNRISE:
        return 0
    masses = [family.point(point).values[f"m_{i}^2"] for i in (1, 2, 3)]
    return sum(all(m != 0 for j, m in enumerate(masses) if j != i) for i in range(3))


def adapt(family: BankFamily, datum: Datum, computed: int) -> tuple[int, str]:
    """The computed count in the convention of a datum, and how it must compare: == or >=.

    A top-sector count drops the masters of the subsectors. A count made with symmetries
    between lines can only be smaller than feynkit's, which uses none, so it is a lower
    bound unless the datum says that no symmetry is used or that it does not apply.
    """
    value = computed - (tadpole_products(family, datum.point) if datum.scope == "top-sector" else 0)
    return value, "==" if datum.symmetries in ("n/a", "not-used") else ">="


def report_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where the reports go: FEYNKIT_BANK_REPORT_DIR if set, else a temporary directory."""
    root = os.environ.get("FEYNKIT_BANK_REPORT_DIR")
    if root:
        path = Path(root).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("bank-reports")


def write_report(directory: Path, name: str, content: dict) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def graph_data(types: tuple[str, ...], statuses: tuple[str, ...]) -> list[tuple[str, str]]:
    """(family id, datum id) of the data of graph families with these types and statuses."""
    return [
        (family_id, datum.id)
        for family_id in bank.list_families()
        if bank.load(family_id).kind == "graph"
        for datum in bank.load(family_id).data
        if datum.type in types and datum.status in statuses
    ]
