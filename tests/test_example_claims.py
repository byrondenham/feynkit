"""The example scripts make none of the claims that earlier reviews retracted."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

EXAMPLE_SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))

# Toric operators may be called an analogue of IBP relations, and nothing more.
_IBP_ALLOWED = re.compile(
    r"analogue of (?:integration-by-parts \(IBP\)|IBP) relations"
    r"|not (?:an IBP relation itself|IBP relations themselves)"
)

RETRACTED = (
    "single master",
    "unique master",
    "is a master integral",
    "likely a master integral",
    '"master integral"',
    "rank = # master integrals",
    "Finite-index (|det| > 1)",
    "I(z) = I(sigma",
    "I_A(z') = I_A(z)",
    "dim ker A = number of toric generators",
)


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_makes_no_retracted_claim(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    ibp = [line.strip() for line in _IBP_ALLOWED.sub("", text).splitlines() if "IBP" in line]
    assert ibp == [], f"{script.name} calls something an IBP relation: {ibp}"
    found = [phrase for phrase in RETRACTED if phrase in text]
    assert found == [], f"{script.name} still says {found}"
