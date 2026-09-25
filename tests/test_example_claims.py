"""The example scripts make none of the claims that earlier reviews retracted."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

EXAMPLE_SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))
assert EXAMPLE_SCRIPTS, "no example scripts found"

# A string literal continued on the next line ("... end "\n    "start ...") reads as one.
_ADJACENT_LITERALS = re.compile(r"[\"']\s*\n\s*f?[\"']")

# Sentences end at . ; ! ? or where a string literal closes an argument list.
_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+|[\"']\s*[),]\s*")

# Toric operators may be called an analogue of IBP relations, and nothing more.
_IBP = re.compile(r"IBP|(?i:integration[- ]by[- ]parts)")
_IBP_ALLOWED = re.compile(
    r"analogue of (?:integration-by-parts \(IBP\)|IBP) relations"
    r"|not (?:an IBP relation itself|IBP relations themselves)"
    r"|not a reduction between different integrals as an IBP relation is"
)

# The volume only bounds the number of master integrals, and a trivial toric
# ideal says nothing about it; any sentence about master integrals must say so.
_MASTER = re.compile(r"\bmaster\b", re.IGNORECASE)
_MASTER_ALLOWED = ("upper bound", "bounds", "at most", "says nothing", "Euler characteristic")

RETRACTED = (
    "Finite-index (|det| > 1)",
    "I(z) = I(sigma",
    "I_A(z') = I_A(z)",
    "(L*D)/2",
    "beta_k = nu_k",
    "rows = L+1",
    "rows = loops+1",
    "r = loops+1",
    "sum over two cosets",
    "ratio of normalised volumes",
    "wait, the volumes",
    "Re(beta_0) > 0",
    "epsilon-expansion module",
    "resonance module",
    "maps nu_2 -> -D",
    "parity projection",
    "geometric origin of the",
    "S_n -> S_{n-1}",
    "is the largest in this table",
    "genuinely different GKZ system",
    "K_{3,3}",
    "divided by 2 mu^2",
    "F/U^{D/2}",
    "edge midpoints",
    "B_4?",
    "faster than exponential",
    "torsion structure at the prime 2",
)

RETRACTED_PATTERNS = (
    # beta = (-D/2, -nu_1, ..., -nu_n): the integrand is u^(nu - 1) G^(-D/2).
    re.compile(r"u\^beta \* G(?:\(u\))?\^\{?-beta_0\}?"),
    re.compile(r"beta(?:_0)?\s*=\s*[\[(]?\s*-D/2\s*\+\s*\d*\s*nu"),
    re.compile(r"beta_0\s*=\s*nu"),
    # The pre-0.3.0 convention beta_0 = sum nu - D/2.
    re.compile(r"beta(?:_0)?\s*=\s*[\[(]?\s*(?:Sigma\s*|sum\s*|\d+\s*)?nu\w*\s*-\s*D/2"),
    # dim ker A is the codimension of the toric ideal, not its number of generators.
    re.compile(r"dim ker A\s*=\s*(?:#|number of)\s*toric generators"),
    re.compile(r"toric generators\s*=\s*dim ker A"),
    re.compile(r"ranks? differs? by"),
    # A has E+1 rows and the Newton polytope lies in R^E, E the number of propagators.
    re.compile(r"N = L\s*\+\s*1"),
    re.compile(r"\b[RZ]\^\{?L\b"),
    re.compile(r"GL_\{?L\b"),
)


def _prose(script: Path) -> str:
    """The script's text with continued string literals joined and whitespace collapsed."""
    text = _ADJACENT_LITERALS.sub("", script.read_text(encoding="utf-8"))
    return re.sub(r"\s+", " ", text)


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_calls_nothing_an_ibp_relation(script: Path) -> None:
    rest = _IBP_ALLOWED.sub("", _prose(script))
    ibp = [rest[max(0, m.start() - 60) : m.end() + 60] for m in _IBP.finditer(rest)]
    assert ibp == [], f"{script.name} calls something an IBP relation: {ibp}"


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_only_bounds_the_master_integral_count(script: Path) -> None:
    unbounded = [
        sentence
        for sentence in _SENTENCE_END.split(_prose(script))
        if _MASTER.search(sentence) and not any(word in sentence for word in _MASTER_ALLOWED)
    ]
    assert unbounded == [], f"{script.name} counts master integrals: {unbounded}"


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_makes_no_retracted_claim(script: Path) -> None:
    prose = _prose(script)
    found = [phrase for phrase in RETRACTED if phrase in prose]
    found += [p.pattern for p in RETRACTED_PATTERNS if p.search(prose)]
    assert found == [], f"{script.name} still says {found}"
