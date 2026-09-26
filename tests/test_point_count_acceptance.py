"""
Candidate master counts against the literature and the Landau determinant database.

C = (-1)^N chi(X) counts master integrals with subsectors included, symmetries
unused and D symbolic (Bitoun, Bogner, Klausen and Panzer, Lett. Math. Phys.
109 (2019) 497, arXiv:1712.09215, Corollary 37; BBKP below). FMT below is
Fevola, Mizera and Telen, Principal Landau determinants, Comput. Phys. Commun.
303 (2024) 109278, arXiv:2311.16219. Their Table 2 gives C = 2^n - 1 for the
one-loop graph with n massive propagators and 2^n - 1 - n with n massless ones
and off-shell legs; Table 1 gives C for the box A4 on subspaces of its
kinematics. Their database (MathRepo, CC BY 4.0) lists chi_generic, |chi(X)|
for generic kinematics, for each entry. It writes F with the opposite sign to
feynkit, which leaves the counts unchanged: its G = U + F has
G(-u) = (-1)^L (U - F).
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from feynkit import FeynmanIntegral
from feynkit.core.exceptions import ValidationError
from feynkit.landau import LandauAnalysis, _singular_binary, landau_analysis
from feynkit.point_count import TorusCount, count_torus_points, critical_point_count
from tests.test_pld import _NAME, FIXTURES, _database_files, _entry, _python, _skip_reason

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

S = sp.Symbol("s", real=True)
P = [sp.Symbol(f"p{i}^2", real=True) for i in range(1, 5)]
S12, S23 = sp.Symbol("s12", real=True), sp.Symbol("s23", real=True)
M = [sp.Symbol(f"m_{e}", nonnegative=True, real=True) for e in range(1, 4)]
ON_SHELL = {"12e|3e|3e|e|:zzzz": dict.fromkeys(P, 0)}

# (CNickel string, C), each with its source; the on-shell box has p_i^2 = 0.
LITERATURE = [
    pytest.param("11e|e|:zz", 1, id="massless-bubble"),  # FMT Table 2, n = 2
    pytest.param("11e|e|:nz", 2, id="one-mass-bubble"),  # mathematics reference 4.7, m_2 = 0
    pytest.param("11e|e|:nn", 3, id="two-mass-bubble"),  # FMT Table 2, n = 2
    pytest.param("12e|2e|e|:zzz", 4, id="massless-triangle"),  # FMT Table 2, n = 3
    pytest.param("12e|3e|3e|e|:zzzz", 3, id="on-shell-box"),  # FMT Table 1, A4, every mass 0
    pytest.param("12e|23|3|e|:zzzzz", 3, id="massless-kite"),  # BBKP Example 53, WS'_3
]


@functools.cache
def _analysis(cnickel: str) -> LandauAnalysis | None:
    """The Landau analysis both seeds share; the on-shell box computes its own."""
    if cnickel in ON_SHELL:
        return None
    return landau_analysis(FeynmanIntegral.from_cnickel(cnickel))


@functools.cache
def _count(cnickel: str, seed: int) -> TorusCount:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    return fi.torus_count(seed=seed, on_shell=ON_SHELL.get(cnickel), landau=_analysis(cnickel))


def _g(fi: FeynmanIntegral, on_shell: dict[sp.Symbol, int] | None = None) -> sp.Expr:
    """G at mu = 1, with the on-shell substitutions made."""
    return sp.expand(fi.symanzik.g.subs(fi.graph.energy_scale, 1).subs(on_shell or {}))


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(("cnickel", "master"), LITERATURE)
def test_literature_master_counts(cnickel: str, master: int, seed: int) -> None:
    count = _count(cnickel, seed)
    assert count.candidate_polynomial is not None, count.reason
    assert len(count.candidate_polynomial) == len(count.variables)
    assert count.candidate_master_count == master
    assert not count.on_landau_surface


def test_off_shell_box() -> None:
    # FMT Table 1 gives C = 11 for A4 with massless propagators and off-shell legs. The five
    # square-test factors are squares at this point.
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
    point = {P[0]: -12, P[1]: -6, P[2]: 5, P[3]: -4, S12: 20, S23: -41}
    assert fi.torus_count(point=point).candidate_master_count == 11


def test_triangle_where_the_kallen_function_vanishes() -> None:
    # lambda(1, 4, 9) = 0: the Kallen function of the external masses vanishes, a
    # second-type singularity, so the point is on a Landau surface. C = 3 there, where
    # generic kinematics give 4.
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
    point = {P[0]: 1, P[1]: 4, P[2]: 9}
    with pytest.raises(ValidationError, match="vanishes at the point"):
        fi.torus_count(point=point)
    count = fi.torus_count(point=point, allow_singular=True)
    assert count.on_landau_surface
    assert count.candidate_master_count == 3


# Points where a square-test factor is not a square, with the number of critical points, the
# generic C: FMT Table 2 for the bubble and the triangle, BBKP Proposition 55 (2^(L+1) - 1
# for L loops) for the sunrise. The Kallen function is -3, not a square, at the bubble's and
# the triangle's points, where the bubble has p - 3 - (-3/p) points; s = 7 is not a square
# at the sunrise's.
NOT_SQUARE = [
    pytest.param("11e|e|:nn", {S: 1, M[0] ** 2: 1, M[1] ** 2: 1}, 3, id="two-mass-bubble"),
    pytest.param("12e|2e|e|:zzz", {P[0]: 1, P[1]: 1, P[2]: 1}, 4, id="massless-triangle"),
    pytest.param(
        "111e|e|:nnn", {S: 7, M[0] ** 2: 2, M[1] ** 2: 3, M[2] ** 2: 5}, 7, id="three-mass-sunrise"
    ),
]


@pytest.mark.parametrize(("cnickel", "point", "critical"), NOT_SQUARE)
def test_no_candidate_where_a_square_test_factor_is_not_a_square(
    cnickel: str, point: dict[sp.Expr, int], critical: int
) -> None:
    count = FeynmanIntegral.from_cnickel(cnickel).torus_count(point=point)
    assert count.candidate_polynomial is None
    assert count.candidate_master_count is None
    assert "non-integer coefficients" in str(count.reason)


@requires_singular
@pytest.mark.parametrize(("cnickel", "point", "critical"), NOT_SQUARE)
def test_critical_points_where_a_square_test_factor_is_not_a_square(
    cnickel: str, point: dict[sp.Expr, int], critical: int
) -> None:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    assert critical_point_count(_g(fi), fi.symanzik.lp_parameters, point) == critical


@requires_singular
@pytest.mark.parametrize(("cnickel", "master"), LITERATURE)
def test_critical_points_match_the_candidates(cnickel: str, master: int) -> None:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    count = _count(cnickel, 0)
    g = _g(fi, ON_SHELL.get(cnickel))
    assert critical_point_count(g, fi.symanzik.lp_parameters, dict(count.point)) == master


# --- the principal Landau determinant database --------------------------------


def read_polynomial(path: Path) -> tuple[sp.Expr, list[sp.Symbol], dict[str, sp.Symbol], int]:
    """G = U - F in feynkit's sign convention, its variables, its parameters by name, chi_generic."""
    head = path.read_text(encoding="utf-8").split("# Component 1")[0]
    fields = dict(re.findall(r"^(\w+) = (.+?)\s*$", head, flags=re.MULTILINE))
    names = [_python(name) for name in _NAME.findall(fields["variables"])]
    symbols = {name: sp.Symbol(name) for name in names}
    parameters = {_python(p): sp.Symbol(_python(p)) for p in _NAME.findall(fields["parameters"])}
    local = {**symbols, **parameters}
    u = parse_expr(_python(fields["U"]), local_dict=local)
    f = parse_expr(_python(fields["F"]), local_dict=local)
    # The key is the Greek letter chi followed by _generic.
    chi = int(next(value for key, value in fields.items() if key.endswith("_generic")))
    return sp.expand(u - f), [symbols[name] for name in names], parameters, chi


def test_committed_entry_off_shell_box() -> None:
    g, variables, parameters, chi = read_polynomial(FIXTURES / "A4_zero_generic.txt")
    assert chi == 11
    values = {"M_1": -12, "M_2": 5, "M_3": -4, "M_4": -6, "s": 4, "t": 20}
    point = {parameters[name]: value for name, value in values.items()}
    assert count_torus_points(g, variables, point=point).candidate_master_count == chi


def test_database_entry_on_shell_box() -> None:
    g, variables, parameters, chi = read_polynomial(_entry("A4_zero_zero"))
    assert chi == 3
    point = {parameters["s"]: 16, parameters["t"]: 9}
    assert count_torus_points(g, variables, point=point).candidate_master_count == chi


def _variable_count(path: Path) -> int:
    line = re.search(r"^variables = (.+)$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert line is not None, path
    return len(_NAME.findall(line.group(1)))


def _small_entries() -> list[Any]:
    """The database entries with at most five variables, 55 of the 114; about 2.5 minutes."""
    reason = _skip_reason()
    if reason is not None:
        return [pytest.param(None, id="no-database", marks=pytest.mark.skip(reason=reason))]
    return [
        pytest.param(path, id=path.stem, marks=pytest.mark.slow)
        for path in _database_files()
        if _variable_count(path) <= 5
    ]


# The small entries that give a candidate at seed 0.
CANDIDATES_AT_SEED_0 = frozenset(
    {
        "A4_zero_zero",
        "B4_zero_equal",
        "B4_zero_generic",
        "B4_zero_zero",
        "acn_zero_zero",
        "debox_zero_zero",
        "par_zero_zero",
        "tdetri_zero_zero",
    }
)


@pytest.mark.parametrize("path", _small_entries())
def test_database_candidates_match_chi_generic(path: Path | None) -> None:
    assert path is not None
    g, variables, _, chi = read_polynomial(path)
    count = count_torus_points(g, variables, seed=0)
    if path.stem not in CANDIDATES_AT_SEED_0:
        assert count.candidate_master_count in (None, chi)
        return
    assert count.candidate_master_count == chi
    # A candidate is checked at four further primes at least, the default.
    assert len(count.verification_primes) >= 4
