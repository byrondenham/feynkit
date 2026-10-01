"""
The pairs of Table 1 of Fevola, Mizera and Telen against the degenerate faces.

Fevola, Mizera and Telen, Principal Landau determinants, arXiv:2311.16219, Tab. 1, p. 13, gives
((-1)^(E-1) chi, vol) for the graph polynomial G = U + F of each diagram on a kinematic family E.
By Proposition 6 of docs/design/2026-10-01-face-degeneracy.md, and Fevola, Mizera and Telen,
Thm. 2.3, p. 9, |chi| is below the volume exactly when some face of the Newton polytope is
degenerate. So a row with |chi| < vol must have a degenerate face, and a row with |chi| = vol
none. Each row is decided at the generic point of its family, over the field of rational functions
in the kinematic symbols, and the volume is checked against the published one.

The polynomials are the entries of the database committed under tests/data/pld/, where the
families are the columns of the left table: K has the masses generic, E(M,0) the internal masses
zero, E(0,m) the external masses zero and E(0,0) both. A family is made from the entry of K by
setting the symbols m_k, M_k to zero; the volume then agrees with the published one. The right
table is made of the entries named custom.

Singular gets 10 s or 30 s for each call, as the rows say. When a face outlasts that, it stays
undecided; a row with |chi| < vol asks only for a degenerate face, so one that was decided is
enough, and a row with |chi| = vol asks that every face be decided.

Left out, as each took more than 120 s at these limits, with five rows running at once:
K of acn, env, npltrb, debox, tdebox, pltrb, dbox and pentb; E(M,0) of env, dbox and pentb;
E(0,m) of env, npltrb, tdebox, pltrb, dbox and pentb (the database holds pentb with massless
propagators only, so K and E(0,m) of pentb would have to be built); and, of the right table,
Hj-npl-dbox, Bhabha-npl-dbox, Hj-npl-pentb, dpent, npl-dpent and npl-dpent2. The slow rows run
with -m slow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

from feynkit.degeneracy import face_degeneracy_from_polynomial
from feynkit.landau import _singular_binary
from tests.test_pld import _NAME, FIXTURES, _python

pytestmark = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")

K, M_I_0, ZERO_M_E, ZERO_ZERO = (False, False), (True, False), (False, True), (True, True)


def read_polynomial(path: Path, zero: tuple[bool, bool]) -> tuple[sp.Expr, list[sp.Symbol]]:
    """G = U + F of an entry and its variables, with the internal masses m_k set to zero when
    zero[0] and the external masses M_k when zero[1]."""
    head = path.read_text(encoding="utf-8").split("# Component 1")[0]
    fields = dict(re.findall(r"^(\w+) = (.+?)\s*$", head, flags=re.MULTILINE))
    names = [_python(name) for name in _NAME.findall(fields["variables"])]
    local = {name: sp.Symbol(name) for name in names}
    local |= {_python(p): sp.Symbol(_python(p)) for p in _NAME.findall(fields["parameters"])}
    g = sum(parse_expr(_python(fields[key]), local_dict=local) for key in ("U", "F"))
    vanishing = {
        symbol: 0
        for name, symbol in local.items()
        if (zero[0] and name.startswith("m_")) or (zero[1] and name.startswith("M_"))
    }
    return sp.expand(g.subs(vanishing)), [local[name] for name in names]


def row(
    entry: str,
    family: tuple[bool, bool],
    chi: int,
    volume: int,
    seconds: int,
    *,
    slow: bool = False,
) -> pytest.ParameterSet:
    name = {K: "K", M_I_0: "E(M,0)", ZERO_M_E: "E(0,m)", ZERO_ZERO: "E(0,0)"}.get(family, "")
    marks = [pytest.mark.slow] if slow else []
    return pytest.param(
        entry, family, chi, volume, seconds, id=f"{entry.split('_')[0]}-{name}", marks=marks
    )


ROWS = [
    # Left table, left to right: K, E(M,0), E(0,m), E(0,0).
    row("A4_generic_generic", K, 15, 15, 30),
    row("A4_generic_generic", M_I_0, 11, 11, 30),
    row("A4_generic_generic", ZERO_M_E, 11, 15, 30),
    row("A4_generic_generic", ZERO_ZERO, 3, 3, 30),
    row("B4_generic_generic", M_I_0, 1, 1, 30),
    row("B4_generic_generic", ZERO_ZERO, 1, 1, 30),
    row("B4_generic_generic", K, 15, 35, 10, slow=True),
    row("B4_generic_generic", ZERO_M_E, 15, 35, 10, slow=True),
    row("par_generic_generic", M_I_0, 4, 8, 30),
    row("par_generic_generic", ZERO_M_E, 13, 35, 30),
    row("par_generic_generic", ZERO_ZERO, 1, 3, 30),
    row("par_generic_generic", K, 19, 35, 10, slow=True),
    row("acn_generic_generic", ZERO_ZERO, 3, 9, 30),
    row("acn_generic_generic", M_I_0, 20, 54, 10, slow=True),
    row("acn_generic_generic", ZERO_M_E, 36, 136, 10, slow=True),
    row("env_generic_generic", ZERO_ZERO, 10, 80, 30),
    row("npltrb_generic_generic", ZERO_ZERO, 5, 61, 30),
    row("npltrb_generic_generic", M_I_0, 28, 252, 10, slow=True),
    row("tdetri_generic_generic", M_I_0, 4, 18, 30),
    row("tdetri_generic_generic", ZERO_ZERO, 1, 5, 30),
    row("tdetri_generic_generic", K, 51, 201, 10, slow=True),
    row("tdetri_generic_generic", ZERO_M_E, 33, 201, 10, slow=True),
    row("debox_generic_generic", M_I_0, 11, 33, 30),
    row("debox_generic_generic", ZERO_ZERO, 3, 10, 30),
    row("debox_generic_generic", ZERO_M_E, 31, 96, 10, slow=True),
    row("tdebox_generic_generic", ZERO_ZERO, 3, 41, 30),
    row("tdebox_generic_generic", M_I_0, 11, 113, 10, slow=True),
    row("pltrb_generic_generic", M_I_0, 16, 201, 30),
    row("pltrb_generic_generic", ZERO_ZERO, 4, 80, 30),
    row("dbox_generic_generic", ZERO_ZERO, 12, 238, 30),
    row("pentb_zero_zero", ZERO_ZERO, 62, 1186, 10, slow=True),
    # Right table, with the custom kinematics of its Fig. 1.
    row("inner-dbox_custom", K, 43, 834, 30, slow=True),
    row("outer-dbox_custom", K, 64, 1302, 30, slow=True),
    row("kite_generic_generic", K, 30, 136, 30, slow=True),
    row("Bhabha-dbox_custom", K, 64, 774, 10, slow=True),
    row("Bhabha2-dbox_custom", K, 79, 910, 10, slow=True),
]


@pytest.mark.parametrize(("entry", "family", "chi", "volume", "seconds"), ROWS)
def test_row_is_degenerate_exactly_when_chi_is_below_the_volume(
    entry: str, family: tuple[bool, bool], chi: int, volume: int, seconds: int
) -> None:
    g, variables = read_polynomial(FIXTURES / f"{entry}.txt", family)
    analysis = face_degeneracy_from_polynomial(g, variables, check=False, timeout=seconds)
    assert analysis.mode == "generic"
    assert analysis.volume == volume
    if chi < volume:
        assert analysis.degenerate_faces
        assert analysis.non_degenerate is False
    else:
        assert analysis.undecided_faces == ()
        assert analysis.non_degenerate is True
