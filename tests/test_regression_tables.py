"""
Published tables of quantities that feynkit computes.

Each row is checked against the value its source prints, with the page; pages are the printed
page numbers of the arXiv versions. C = (-1)^N chi((C^*)^N minus {G = 0}) counts master integrals
with subsectors included and symmetries unused (Bitoun, Bogner, Klausen and Panzer,
arXiv:1712.09215, BBKP below). It is computed as the number of critical points of
sum_e nu_e log u_e - (D/2) log G at a random rational point, by
:func:`~feynkit.point_count.critical_point_count`, which works modulo two primes; the volume is
N! Vol of the Newton polytope relative to Z^N.

- Klausen, arXiv:1910.08651 (Kla20 below), Table 1, p. 34: N, r = N - n - 1, vol_0, |U|, |F| and
  C for 28 one-, two- and three-loop graphs, every leg off shell. C is checked where the table
  prints it, except for the three amputated cap rows. Where C < vol_0 some face of the Newton
  polytope must be degenerate, and where C = vol_0 none, by Fevola, Mizera and Telen,
  arXiv:2311.16219, Thm. 2.3, p. 9, and Gelfand, Kapranov and Zelevinsky; that is checked
  exactly, at the generic point of the family, and certifies C = vol_0 on those rows.
- Fevola, Mizera and Telen, arXiv:2311.16219 (FMT below), Table 3, p. 58: dim E and the f-vector
  of the Newton polytope of G on the kinematic subspace E of each of its twelve diagrams, read
  from the headers of their database committed under tests/data/pld/. The degrees of the
  components of the Euler discriminant are checked for inner-dbox, with -m slow, against those
  of the principal Landau determinant, which is all feynkit computes of it. Left out: outer-dbox,
  Hj-npl-dbox, Bhabha-dbox, Bhabha2-dbox, kite and par, where the table counts components that
  HyperInt alone found (FMT, Sec. 5.4, p. 57); and the others, which took more than 120 s.
- Klausen, arXiv:2109.07584 (Kla21 below): the triangle's 21 face discriminants, App. A, p. 36;
  its proper mixed faces, Ex. 4.3, p. 28; and the two parametrisations of the leading Landau
  variety of the dunce's cap, Eqs. 6.6 and 6.7, pp. 33-34.
- BBKP: C in Table 1, p. 33, Prop. 54, p. 33, Props. 55 and 56, pp. 34-35, Ex. 57, p. 36,
  Table 2, p. 37, and Ex. 59, p. 38, for the graphs that the figures there define unambiguously.
- Chestnov, Matsubara-Heo, Munch and Takayama, arXiv:2305.01585 (CMMT below): the GKZ rank, the
  volume, and the rank of the restriction to the Feynman integral, C, for the polynomials of
  Ex. 2.2, p. 7, and Secs. 5.5.1 to 5.5.4, pp. 47-50, and the rank of the three-mass sunrise,
  Eq. 5.78, p. 44.

Each heavy call here takes under two minutes.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.degeneracy import face_degeneracy_from_polynomial
from feynkit.landau import _singular_binary, landau_analysis, landau_analysis_from_polynomial
from feynkit.point_count import critical_point_count
from feynkit.polytope import polytope_data
from tests.test_degeneracy_table import read_polynomial
from tests.test_pld import FIXTURES

requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")


def _g(cnickel: str, on_shell: tuple[int, ...] = ()) -> tuple[sp.Expr, list[sp.Symbol]]:
    """G = U + F at mu = 1 and its variables, with p_i^2 = 0 for each leg i in on_shell."""
    fi = FeynmanIntegral.from_cnickel(cnickel)
    g = fi.symanzik.g.subs(fi.graph.energy_scale, 1)
    names = {x.name: x for x in g.free_symbols}
    g = g.subs({names[f"p{i}^2"]: 0 for i in on_shell})
    return sp.expand(g), list(fi.symanzik.lp_parameters)


def _volume(g: sp.Expr, variables: list[sp.Symbol]) -> int:
    """N! Vol of the Newton polytope of g, relative to Z^N."""
    data = polytope_data([m for m, _ in sp.Poly(g, *variables).terms()])
    return data.normalized_volume * data.sublattice_index


def _count(g: sp.Expr, variables: list[sp.Symbol], seed: int = 0) -> int:
    """C at a random point: masses from 1 to 30, every other symbol from -30 to 30 but not 0."""
    rng = random.Random(seed)
    point: dict[sp.Expr, int] = {}
    for x in sorted(g.free_symbols - set(variables), key=lambda s: s.name):
        if x.name.startswith("m_"):
            point[x] = rng.randint(1, 30)
        else:
            point[x] = rng.choice([v for v in range(-30, 31) if v != 0])
    return critical_point_count(sp.expand(g.subs(point)), variables, {})


# --- Klausen 2020, Table 1, p. 34 ---------------------------------------------

# Klausen writes the Nickel index with the legs first and colours every entry, n for a massive
# propagator or an off-shell leg; feynkit's CNickel string lists the legs last and colours only
# the propagators. (CNickel, N, r, vol_0, |U|, |F|, C), C None where the table leaves it blank.
KLA20 = [
    pytest.param("11e|e|:zz", 3, 0, 1, 2, 1, 1, id="bubble-e11|e:n00|n"),
    pytest.param("11e|e|:nz", 4, 1, 2, 2, 2, 2, id="bubble-e11|e:nn0|n"),
    pytest.param("11e|e|:nn", 5, 2, 3, 2, 3, 3, id="bubble-e11|e:nnn|n"),
    pytest.param("111e|e|:zzz", 4, 0, 1, 3, 1, 1, id="sunset-e111|e:n000|n"),
    pytest.param("111e|e|:nzz", 6, 2, 3, 3, 3, 2, id="sunset-e111|e:nn00|n"),
    pytest.param("111e|e|:nnz", 8, 4, 6, 3, 5, 4, id="sunset-e111|e:nnn0|n"),
    pytest.param("111e|e|:nnn", 10, 6, 10, 3, 7, 7, id="sunset-e111|e:nnnn|n"),
    # C of the three amputated cap rows is not checked.
    pytest.param("112e|2|e|:zzzz", 8, 3, 5, 5, 3, None, id="amputated-cap-e112|2|e:n000|0|n"),
    pytest.param("112e|2|e|:zznz", 10, 5, 8, 5, 5, None, id="amputated-cap-e112|2|e:n00n|0|n"),
    pytest.param("112e|2|e|:zznn", 13, 8, 14, 5, 8, None, id="amputated-cap-e112|2|e:n00n|n|n"),
    pytest.param("12e|2e|e|:zzz", 6, 2, 4, 3, 3, 4, id="vertex-e12|e2|e:n00|n0|n"),
    pytest.param("12e|2e|e|:nzz", 7, 3, 5, 3, 4, 5, id="vertex-e12|e2|e:nn0|n0|n"),
    pytest.param("12e|2e|e|:nnz", 8, 4, 6, 3, 5, 6, id="vertex-e12|e2|e:nnn|n0|n"),
    pytest.param("12e|2e|e|:nnn", 9, 5, 7, 3, 6, 7, id="vertex-e12|e2|e:nnn|nn|n"),
    pytest.param("12e|22e|e|:zzzz", 9, 4, 8, 5, 4, 4, id="dunces-cap-e12|e22|e:n00|n00|n"),
    pytest.param("12e|22e|e|:nzzz", 11, 6, 11, 5, 6, None, id="dunces-cap-e12|e22|e:nn0|n00|n"),
    pytest.param("12e|22e|e|:nnzz", 13, 8, 14, 5, 8, None, id="dunces-cap-e12|e22|e:nnn|n00|n"),
    pytest.param("13e|2e|3e|e|:zzzz", 10, 5, 11, 4, 6, 11, id="box-e13|e2|e3|e:n00|n0|n0|n"),
    pytest.param("13e|2e|3e|e|:nzzz", 11, 6, 12, 4, 7, 12, id="box-e13|e2|e3|e:nn0|n0|n0|n"),
    pytest.param("13e|2e|3e|e|:nnzz", 12, 7, 13, 4, 8, 13, id="box-e13|e2|e3|e:nnn|n0|n0|n"),
    pytest.param("13e|2e|3e|e|:nnnz", 13, 8, 14, 4, 9, 14, id="box-e13|e2|e3|e:nnn|nn|n0|n"),
    pytest.param("13e|2e|3e|e|:nnnn", 14, 9, 15, 4, 10, 15, id="box-e13|e2|e3|e:nnn|nn|nn|n"),
    pytest.param("13e|23|3e||:zzzzz", 16, 10, 42, 8, 8, 3, id="kite-e13|23|e3:n00|00|n0"),
    pytest.param("1111e|e|:zzzz", 5, 0, 1, 4, 1, 1, id="banana-e1111|e:n0000|n"),
    pytest.param("1111e|e|:nzzz", 8, 3, 4, 4, 4, None, id="banana-e1111|e:nn000|n"),
    pytest.param("1111e|e|:nnzz", 11, 6, 10, 4, 7, None, id="banana-e1111|e:nnn00|n"),
    pytest.param("1111e|e|:nnnz", 14, 9, 20, 4, 10, None, id="banana-e1111|e:nnnn0|n"),
    pytest.param("1111e|e|:nnnn", 17, 12, 35, 4, 13, 15, id="banana-e1111|e:nnnnn|n"),
]
KLA20_ARGS = ("cnickel", "n_g", "corank", "volume", "n_u", "n_f", "master")


@pytest.mark.parametrize(KLA20_ARGS, KLA20)
def test_klausen_table_1_polytope_columns(
    cnickel: str, n_g: int, corank: int, volume: int, n_u: int, n_f: int, master: int | None
) -> None:
    fi = FeynmanIntegral.from_cnickel(cnickel)
    sym = fi.symanzik
    schwinger = list(sym.schwinger_parameters)
    g, variables = _g(cnickel)
    assert len(sp.Poly(g, *variables).terms()) == n_g
    assert n_g - len(variables) - 1 == corank
    assert _volume(g, variables) == volume
    assert len(sp.Poly(sp.expand(sym.u), *schwinger).terms()) == n_u
    assert len(sp.Poly(sp.expand(sym.f), *schwinger).terms()) == n_f


@requires_singular
@pytest.mark.parametrize(KLA20_ARGS, [row for row in KLA20 if row.values[-1] is not None])
def test_klausen_table_1_master_counts(
    cnickel: str, n_g: int, corank: int, volume: int, n_u: int, n_f: int, master: int
) -> None:
    g, variables = _g(cnickel)
    assert _count(g, variables) == master


@requires_singular
@pytest.mark.parametrize(KLA20_ARGS, [row for row in KLA20 if row.values[-1] is not None])
def test_klausen_table_1_degenerate_faces(
    cnickel: str, n_g: int, corank: int, volume: int, n_u: int, n_f: int, master: int
) -> None:
    g, variables = _g(cnickel)
    analysis = face_degeneracy_from_polynomial(g, variables, check=False, timeout=10)
    assert analysis.volume == volume
    if master < volume:
        assert analysis.degenerate_faces
    else:
        assert analysis.undecided_faces == ()
        assert analysis.non_degenerate is True


# --- Fevola, Mizera and Telen, Table 3, p. 58 ---------------------------------

FMT_TABLE_3 = [
    ("inner-dbox_custom", 3, (37, 156, 294, 310, 195, 72, 14)),
    ("outer-dbox_custom", 3, (39, 153, 271, 272, 165, 60, 12)),
    ("Hj-npl-dbox_custom", 4, (40, 174, 333, 350, 215, 76, 14)),
    ("Bhabha-dbox_custom", 4, (32, 162, 347, 393, 252, 90, 16)),
    ("Bhabha2-dbox_custom", 4, (37, 187, 394, 435, 272, 96, 17)),
    ("Bhabha-npl-dbox_custom", 4, (35, 184, 402, 457, 291, 103, 18)),
    ("kite_generic_generic", 6, (24, 66, 73, 39, 10)),
    ("par_generic_generic", 7, (15, 33, 27, 9)),
    ("Hj-npl-pentb_custom", 7, (56, 294, 681, 884, 699, 343, 101, 16)),
    ("dpent_zero_zero", 9, (64, 528, 1770, 3158, 3336, 2171, 867, 202, 24)),
    ("npl-dpent_zero_zero", 9, (64, 597, 2117, 3852, 4058, 2606, 1029, 239, 28)),
    ("npl-dpent2_zero_zero", 9, (63, 562, 1969, 3591, 3820, 2482, 988, 230, 27)),
]


@pytest.mark.parametrize(
    ("entry", "dimension", "f_vector"),
    [pytest.param(*row, id=row[0].split("_")[0]) for row in FMT_TABLE_3],
)
def test_fevola_mizera_telen_table_3(entry: str, dimension: int, f_vector: tuple[int, ...]) -> None:
    # The table's f-vector stops at the facets; feynkit's ends with the polytope itself.
    path = FIXTURES / f"{entry}.txt"
    g, variables = read_polynomial(path, (False, False))
    head = path.read_text(encoding="utf-8")
    parameters = re.search(r"^parameters = \[(.*)\]$", head, flags=re.MULTILINE)
    assert parameters is not None
    assert len(g.free_symbols - set(variables)) == len(parameters.group(1).split(",")) == dimension
    data = polytope_data([m for m, _ in sp.Poly(g, *variables).terms()])
    assert data.f_vector == (*f_vector, 1)


@requires_singular
@pytest.mark.slow
def test_fevola_mizera_telen_table_3_inner_dbox_degrees() -> None:
    # The table gives [1^7, 2]_1 for inner-dbox: seven components of degree 1 and one of degree
    # 2. The principal Landau determinant finds all eight, though it skips the 126 faces with
    # more than 14 points; about a minute.
    g, variables = read_polynomial(FIXTURES / "inner-dbox_custom.txt", (False, False))
    analysis = landau_analysis_from_polynomial(g, variables)
    symbols = sorted(g.free_symbols - set(variables), key=lambda x: x.name)
    degrees = sorted(sp.Poly(s, *symbols).total_degree() for s in analysis.landau_surfaces)
    assert degrees == [1] * 7 + [2]


# --- Klausen 2022 (arXiv:2109.07584) ------------------------------------------


@requires_singular
def test_klausen_triangle_face_discriminants() -> None:
    # App. A, p. 36: the principal A-determinant of U + F for the triangle, with every
    # coefficient of F a free symbol, as Macaulay2 lists it face by face.
    x = sp.symbols("x1:4")
    g = sp.sympify("x1 + x2 + x3 + s1*x2*x3 + s2*x1*x3 + s3*x1*x2 + b1*x1**2 + b2*x2**2 + b3*x3**2")
    printed = [
        "s1^2 - 2*s1*s2 + s2^2 - 2*s1*s3 - 2*s2*s3 + s3^2 + 4*s1*b1 + 4*s2*b2 - 4*b1*b2 + 4*s3*b3"
        " - 4*b1*b3 - 4*b2*b3",
        "-s1 + b2 + b3", "1", "1", "-s2 + b1 + b3", "1", "b3", "-s3 + b1 + b2", "1", "1", "1",
        "-s1^2 + 4*b2*b3", "b2", "-s1*s2*s3 + s1^2*b1 + s2^2*b2 + s3^2*b3 - 4*b1*b2*b3", "1", "1",
        "1", "b1", "-s2^2 + 4*b1*b3", "1", "-s3^2 + 4*b1*b2",
    ]  # fmt: skip
    analysis = landau_analysis_from_polynomial(g, list(x))
    assert analysis.skipped_faces == ()

    def up_to_sign(p: sp.Expr) -> str:
        return max(sp.srepr(sp.expand(p)), sp.srepr(sp.expand(-p)))

    found = sorted(up_to_sign(f.discriminant) for f in analysis.face_discriminants)
    assert found == sorted(up_to_sign(sp.sympify(p.replace("^", "**"))) for p in printed)


@requires_singular
def test_klausen_triangle_proper_mixed_faces() -> None:
    # Ex. 4.3, p. 28: of the 21 faces of Newt(U + F), 7 are faces of Newt(U), 7 of Newt(F) and 6
    # proper mixed faces, whose discriminants multiply to R = p_1^2 p_2^2 p_3^2, Eq. 4.38.
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
    faces = landau_analysis(fi).face_discriminants
    assert len(faces) == 21
    degrees = [{sum(e) for e in face.exponents} for face in faces]
    assert degrees.count({1}) == 7
    assert degrees.count({2}) == 7
    mixed = [f for f, d in zip(faces, degrees, strict=True) if d == {1, 2} and f.dimension < 3]
    assert len(mixed) == 6
    p = [x for x in fi.symanzik.g.free_symbols if x.name.startswith("p")]
    assert len(p) == 3
    assert sp.expand(sp.prod(f.discriminant for f in mixed)) == sp.prod(p)


def _dunces_cap_f() -> tuple[sp.Expr, list[sp.Symbol], list[sp.Symbol]]:
    # Eq. 6.1, p. 33, with s_1, s_2, s_3 and m_1^2, ..., m_4^2 as the symbols S and M.
    x = sp.symbols("x1:5")
    kinematics = sp.symbols("S1 S2 S3 M1 M2 M3 M4")
    s1, s2, s3, m1, m2, m3, m4 = kinematics
    f = (
        s1 * x[0] * x[1] * (x[2] + x[3])
        + s2 * x[0] * x[2] * x[3]
        + s3 * x[1] * x[2] * x[3]
        + m1 * x[0] ** 2 * (x[2] + x[3])
        + m2 * x[1] ** 2 * (x[2] + x[3])
        + m3 * x[2] ** 2 * (x[0] + x[1] + x[3])
        + m4 * x[3] ** 2 * (x[0] + x[1] + x[2])
    )
    return sp.expand(f), list(x), list(kinematics)


def _eq_6_6(t3: Fraction, t5: Fraction, t6: Fraction, t8: Fraction, t9: Fraction, t10: Fraction):
    d = (t5 + t6) * (t9 + t10)
    return [
        -(t5 * (t6 * t9 + t3 * t10) + t6 * (t6 * t9 + t8 * t9 + t3 * t10 + t9 * t10))
        / (t5 * t9 * t10),
        -(
            t5 * (t9 + t10) * (t6 * t9 + t3 * t10)
            + t6 * (t6 * t9**2 + t8 * t9 * (t9 + t10) - t10 * (t9**2 + t10 * t9 - t3 * t10))
        )
        / (d * t9 * t10),
        -t6
        * (
            t6 * (t9 + t10) * (t6 * t9 - t8 * t9 + t3 * t10 + t9 * t10)
            + t5 * (t6 * t9**2 + t3 * t10**2)
        )
        / (d * t5 * t9 * t10),
        Fraction(1),
        t6**2 * t8 / (t5**2 * t10),
        -(t6**2) * t9 / (d * t10),
        -t3 * t6 * t10 / (d * t9),
    ]


def _eq_6_7(t4: Fraction, t5: Fraction, t6: Fraction, t8: Fraction, t9: Fraction, t10: Fraction):
    return [
        t6 * ((t4 - t9) * t10 - t8 * t9) / (t5 * t9 * t10),
        -t6
        * (t10 * (t4 * (t9 + t10) + t9 * (2 * t6 + t9 + t10)) - t8 * t9 * (t9 + t10))
        / (t4 * t9 * t10**2),
        -(t6**2)
        * (t8 * t9 * (t9 + t10) + t10 * (t4 * (t9 + t10) - t9 * (-2 * t5 + t9 + t10)))
        / (t4 * t5 * t9 * t10**2),
        Fraction(1),
        t6**2 * t8 / (t5**2 * t10),
        t6**2 * t9 / (t4 * t10**2),
        t6**2 / (t4 * t9),
    ]


@requires_singular
@pytest.mark.parametrize(
    ("parametrisation", "t"),
    [
        pytest.param(_eq_6_6, (2, 3, -1, 5, 7, -4), id="eq-6.6"),
        pytest.param(_eq_6_6, (-3, 1, 4, -2, 5, 3), id="eq-6.6-second"),
        pytest.param(_eq_6_7, (2, 3, -1, 5, 7, -4), id="eq-6.7"),
        pytest.param(_eq_6_7, (-3, 1, 4, -2, 5, 3), id="eq-6.7-second"),
    ],
)
def test_klausen_dunces_cap_leading_landau_variety(parametrisation, t: tuple[int, ...]) -> None:
    # Sec. 6, pp. 33-34: at a point of either parametrisation, with m_1^2 = 1, F has a singular
    # point in the torus on the whole of its Newton polytope, which is three-dimensional.
    f, x, kinematics = _dunces_cap_f()
    point = dict(zip(kinematics, parametrisation(*(Fraction(v) for v in t)), strict=True))
    analysis = face_degeneracy_from_polynomial(f, x, point, check=False, timeout=60)
    (top,) = [face for face in analysis.faces if face.dimension == 3]
    assert top.degenerate is True


@requires_singular
def test_klausen_dunces_cap_off_the_landau_variety() -> None:
    f, x, kinematics = _dunces_cap_f()
    point = dict(zip(kinematics, (Fraction(v) for v in (-4, 4, -5, 7, -15, 1, 11)), strict=True))
    analysis = face_degeneracy_from_polynomial(f, x, point, check=False, timeout=60)
    (top,) = [face for face in analysis.faces if face.dimension == 3]
    assert top.degenerate is False


# --- Bitoun, Bogner, Klausen and Panzer ---------------------------------------

# (CNickel, legs on shell, C). The triangle, box, dunce's cap and diagonal box of Table 2 have
# every leg off shell; the form factors of Table 1 have p_2^2 = p_3^2 = 0 in feynkit's labels.
BBKP = [
    # Table 2, p. 37: massless and massive, every mass distinct.
    pytest.param("12e|2e|e|:zzz", (), 4, id="table-2-triangle-massless"),
    pytest.param("12e|2e|e|:nnn", (), 7, id="table-2-triangle-massive"),
    pytest.param("12e|3e|3e|e|:zzzz", (), 11, id="table-2-box-massless"),
    pytest.param("12e|3e|3e|e|:nnnn", (), 15, id="table-2-box-massive"),
    pytest.param("13e|23|3e||:zzzzz", (), 3, id="table-2-WS3-massless"),
    pytest.param("13e|23|3e||:nnnnn", (), 30, id="table-2-WS3-massive"),
    pytest.param("12e|22e|e|:zzzz", (), 4, id="table-2-dunces-cap-massless"),
    pytest.param("12e|22e|e|:nnnn", (), 19, id="table-2-dunces-cap-massive"),
    pytest.param("123e|2e|3e|e|:zzzzz", (), 20, id="table-2-diagonal-box-massless"),
    pytest.param("123e|2e|3e|e|:nnnnn", (), 55, id="table-2-diagonal-box-massive"),
    # Table 1, p. 33, for the graphs of Fig. 4.1, p. 32.
    pytest.param("12e|2e|e|:zzz", (2, 3), 1, id="table-1-F1"),
    pytest.param("12e|23|4|4e|e|:zzzzzz", (2, 3), 4, id="table-1-F2"),
    pytest.param("123e|23|3e|e|:zzzzzz", (2, 3), 4, id="table-1-F4"),
    pytest.param("12e|23|4|45|5|e|:zzzzzzzz", (), 10, id="table-1-P3"),
    # Prop. 54, p. 33: C = L(L - 1)/2 for the wheel with L = 4 spokes cut at a rim or a spoke.
    pytest.param("1234|2e|3|4|e|:zzzzzzz", (), 6, id="prop-54-WS4-rim"),
    pytest.param("123e|24|3|4|e|:zzzzzzz", (), 6, id="prop-54-WS4-spoke"),
    # Prop. 55, p. 34: 2^(L+1) - 1 for the L-loop sunrise with L + 1 masses.
    pytest.param("11e|e|:nn", (), 3, id="prop-55-L1"),
    pytest.param("111e|e|:nnn", (), 7, id="prop-55-L2"),
    pytest.param("1111e|e|:nnnn", (), 15, id="prop-55-L3"),
    # Prop. 56, p. 35: 2^R for the L-loop sunrise with R <= L masses.
    pytest.param("111e|e|:nzz", (), 2, id="prop-56-L2-R1"),
    pytest.param("111e|e|:nnz", (), 4, id="prop-56-L2-R2"),
    pytest.param("1111e|e|:nzzz", (), 2, id="prop-56-L3-R1"),
    pytest.param("1111e|e|:nnzz", (), 4, id="prop-56-L3-R2"),
    pytest.param("1111e|e|:nnnz", (), 8, id="prop-56-L3-R3"),
    # Ex. 59, p. 38, Fig. 4.4: propagators 1 and 2 massless, 3, 4 and 5 of one mass, and only
    # the leg at the end of propagators 1, 4 and 5 off shell.
    pytest.param("13e|23e|3e|e|:zzaaa", (1, 2, 3), 15, id="example-59"),
]


@requires_singular
@pytest.mark.parametrize(("cnickel", "on_shell", "master"), BBKP)
def test_bitoun_bogner_klausen_panzer(cnickel: str, on_shell: tuple[int, ...], master: int) -> None:
    g, variables = _g(cnickel, on_shell)
    assert _count(g, variables) == master


@requires_singular
def test_bitoun_bogner_klausen_panzer_example_57() -> None:
    # Ex. 57, p. 36: the bubble with m_1^2 = m_2^2 = -p^2 = 1 has C = 3.
    u = sp.symbols("u1:3")
    g = (u[0] + u[1]) * (u[0] + u[1] + 1) + u[0] * u[1]
    assert critical_point_count(sp.expand(g), list(u), {}) == 3


# --- Chestnov, Matsubara-Heo, Munch and Takayama --------------------------------


def _monomials(text: str) -> list[sp.Expr]:
    return [sp.prod(sp.Symbol(f"x{i}") ** int(e) for i, e in term) for term in _terms(text)]


def _terms(text: str) -> list[list[tuple[str, str]]]:
    """'1.2 1.4^2' -> [[('1', '1'), ('2', '1')], [('1', '1'), ('4', '2')]]."""
    return [
        [(m.group(1), m.group(2) or "1") for m in re.finditer(r"(\d)(?:\^(\d))?", word)]
        for word in text.split()
    ]


# The monomials of g in the order of the paper, written i.j^2 for x_i x_j^2, and the
# coefficients that are not 1, by position.
CMMT = [
    pytest.param(
        "1.2 1.4 1.5 2.3 2.4 2.5 3.4 3.5 1.2.4 2.3.5",
        {9: sp.Rational(3, 7)},
        9,
        3,
        id="example-2.2-N-box",
    ),
    pytest.param("1 2 3 1^2 1.2 1.3", {5: sp.Rational(5, 17)}, 3, 2, id="5.5.1-triangle"),
    pytest.param(
        "1.2 1.4 1.5 2.3 2.4 2.5 3.4 3.5 1.2.4 1.4^2 1.4.5 2.3.4 2.3.5 2.4^2 2.4.5 3.4^2 3.4.5",
        {8: sp.Rational(18, 17), 12: sp.Rational(1, 19)},
        33,
        7,
        id="5.5.2-N-box-one-mass",
    ),
    pytest.param(
        "1.4 1.5 1.6 2.4 2.5 2.6 3.4 3.5 3.6 4.5 4.6 4.7 5.7 6.7 1.3.4 1.3.5 1.3.6 1.4.5 2.4.6 "
        "2.4.7 2.5.7 2.6.7",
        dict.fromkeys((18, 19, 20, 21), sp.Rational(1, 7)),
        115,
        7,
        id="5.5.3-diagonal-box",
    ),
    pytest.param(
        "1.2 1.3 1.6 1.7 2.3 2.4 2.5 3.4 3.5 3.6 3.7 4.6 4.7 5.6 5.7 1.2.5 1.2.6 1.3.5 1.3.6 "
        "1.5.6 1.5.7 2.3.5 2.3.6 2.4.6 2.5.6 3.4.7",
        {25: sp.Rational(-8, 7)},
        238,
        12,
        id="5.5.4-double-box",
    ),
]


@requires_singular
@pytest.mark.parametrize(("monomials", "special", "rank", "master"), CMMT)
def test_chestnov_matsubara_heo_munch_takayama(
    monomials: str, special: dict[int, sp.Rational], rank: int, master: int
) -> None:
    # The GKZ rank at generic coefficients is the volume; C is the rank they find, or expect
    # from the Euler characteristic, at the coefficients of the Feynman integral, every
    # coefficient 1 but those in special, which take a random value of the kinematic ratio.
    terms = _monomials(monomials)
    variables = sorted(set().union(*(t.free_symbols for t in terms)), key=lambda s: s.name)
    assert _volume(sp.Add(*terms), variables) == rank
    g = sp.Add(*(special.get(i, 1) * t for i, t in enumerate(terms)))
    assert critical_point_count(g, variables, {}) == master


@requires_singular
def test_chestnov_matsubara_heo_munch_takayama_sunrise() -> None:
    # Eq. 5.78, p. 44: the Pfaffian system of the sunrise with three distinct masses has rank 7.
    g, variables = _g("111e|e|:nnn")
    assert _count(g, variables, seed=1) == 7
