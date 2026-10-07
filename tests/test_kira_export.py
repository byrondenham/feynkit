"""The Kira export: job files written from an integral family.

Wei22 is S. Weinzierl, Feynman Integrals, Springer 2022, arXiv:2201.03593; its App. J
(Exercise 44, pp. 707-709) reduces the planar double box with Kira. The exported text is
parsed back here with a few lines of its own, and compared with the family's functions.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from feynkit import FeynmanIntegral
from feynkit.core import Edge, Graph
from feynkit.core.exceptions import ValidationError
from feynkit.exporters import (
    KiraJob,
    kira_job,
    read_masters,
    read_sector_mappings,
    read_trivial_sectors,
)
from feynkit.family import FamilyFunction, IntegralFamily

BOX = ("13e|2e|3e|e|:zzzz", "massless_on_shell")
SUNRISE = ("111e|e|:nnn", None)
KITE = ("12e|23|3e|e|:nnnnn", None)
DOUBLE_BOX = ("15e|24|3e|4e|5|e|:zzzzzzz", "massless_on_shell")
LADDER = ("123|45|46|5e|7|e|7e|e|:zzzzzzzzzz", None)

ROUND_TRIP = [BOX, SUNRISE, KITE, DOUBLE_BOX, LADDER]


def _family(instance: tuple[str, str | None]) -> IntegralFamily:
    cnickel, kinematics = instance
    fi = FeynmanIntegral.from_cnickel(cnickel, kinematics=kinematics)  # type: ignore[arg-type]
    return IntegralFamily.from_integral(fi)


def _weinzierl() -> IntegralFamily:
    """The guide's family: Wei22 Exercise 44 with k1 = q3, k2 = q6 and his two ISPs."""
    ends = [(1, 2), (2, 4), (3, 1), (4, 3), (6, 4), (3, 5), (5, 6)]
    edges = [Edge(idx=e, v1=a, v2=b, is_internal=True, mass=0) for e, (a, b) in enumerate(ends, 1)]
    edges += [
        Edge(idx=7 + j, v1=v, v2=6 + j, is_internal=False) for j, v in enumerate([1, 2, 6, 5], 1)
    ]
    graph = Graph(internal_vertices=6, external_legs=4, edges=edges)
    fi = FeynmanIntegral(graph).with_kinematics("massless_on_shell")
    isps = [FamilyFunction.squared((1, 0), (1, 0, 1)), FamilyFunction.squared((0, 1), (-1, 0, -1))]
    return IntegralFamily.from_integral(fi, isps=isps, chords=(3, 6))


# A parser for the text the export writes; it is not a YAML parser.


def _momentum(text: str) -> dict[str, int]:
    """``l1-2*p1`` as {name: coefficient}."""
    out: dict[str, int] = {}
    for sign, coefficient, name in re.findall(r"([+-]?)(?:(\d+)\*)?([lp]\d+)", text):
        out[name] = out.get(name, 0) + (-1 if sign == "-" else 1) * int(coefficient or 1)
    return out


def _normal(momentum: dict[str, int]) -> dict[str, int]:
    """The momentum up to its sign: the first coefficient positive, in the order l, p."""
    items = sorted(momentum.items(), key=lambda kv: (kv[0][0] != "l", int(kv[0][1:])))
    items = [(k, v) for k, v in items if v]
    if items and items[0][1] < 0:
        items = [(k, -v) for k, v in items]
    return dict(items)


def _propagators(text: str) -> list[tuple]:
    out: list[tuple] = []
    for line in text.splitlines():
        line = line.strip()
        if match := re.fullmatch(r'- \["([^"]*)", (0|"[^"]*")\]', line):
            mass = match.group(2).strip('"')
            out.append(("squared", _momentum(match.group(1)), sp.sympify(mass.replace("^", "**"))))
        elif match := re.fullmatch(r'- \{ bilinear: \[ \["(l\d+)", "(p\d+)"\], 0 \] \}', line):
            out.append(("bilinear", match.group(1), match.group(2)))
    return out


def _rules(text: str) -> dict[tuple[int, int], sp.Expr]:
    out = {}
    for j, k, expr in re.findall(r"- \[\[p(\d+), p(\d+)\], ([^\]]*)\]", text):
        out[(int(j), int(k))] = sp.sympify(expr.replace("^", "**"))
    return out


def _plain(expr: sp.Expr, names: dict[str, str] | None = None) -> sp.Expr:
    """The expression with plain symbols, renamed as the export did."""
    names = names or {}
    return expr.subs({s: sp.Symbol(names.get(s.name, s.name)) for s in expr.free_symbols})


def _pairing(rules: dict[tuple[int, int], sp.Expr], a: str, b: str) -> sp.Expr:
    """a . b for two names of loop momenta or external momenta, with the rules for p . p."""
    if a[0] == "p" and b[0] == "p":
        j, k = sorted((int(a[1:]), int(b[1:])))
        return rules[(j, k)]
    first, second = sorted((a, b))
    return sp.Symbol(f"{first}.{second}")


def _square(rules: dict[tuple[int, int], sp.Expr], q: dict[str, int]) -> sp.Expr:
    return sp.expand(sum(c * d * _pairing(rules, a, b) for a, c in q.items() for b, d in q.items()))


def _family_function(
    rules: dict[tuple[int, int], sp.Expr], f: FamilyFunction, names: dict[str, str]
) -> sp.Expr:
    """D = a l.l + 2 b l.p + e p.p + c, in the same symbols as :func:`_pairing`."""
    total = sp.sympify(f.constant)
    for i, row in enumerate(f.quadratic):
        for j, x in enumerate(row):
            total += sp.Rational(x.numerator, x.denominator) * _pairing(
                rules, f"l{i + 1}", f"l{j + 1}"
            )
    for i, row in enumerate(f.linear):
        for k, x in enumerate(row):
            total += (
                2
                * sp.Rational(x.numerator, x.denominator)
                * _pairing(rules, f"l{i + 1}", f"p{k + 1}")
            )
    for j, row in enumerate(f.external):
        for k, x in enumerate(row):
            total += sp.Rational(x.numerator, x.denominator) * _pairing(
                rules, f"p{j + 1}", f"p{k + 1}"
            )
    return sp.expand(_plain(total, names))


class TestWeinzierlDoubleBox:
    """Wei22 App. J, pp. 707-708, after the map p_j -> -p_j between outgoing and incoming."""

    def test_propagators(self) -> None:
        job = kira_job(_weinzierl(), name="doublebox")
        # q_1, ..., q_9 of Eq. J.258, with k1, k2 as l1, l2 and p_j -> -p_j.
        his = [
            "l1+p1",
            "l1+p1+p2",
            "l1",
            "l1+l2",
            "l2-p1-p2",
            "l2",
            "l2-p1-p2-p3",
            "l1+p1+p3",
            "l2-p1-p3",
        ]
        mine = _propagators(job.integralfamilies)
        assert [item[0] for item in mine] == ["squared"] * 9
        assert [_normal(item[1]) for item in mine] == [_normal(_momentum(q)) for q in his]
        assert all(item[2] == 0 for item in mine)

    def test_top_sector_and_name(self) -> None:
        job = kira_job(_weinzierl(), name="doublebox")
        assert 'name: "doublebox"' in job.integralfamilies
        assert "top_level_sectors: [127]" in job.integralfamilies
        assert "loop_momenta: [l1, l2]" in job.integralfamilies

    def test_kinematics(self) -> None:
        job = kira_job(_weinzierl(), name="doublebox")
        text = job.kinematics
        assert "incoming_momenta: [p1, p2, p3, p4]" in text
        assert "outgoing_momenta: []" in text
        assert "momentum_conservation: [p4, -p1-p2-p3]" in text
        rules = _rules(text)
        s, t = sp.symbols("s12 s23")
        # Wei22: p1^2 = p2^2 = p3^2 = 0, (p1 + p2)^2 = s, (p2 + p3)^2 = t, (p1 + p3)^2 = -s - t
        # which the map p_j -> -p_j leaves as it is.
        assert [rules[(j, j)] for j in (1, 2, 3)] == [0, 0, 0]
        assert 2 * rules[(1, 2)] == s
        assert 2 * rules[(2, 3)] == t
        assert sp.expand(2 * rules[(1, 3)] + s + t) == 0
        assert "- [s12, 2]" in text and "- [s23, 2]" in text

    def test_signs(self) -> None:
        assert kira_job(_weinzierl()).signs == (-1,) * 9


@pytest.mark.parametrize("instance", ROUND_TRIP)
def test_round_trip(instance: tuple[str, str | None]) -> None:
    """Each exported function, parsed back, is s D with the kinematics substituted."""
    family = _family(instance)
    job = kira_job(family)
    names = dict(job.symbols)
    rules = {key: _plain(value) for key, value in _rules(job.kinematics).items()}
    parsed = _propagators(job.integralfamilies)
    assert len(parsed) == family.size == len(job.signs)
    for item, f, sign in zip(parsed, family.functions, job.signs, strict=True):
        assert item[0] == "squared"
        kira = sp.expand(_square(rules, item[1]) - _plain(item[2]))
        assert kira == sign * _family_function(rules, f, names), f.label


def _sunrise(masses: tuple[object, object, object] = (0, 0, 0)) -> FeynmanIntegral:
    """The two-loop sunrise with three given masses and two legs."""
    edges = [Edge(idx=i, v1=1, v2=2, is_internal=True, mass=m) for i, m in enumerate(masses, 1)]
    edges += [
        Edge(idx=4, v1=1, v2=3, is_internal=False),
        Edge(idx=5, v1=2, v2=4, is_internal=False),
    ]
    return FeynmanIntegral(Graph(internal_vertices=2, external_legs=2, edges=edges))


class TestFunctions:
    def test_scalar_products_are_bilinear(self) -> None:
        isps = [
            FamilyFunction.product(1, 1, loops=2, externals=1),
            FamilyFunction.product(2, 1, loops=2, externals=1),
        ]
        family = IntegralFamily.from_integral(_sunrise(), isps=isps)
        job = kira_job(family)
        assert 'bilinear: [ ["l1", "p1"], 0 ]' in job.integralfamilies
        assert 'bilinear: [ ["l2", "p1"], 0 ]' in job.integralfamilies
        assert job.signs == (-1, -1, -1, 1, 1)
        rules = _rules(job.kinematics)
        names = dict(job.symbols)
        for item, f, sign in zip(
            _propagators(job.integralfamilies), family.functions, job.signs, strict=True
        ):
            if item[0] == "bilinear":
                kira = _pairing(rules, item[1], item[2])
            else:
                kira = _square(rules, item[1]) - _plain(item[2])
            assert sp.expand(kira - sign * _family_function(rules, f, names)) == 0

    def test_negative_scalar_product(self) -> None:
        minus = FamilyFunction(
            label="-l1.p1",
            quadratic=((Fraction(0),) * 2,) * 2,
            linear=((Fraction(-1, 2),), (Fraction(0),)),
            external=((Fraction(0),),),
        )
        plus = FamilyFunction.product(2, 1, loops=2, externals=1)
        family = IntegralFamily.from_integral(_sunrise(), isps=[minus, plus])
        assert kira_job(family).signs == (-1, -1, -1, -1, 1)

    def test_masses(self) -> None:
        m1, m2 = sp.symbols("m1 m2", positive=True)
        family = IntegralFamily.from_integral(_sunrise((m1, m2, m1)))
        job = kira_job(family)
        assert '- ["l1+l2-p1", "m1^2"]' in job.integralfamilies
        assert '- ["l1", "m2^2"]' in job.integralfamilies
        assert "- [m1, 1]" in job.kinematics and "- [m2, 1]" in job.kinematics

    def test_invariant_with_a_power(self) -> None:
        job = kira_job(_family(KITE))
        assert ("p1^2", "p1sq") in job.symbols
        assert "- [p1sq, 2]" in job.kinematics
        assert "^" not in job.kinematics.replace("p1^2", "")

    def test_replace_by_one(self) -> None:
        job = kira_job(_family(DOUBLE_BOX), replace_by_one="s12")
        assert "symbol_to_replace_by_one: s12" in job.kinematics
        with pytest.raises(ValidationError, match="not one of the invariants"):
            kira_job(_family(DOUBLE_BOX), replace_by_one="u")


class TestRefusals:
    def test_function_kira_cannot_express(self) -> None:
        fi = _sunrise()
        mixed = FamilyFunction(
            label="l1.l2",
            quadratic=((Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(0))),
            linear=((Fraction(0),), (Fraction(0),)),
            external=((Fraction(0),),),
        )
        family = IntegralFamily.from_integral(
            fi, isps=[FamilyFunction.product(1, 1, loops=2, externals=1), mixed]
        )
        with pytest.raises(ValidationError, match="Kira cannot express l1.l2"):
            kira_job(family)

    def test_positive_square(self) -> None:
        positive = FamilyFunction(
            label="+(l1 + p1)^2",
            quadratic=((Fraction(1), Fraction(0)), (Fraction(0), Fraction(0))),
            linear=((Fraction(1),), (Fraction(0),)),
            external=((Fraction(1),),),
        )
        family = IntegralFamily.from_integral(
            _sunrise(), isps=[positive, FamilyFunction.product(2, 1, loops=2, externals=1)]
        )
        with pytest.raises(ValidationError, match="Kira cannot express"):
            kira_job(family)

    @pytest.mark.parametrize("name", ["d", "l1", "m.1", "I"])
    def test_symbol_names(self, name: str) -> None:
        family = IntegralFamily.from_integral(_sunrise((sp.Symbol(name), 0, 0)))
        with pytest.raises(ValidationError, match="Kira"):
            kira_job(family)

    def test_inhomogeneous_kinematics(self) -> None:
        m = sp.Symbol("m")
        fi = _sunrise((m, 0, 0)).with_(momentum_products={(1, 2): -(m**2) + 1})
        with pytest.raises(ValidationError, match="mass dimension"):
            kira_job(IntegralFamily.from_integral(fi))

    def test_bad_name_and_sector(self) -> None:
        family = _family(BOX)
        with pytest.raises(ValidationError, match="family name"):
            kira_job(family, name="two words")
        with pytest.raises(ValidationError, match="top sector"):
            kira_job(family, top_sectors=[16])

    def test_overwrite(self, tmp_path: Path) -> None:
        job = kira_job(_family(BOX), name="box")
        job.write(tmp_path / "run")
        assert (tmp_path / "run" / "config" / "integralfamilies.yaml").read_text() == (
            job.integralfamilies
        )
        with pytest.raises(ValidationError, match="refusing to overwrite"):
            job.write(tmp_path / "run")
        job.write(tmp_path / "run", overwrite=True)
        assert isinstance(job, KiraJob)


RECORDED = Path(__file__).parent / "data" / "kira" / "doublebox"


class TestRecordedRun:
    """Kira 3.1 on the exported double box, r = 8, s = 2 and s12 set to one.

    The fixtures are the files that run wrote (the inputs, the zero sectors, the sector
    relations and the masters); the run took 49 s with Fermat 7.9b.
    """

    def test_inputs_are_the_export(self) -> None:
        job = kira_job(_weinzierl(), name="doublebox", r=8, s=2, replace_by_one="s12")
        assert job.jobs == (RECORDED / "jobs.yaml").read_text()
        assert job.integralfamilies == (RECORDED / "config" / "integralfamilies.yaml").read_text()
        assert job.kinematics == (RECORDED / "config" / "kinematics.yaml").read_text()

    def test_trivial_sectors_are_the_zero_sectors(self) -> None:
        family = _weinzierl()
        trivial = read_trivial_sectors(RECORDED, "doublebox", propagators=7)
        assert trivial == {s.id for s in family.integral.sectors().zero()}
        assert len(trivial) == 85
        assert max(read_trivial_sectors(RECORDED, "doublebox")) > 127

    def test_sector_mappings(self) -> None:
        mappings = read_sector_mappings(RECORDED, "doublebox")
        assert mappings[28] == 42 and mappings[93] == 107 and mappings[29] == 43
        assert all(0 < s < 128 and 0 < t < 128 for s, t in mappings.items())
        assert read_sector_mappings(RECORDED, "doublebox", propagators=5) == {
            s: t for s, t in mappings.items() if s < 32
        }

    def test_masters_are_in_weinzierls_sectors(self) -> None:
        """Wei22 Eq. 6.83: sectors 28, 73, 54, 57, 79, 93 and 127 twice, up to the mappings."""
        family = _weinzierl()
        masters = read_masters(RECORDED, "doublebox")
        assert len(masters) == 8
        assert masters[-2:] == ((1, 1, 1, 1, 1, 1, 1, 0, 0), (1, 1, 1, 1, 1, 1, 1, -1, 0))
        sectors = [family.sector_id(n) for n in masters]
        mappings = read_sector_mappings(RECORDED, "doublebox")
        his = [28, 73, 54, 57, 79, 93, 127, 127]
        assert sorted(sectors) == sorted(mappings.get(s, s) for s in his)
        assert sorted(sectors) == [42, 54, 57, 73, 79, 107, 127, 127]

    def test_bad_outputs(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="cannot read"):
            read_masters(tmp_path, "doublebox")
        results = tmp_path / "results" / "doublebox"
        results.mkdir(parents=True)
        (results / "masters").write_text("other[1,1]  # 3\n")
        with pytest.raises(ValidationError, match="not a master"):
            read_masters(tmp_path, "doublebox")
        (results / "masters").write_text("doublebox[1,1]  # 3\n")
        assert read_masters(tmp_path, "doublebox") == ((1, 1),)
        mapping = tmp_path / "sectormappings" / "doublebox"
        mapping.mkdir(parents=True)
        (mapping / "sectorRelations").write_text("1 a b 1 2 3 0 {x} {y}\n1 a b 1 4 3 0 {x} {y}\n")
        with pytest.raises(ValidationError, match="mapped to 2 and 4"):
            read_sector_mappings(tmp_path, "doublebox")
        (mapping / "sectorRelations").write_text("junk\n")
        with pytest.raises(ValidationError, match="not a sector relation"):
            read_sector_mappings(tmp_path, "doublebox")
        (mapping / "trivialsector").write_text("1,x\n")
        with pytest.raises(ValidationError, match="not a list"):
            read_trivial_sectors(tmp_path, "doublebox")
        (mapping / "trivialsector").write_text("\n")
        assert read_trivial_sectors(tmp_path, "doublebox") == frozenset()


class TestOtherLoopOrders:
    """The sector relations of one-loop and three-loop runs have a different number of columns."""

    def test_one_loop_box(self) -> None:
        box = Path(__file__).parent / "data" / "kira" / "box"
        assert read_sector_mappings(box, "box") == {7: 14, 11: 13}
        assert read_trivial_sectors(box, "box", propagators=4) >= {0, 1, 2, 3}

    def test_three_loop_chain(self) -> None:
        chain = Path(__file__).parent / "data" / "kira" / "chain"
        mappings = read_sector_mappings(chain, "chain")
        assert mappings == {29: 23, 53: 23, 55: 31, 61: 31}


class TestVacuum:
    @staticmethod
    def _family() -> IntegralFamily:
        m = sp.Symbol("m", positive=True)
        edges = [
            Edge(idx=i, v1=1, v2=2, is_internal=True, mass=mass)
            for i, mass in enumerate((m, m, 0), 1)
        ]
        graph = Graph(internal_vertices=2, external_legs=0, edges=edges)
        return IntegralFamily.from_integral(FeynmanIntegral(graph, momentum_products={}))

    def test_kinematics_without_legs(self) -> None:
        job = kira_job(self._family(), name="vac", integrals=[(2, 1, 1)])
        assert job.kinematics == (
            "kinematics:\n"
            "  incoming_momenta: []\n"
            "  outgoing_momenta: []\n"
            "  kinematic_invariants:\n"
            "    - [m, 1]\n"
            "  scalarproduct_rules: []\n"
        )
        assert 'name: "vac"' in job.integralfamilies
        assert '- ["l1+l2", "m^2"]' in job.integralfamilies
        assert job.integrals == "vac[2,1,1]\n"

    def test_inputs_are_the_recorded_run(self) -> None:
        """Kira 3.1 reduced vac(2,1,1) with these files to (d-2)/(4 m^4) vac(1,1,0)."""
        recorded = Path(__file__).parent / "data" / "kira" / "vac"
        job = kira_job(self._family(), name="vac", integrals=[(2, 1, 1)], s=0)
        assert job.jobs == (recorded / "jobs.yaml").read_text()
        assert job.integrals == (recorded / "integrals").read_text()
        assert job.integralfamilies == (recorded / "config" / "integralfamilies.yaml").read_text()
        assert job.kinematics == (recorded / "config" / "kinematics.yaml").read_text()
        result = (recorded / "results" / "vac" / "kira_integrals.inc").read_text()
        assert "vac(1,1,0)" in result and "(d-2)" in result


class TestJobs:
    def test_integrals_job(self) -> None:
        family = _weinzierl()
        job = kira_job(family, name="db", integrals=[(1, 1, 1, 1, 1, 1, 1, -1, -1)])
        assert job.integrals == "db[1,1,1,1,1,1,1,-1,-1]\n"
        assert "r: 8, s: 2}" in job.jobs
        assert "select_mandatory_list:\n          - [db, integrals]" in job.jobs
        assert "kira2form" in job.jobs and "reconstruct_mass" not in job.jobs
        assert (
            "reconstruct_mass: true"
            in kira_job(family, integrals=[(1,) * 7 + (0, 0)], replace_by_one="s12").jobs
        )

    def test_defaults_and_sectors(self) -> None:
        job = kira_job(_weinzierl(), top_sectors=[127, 28])
        assert "sectors: [127, 28], r: 8, s: 1}" in job.jobs
        assert job.integrals is None and "kira2form" not in job.jobs

    def test_default_seed_range_has_a_numerator(self) -> None:
        """Too small an s lists spurious masters: the double box gives 13 at s = 0, 8 at s = 1,
        and the massless kite, with no ISPs, 3 at s = 0 and 2 at s = 1 (Kira 3.1)."""
        assert "r: 8, s: 1}" in kira_job(_weinzierl()).jobs
        assert "r: 8, s: 3}" in kira_job(_weinzierl(), integrals=[(1,) * 7 + (-1, -2)]).jobs
        assert "r: 5, s: 1}" in kira_job(_family(BOX)).jobs
        kite = IntegralFamily.from_integral(FeynmanIntegral.from_cnickel("12e|23|3|e|:zzzzz"))
        assert "r: 6, s: 1}" in kira_job(kite).jobs
        assert "r: 8, s: 0}" in kira_job(_weinzierl(), s=0).jobs

    def test_bad_arguments(self) -> None:
        family = _weinzierl()
        with pytest.raises(ValidationError, match="non-negative"):
            kira_job(family, r=-1)
        with pytest.raises(ValidationError, match="ISP"):
            kira_job(family, integrals=[(1,) * 8 + (1,)])
        with pytest.raises(ValidationError, match="indices"):
            kira_job(family, integrals=[(1, 1)])


def test_guide_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first code block of the guide's Kira subsection, run on the guide's family."""
    guide = (Path(__file__).resolve().parents[1] / "docs" / "guide.md").read_text(encoding="utf-8")
    section = guide.split("\n### Exporting a family to Kira\n", 1)[1].split("\n## ", 1)[0]
    block = section.split("```python\n", 1)[1].split("```", 1)[0]
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"family": _weinzierl()}
    exec(compile(block, "docs/guide.md", "exec"), namespace)
    job = namespace["job"]
    assert isinstance(job, KiraJob)
    shown = [line[2:] for line in block.splitlines() if line.startswith("# ")]
    assert shown == job.integralfamilies.splitlines()
    assert (tmp_path / "doublebox_run" / "jobs.yaml").read_text() == job.jobs
