"""Tests for the ``fk`` command-line entry point."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

import feynkit.cli as cli
from feynkit import FeynmanIntegral
from feynkit.cli import main

# One printed Euler equation, '    [r]  A_r1 z_1 d_1 + ...  =  beta_r', and one of its terms.
_EULER = re.compile(r"^    \[(\d+)\]  (.*)  =  (.*)$")
_TERM = re.compile(r"^(?:(\d+) )?z_(\d+) d_\2$")


def _run(capsys: pytest.CaptureFixture[str], tmp_path: Path, *args: str) -> str:
    main([*args, "--db", str(tmp_path / "cli.db")])
    return capsys.readouterr().out


class TestSingleDiagram:
    def test_prints_nickel_index(self, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
        out = _run(capsys, tmp_path, "11e|e|:zz", "-s")
        assert "11e|e|" in out

    def test_section_flags_restrict_output(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        symanzik_only = _run(capsys, tmp_path, "11e|e|:zz", "-s")
        gkz_only = _run(capsys, tmp_path, "11e|e|:zz", "-g")
        assert symanzik_only != gkz_only
        assert "Euler" in gkz_only
        assert "Euler" not in symanzik_only

    def test_bare_nickel_is_treated_as_massless(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "11e|e|", "-s")
        assert "11e|e|:zz" in out

    def test_creates_database_file(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        _run(capsys, tmp_path, "11e|e|:zz", "-t")
        assert (tmp_path / "cli.db").exists()

    def test_trivial_toric_ideal_claims_no_master_integral_count(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "11e|e|:zz", "-t")
        assert "Toric ideal  (generators: an analogue of IBP relations)" in out
        assert "Trivial  (the zero ideal)" in out
        assert "master integral" not in out

    def test_symmetry_pairs_are_all_unimodular(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "11e|e|:zz", "-S")
        assert "Symmetry pairs  (|det M|=1)  6" in out
        assert "finite-index" not in out

    def test_symanzik_parameters_are_the_symbols_of_u(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        lines = _run(capsys, tmp_path, "12e|2e|e|:zzz", "-s").splitlines()
        params = next(line for line in lines if line.startswith("  Schwinger params"))
        u = next(line for line in lines if line.startswith("  U  ="))
        symbols = re.compile(r"\b[a-z]+_\d+\b")
        assert symbols.findall(params) == ["a_1", "a_2", "a_3"]
        assert set(symbols.findall(u)) == set(symbols.findall(params))

    @pytest.mark.parametrize("cnickel", ["11e|e|:nn", "111e|e|:nnn"])
    def test_euler_equations_have_the_rows_of_a(
        self, cnickel: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # A mass m_e puts m_e^2 u_e^2 into G, so row e of A has a 2; the massive
        # sunrise has two in each row but the first.
        gkz = FeynmanIntegral.from_cnickel(cnickel).gkz
        rows = [[int(x) for x in gkz.a_matrix.row(r)] for r in range(gkz.a_matrix.rows)]
        assert sum(max(row) >= 2 for row in rows) >= 2
        out = _run(capsys, tmp_path, cnickel, "-g")
        equations = [match for line in out.splitlines() if (match := _EULER.match(line))]
        assert [int(match[1]) for match in equations] == list(range(len(rows)))
        for match in equations:
            r = int(match[1])
            printed = [0] * len(rows[r])
            for term in match[2].split(" + "):
                parsed = _TERM.match(term)
                assert parsed, term
                printed[int(parsed[2]) - 1] = int(parsed[1] or 1)
            assert printed == rows[r]
            assert match[3] == str(gkz.beta_parameters[r])
        assert (equations[0][3], equations[1][3]) == ("-D/2", "-nu_1")

    def test_massive_tadpole_prints_every_section(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # With no external legs there are no Mandelstam invariants, and the
        # Newton polytope is a segment.
        out = _run(capsys, tmp_path, "0|:n")
        assert "External legs                0" in out
        assert "F  =  a_1**2*m_1**2/mu**2" in out
        assert "Hull vertices                2" in out
        assert "Normalised volume            1  (the holonomic rank" in out
        assert f"  {'|Aut(P)|':<28} 2  (polytope automorphisms)\n" in out
        assert "Stored:" in out

    @pytest.mark.parametrize(
        ("cnickel", "ambient", "affine", "volume"),
        [
            ("01e|e|:zn", 2, 1, 1),
            ("00|:nz", 2, 1, 1),
            # The massless triangle and the massive bubble, each with a massless self-loop.
            ("012e|2e|e|:zzzz", 4, 3, 4),
            ("011e|e|:znn", 3, 2, 3),
        ],
    )
    def test_newton_section_of_a_polytope_that_is_not_full_dimensional(
        self,
        cnickel: str,
        ambient: int,
        affine: int,
        volume: int,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        # A massless self-loop is scaleless: G is its parameter times the G of
        # the rest of the graph, so P is the polytope of the rest, with its
        # volume, lifted into one more dimension. The rows of A are then
        # dependent, so the volume is not the rank.
        lines = _run(capsys, tmp_path, cnickel).splitlines()
        assert f"  Ambient dimension            {ambient}" in lines
        assert f"  Affine dimension             {affine}" in lines
        assert f"  {'Scaleless':<28} yes" in lines
        assert f"  Normalised volume            {volume}" in lines
        assert any(line.startswith("  Lattice base point") for line in lines)

    def test_symmetries_below_full_dimension(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # The permutations of u_2, u_3 and u_4 preserve the monomials of G.
        out = _run(capsys, tmp_path, "012e|2e|e|:znnn", "-S")
        assert f"  {'|Aut(P)|':<28} 6  (automorphisms of P in its affine hull)\n" in out
        assert f"  {'Vertex orbits under Aut(P)':<28} [[0, 1, 2], [3, 4, 5]]\n" in out
        assert f"  {'Symmetry pairs  (|det M|=1)':<28} 6\n" in out
        assert (
            "  P is not full-dimensional: the identities of the pairs hold only trivially.\n"
        ) in out

    @pytest.mark.parametrize(
        ("cnickel", "symmetries"),
        [
            ("1e|e|", f"  {'|Aut(P)|':<28} 2  (polytope automorphisms)\n"),
            ("12e|e|e|", f"  {'|Aut(graph)|':<28} 2\n"),
        ],
    )
    def test_tree_graph_prints_every_section_and_writes_its_report(
        self, cnickel: str, symmetries: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # A tree has U = 1. fk analyse failed on one before 0.3.0.
        text = tmp_path / "tree.txt"
        out = _run(capsys, tmp_path, cnickel, "--text", str(text))
        assert f"  {'Loop count':<28} 0\n" in out
        assert symmetries in out
        assert "Stored:" in out
        assert text.read_text(encoding="utf-8")

    def test_polytope_automorphisms_line_up_with_the_other_keys(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "12e|2e|e|:zzz", "-S")
        assert f"  {'|Aut(P)|':<28} 48  (polytope automorphisms)\n" in out

    def test_newton_section_of_a_full_dimensional_polytope(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "12e|2e|e|:zzz", "-n")
        assert "Normalised volume            4  (the holonomic rank for generic beta)" in out
        assert f"  {'Scaleless':<28} no\n" in out
        assert "Lattice base point           (1, 1, 0)" in out


class TestLatticeInvariants:
    def test_newton_section_gives_them(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        found = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz").lattice_invariants()
        assert found.normal and found.gorenstein_index not in (None, 1)
        out = _run(capsys, tmp_path, "12e|2e|e|:zzz", "-n")
        interior = f"{found.lattice_points}  ({found.interior_points} interior)"
        assert f"  {'Lattice points':<28} {interior}\n" in out
        assert f"  {'h*-vector':<28} {found.h_star}\n" in out
        assert f"  {'Gorenstein index':<28} {found.gorenstein_index}\n" in out
        assert f"  {'Lattice width':<28} {found.lattice_width}\n" in out
        assert f"  {'Integer decomposition (IDP)':<28} yes\n" in out
        assert f"  {'Normal configuration':<28} yes\n" in out
        assert (
            "  NA is normal, so C[NA] is Cohen-Macaulay (Hochster 1972)\n"
            "  and there are no rank jumps (Matusevich, Miller and Walther 2005).\n"
        ) in out

    def test_below_full_dimension_only_cohen_macaulay_is_claimed(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "012e|2e|e|:zzzz", "-n")
        assert "  NA is normal, so C[NA] is Cohen-Macaulay (Hochster 1972).\n" in out
        assert "rank jumps" not in out

    def test_a_configuration_that_is_not_normal(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        found = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz").lattice_invariants()
        changed = dataclasses.replace(
            found,
            lattice_points=found.lattice_points + 1,
            gorenstein_index=1,
            reflexive=True,
            idp=False,
            support_is_saturated=False,
            normal=False,
        )
        monkeypatch.setattr(FeynmanIntegral, "lattice_invariants", lambda self, *a, **k: changed)
        out = _run(capsys, tmp_path, "12e|2e|e|:zzz", "-n")
        assert f"  {'Gorenstein index':<28} 1  (reflexive)\n" in out
        assert f"  {'Integer decomposition (IDP)':<28} no\n" in out
        reasons = "no  (P is not IDP; the support misses 1 lattice point of P)"
        assert f"  {'Normal configuration':<28} {reasons}\n" in out
        assert "Cohen-Macaulay" not in out


class TestLatticeInvariantsNotComputed:
    def test_newton_section_says_what_was_not_computed(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        found = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz").lattice_invariants()
        unknown = dataclasses.replace(found, ehrhart=None, h_star=None, idp=None, normal=None)
        monkeypatch.setattr(FeynmanIntegral, "lattice_invariants", lambda self, *a, **k: unknown)
        out = _run(capsys, tmp_path, "12e|2e|e|:zzz", "-n")
        assert f"  {'h*-vector':<28} not computed\n" in out
        assert f"  {'Integer decomposition (IDP)':<28} not computed\n" in out
        reason = "not computed  (install PyNormaliz, or call lattice_invariants directly)"
        assert f"  {'Normal configuration':<28} {reason}\n" in out
        assert "Cohen-Macaulay" not in out

    def test_json_gives_null(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        found = FeynmanIntegral.from_cnickel("11e|e|:zz").lattice_invariants()
        unknown = dataclasses.replace(found, idp=None, normal=None)
        monkeypatch.setattr(FeynmanIntegral, "lattice_invariants", lambda self, *a, **k: unknown)
        main(["analyse", "11e|e|:zz", "--json", "--sections", "polytope", "--no-db"])
        assert json.loads(capsys.readouterr().out)["summary"]["normal_configuration"] is None


def test_fk_analyse_computes_the_invariants_once_within_the_budget(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import feynkit.lattice_invariants as module
    from feynkit.io.report import LATTICE_BUDGET, NORMALIZ_TIMEOUT

    calls: list[dict[str, object]] = []
    original = module.lattice_invariants

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "lattice_invariants", counting)
    text = tmp_path / "report.txt"
    _run(capsys, tmp_path, "12e|2e|e|:zzz", "-n", "--text", str(text), "--sections", "polytope")
    assert "Normal configuration" in text.read_text(encoding="utf-8")
    assert len(calls) == 1
    assert calls[0]["budget"] == LATTICE_BUDGET == 2 * 10**7
    assert calls[0]["timeout"] == NORMALIZ_TIMEOUT


def test_every_key_fits_its_column() -> None:
    # _kv pads keys to 28 characters; a longer key pushes its value out of line.
    # Every call must pass a literal key, so that each one is checked here.
    source = Path(cli.__file__).read_text(encoding="utf-8")
    keys = re.findall(r'_kv\(\s*"([^"]*)"', source)
    calls = source.count("_kv(") - source.count("def _kv(")
    assert calls > 30
    assert len(keys) == calls
    assert [key for key in keys if len(key) > 28] == []


class TestPairwise:
    def test_equivalent_triangles_reported(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "12e|2e|e|:nzz", "12e|2e|e|:znz")
        assert "Equivalence checks" in out
        assert "unimodular" in out

    def test_identity_printed_with_permutation_and_factor(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # Swapping u_1 and u_2 sends the columns of A onto those of B.
        out = _run(capsys, tmp_path, "12e|2e|e|:nzz", "12e|2e|e|:znz")
        identity = "GKZ identity I_A(beta, z_P) = |det M| I_B(T beta, z)"
        # Printed for point_config and finite_index, not for the hull-only relations.
        assert out.count(identity) == 2
        assert out.count("Relates the Newton polytopes only") == 2
        assert "point_config gives the identity" in out
        assert "P       =  [3, 1, 4, 2, 6, 5, 7]" in out
        assert "z_P     =  (z_3, z_1, z_4, z_2, z_6, z_5, z_7)" in out
        assert "|det M| =  1" in out
        assert "T beta  =  [-D/2, -nu_2, -nu_1, -nu_3]" in out
        assert "u_1  =  v_2" in out
        assert "sum_j M_ij" not in out

    def test_tadpoles_are_compared_by_their_columns(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # A segment in R^1 is compared as a polytope too.
        out = _run(capsys, tmp_path, "0|:n", "0|:n")
        assert "unimodular             YES\n" in out
        assert "affine_polytope        YES  (det = 1)" in out
        assert "point_config           YES  (det = 1)" in out
        assert "T beta  =  [-D/2, -nu_1]" in out

    @pytest.mark.parametrize("cnickel", ["011e|e|:zzz", "012e|2e|e|:znnn"])
    def test_hull_checks_below_full_dimension(
        self, cnickel: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # Below full dimension the unimodular check found a diagram not equivalent to itself.
        out = _run(capsys, tmp_path, cnickel, cnickel)
        assert "unimodular             YES\n" in out
        assert "affine_polytope        YES  (det = 1)" in out
        assert "point_config           YES  (det = 1)" in out
        assert (
            "  The Newton polytopes are not full-dimensional: the identity holds\n"
            "  only trivially.\n"
        ) in out

    def test_finite_index_below_full_dimension(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "01e|e|:zn", "00|:nz")
        # Both witnesses extend a map between the lattice charts of the two segments; the
        # finite_index search printed n/a here.
        assert "point_config           YES  (det = 1)" in out
        assert "finite_index           YES  (det = 1)" in out
        assert "Witness map  [finite_index]" in out
        assert out.count("the identity holds\n  only trivially.") == 2
