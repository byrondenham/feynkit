"""Tests for the ``fk`` command-line entry point."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

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
        assert "Not computed for a Newton polytope of dimension below 2" in out
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
        assert f"  Normalised volume            {volume}" in lines
        assert not any("Lattice base point" in line for line in lines)

    def test_newton_section_of_a_full_dimensional_polytope(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "12e|2e|e|:zzz", "-n")
        assert "Normalised volume            4  (the holonomic rank for generic beta)" in out
        assert "Lattice base point           (1, 1, 0)" in out


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
        # The segments of two tadpoles are too small for the checks of the polytopes.
        out = _run(capsys, tmp_path, "0|:n", "0|:n")
        assert "unimodular             n/a  (a Newton polytope of dimension below 2)" in out
        assert "affine_polytope        n/a  (a Newton polytope of dimension below 2)" in out
        assert "point_config           YES  (det = 1)" in out
        assert "T beta  =  [-D/2, -nu_1]" in out

    @pytest.mark.parametrize("cnickel", ["011e|e|:zzz", "012e|2e|e|:znnn"])
    def test_hull_checks_are_skipped_below_full_dimension(
        self, cnickel: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # Below full dimension the unimodular check failed even for a diagram and itself.
        out = _run(capsys, tmp_path, cnickel, cnickel)
        skipped = "n/a  (a Newton polytope that is not full-dimensional)"
        assert f"unimodular             {skipped}" in out
        assert f"affine_polytope        {skipped}" in out
        assert "unimodular             no" not in out
        assert "point_config           YES  (det = 1)" in out

    def test_finite_index_is_skipped_when_a_is_not_full_dimensional(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "01e|e|:zn", "00|:nz")
        assert "point_config           YES  (det = -1)" in out
        assert (
            "finite_index           n/a  (the Newton polytope of A is not full-dimensional)" in out
        )
