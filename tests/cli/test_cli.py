"""Tests for the ``fk`` command-line entry point."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from feynkit.cli import main


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
