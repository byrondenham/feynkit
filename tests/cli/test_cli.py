"""Tests for the ``fk`` command-line entry point."""

from __future__ import annotations

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


class TestPairwise:
    def test_equivalent_triangles_reported(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        out = _run(capsys, tmp_path, "12e|2e|e|:nzz", "12e|2e|e|:znz")
        assert "Equivalence checks" in out
        assert "unimodular" in out
