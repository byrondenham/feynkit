"""Tests for --sectors of fk analyse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feynkit.cli import main
from feynkit.landau import _singular_binary

BOX = ["analyse", "13e|2e|3e|e|:zzzz", "--kinematics", "massless_on_shell", "--no-db"]
requires_singular = pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed")


def test_sectors_prints_the_table_and_nothing_else(capsys: pytest.CaptureFixture[str]) -> None:
    main([*BOX, "--sectors"])
    out = capsys.readouterr().out
    assert "Sector hierarchy" in out
    assert "7 sectors of 16 are non-zero, 8 are scaleless and 1 is cut by a cycle" in out
    assert "N_id" in out and "t_gen" in out and "Resonance" in out
    assert "t(T)" not in out
    assert "Symanzik" not in out


@requires_singular
def test_the_counts_at_a_point_are_opt_in(capsys: pytest.CaptureFixture[str]) -> None:
    main([*BOX, "--sectors", "--sectors-counts", "critical", "--seed", "3"])
    out = capsys.readouterr().out
    assert "t(T)" in out and "m(T)" in out and "N_T" in out


def test_the_counts_need_the_sectors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main([*BOX, "-g", "--sectors-counts", "critical"])
    assert caught.value.code == 2
    assert "--sectors-counts" in capsys.readouterr().err


def test_the_reports_gain_the_section(tmp_path: Path) -> None:
    text = tmp_path / "box.txt"
    main([*BOX, "--sectors", "--text", str(text), "--sections", "gkz"])
    assert "Sector hierarchy" in text.read_text(encoding="utf-8")


def test_the_json_summary_holds_the_sector_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    main([*BOX, "--json", "--sections", "sectors"])
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["nonzero_sectors"] == 7 and summary["unique_sectors"] == 5


def test_a_large_hierarchy_is_left_out(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("feynkit.io.sections.sectors.MAX_PROPAGATORS", 3)
    main([*BOX, "--sectors"])
    assert "more than the 3" in capsys.readouterr().out
