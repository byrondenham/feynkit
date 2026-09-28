"""Tests for the report files and the JSON summary of fk analyse."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from feynkit import FeynkitDatabase, FeynmanIntegral
from feynkit.cli import main
from feynkit.io.report import DEFAULT_SECTIONS


def test_latex_and_text_reports_match_the_facade(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tex, txt = tmp_path / "bubble.tex", tmp_path / "bubble.txt"
    main(
        ["analyse", "11e|e|:zz", "-s", "--latex", str(tex), "--text", str(txt)]
        + ["--sections", "polytope,gkz", "--no-db"]
    )
    fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
    assert tex.read_text(encoding="utf-8") == fi.to_latex(["gkz", "polytope"])
    assert txt.read_text(encoding="utf-8") == fi.to_text(["gkz", "polytope"])
    out = capsys.readouterr().out
    assert "Symanzik polynomials" in out
    assert f"  Wrote the LaTeX report to {tex}\n" in out
    assert f"  Wrote the text report to {txt}\n" in out


def test_text_report_has_every_section_by_default(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    txt = tmp_path / "bubble.txt"
    main(["analyse", "11e|e|:zz", "-s", "--text", str(txt), "--no-db"])
    expected = FeynmanIntegral.from_cnickel("11e|e|:zz").to_text()
    assert txt.read_text(encoding="utf-8") == expected


def test_json_summary_of_the_chosen_sections(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|", "--json", "--sections", "gkz", "--no-db"])
    assert json.loads(capsys.readouterr().out) == {
        "input": "11e|e|",
        "cnickel": "11e|e|:zz",
        "sections": ["gkz"],
        "summary": {
            "loops": 1,
            "propagators": 2,
            "external_legs": 2,
            "monomials_of_f": 1,
            "monomials_of_g": 3,
            "independent_invariants": 1,
            "codimension": 0,
            "toric_generators": 0,
        },
    }


def test_json_summary_of_the_default_sections(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:zz", "--json", "--no-db"])
    data = json.loads(capsys.readouterr().out)
    assert data["sections"] == list(DEFAULT_SECTIONS)
    assert "candidate_master_count" not in data["summary"]
    assert data["summary"]["normalised_volume"] == 1
    assert data["summary"]["polytope_automorphisms"] == 6
    assert data["summary"]["landau_surfaces"] == 1


def test_json_summary_with_the_point_counts(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:nn", "--json", "--sections", "torus", "--no-db"])
    data = json.loads(capsys.readouterr().out)
    assert data["sections"] == ["torus"]
    assert data["summary"]["candidate_master_count"] == 3


def test_sections_help_names_the_default(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["analyse", "--help"])
    assert "all but torus by default" in " ".join(capsys.readouterr().out.split())


def test_json_with_a_report_file_keeps_stdout_pure_json(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    tex = tmp_path / "bubble.tex"
    main(["analyse", "11e|e|:zz", "--json", "--latex", str(tex), "--sections", "gkz", "--no-db"])
    json.loads(capsys.readouterr().out)
    assert tex.exists()


def test_json_stores_the_integral(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    main(["analyse", "11e|e|:zz", "--json", "--sections", "gkz", "--db", str(db)])
    json.loads(capsys.readouterr().out)
    with FeynkitDatabase(db) as store:
        assert store.lookup(FeynmanIntegral.from_cnickel("11e|e|:zz")) is not None


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--json", "--sections", "gkz,nope"],
            "unknown section nope; choose from identity, conventions",
        ),
        (["--json", "--sections", ","], "no section given"),
        (["--json", "-s"], "--json prints only the summary"),
        (["--sections", "gkz"], "--sections chooses report sections"),
    ],
    ids=["unknown-section", "empty-sections", "json-with-flags", "sections-alone"],
)
def test_report_usage_errors_exit_2(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["analyse", "11e|e|:zz", "--no-db", *argv])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "kind", "where", "reason"),
    [
        ("--latex", "LaTeX", "missing/bubble.tex", "No such file or directory"),
        ("--text", "text", "missing/bubble.txt", "No such file or directory"),
        ("--latex", "LaTeX", ".", "Is a directory"),
        ("--text", "text", "notes.txt/bubble.txt", "Not a directory"),
    ],
    ids=["missing-directory", "missing-directory-text", "directory", "file-as-directory"],
)
def test_report_that_cannot_be_written_exits_1_before_the_analysis(
    option: str,
    kind: str,
    where: str,
    reason: str,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("notes\n", encoding="utf-8")
    target = tmp_path / where
    with pytest.raises(SystemExit) as excinfo:
        main(["analyse", "11e|e|:zz", "-s", option, str(target), "--no-db"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"fk: error: cannot write the {kind} report to {target}: {reason}\n"


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs file permissions to apply"
)
def test_report_in_a_read_only_directory_exits_1_before_the_analysis(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(SystemExit) as excinfo:
            main(["analyse", "11e|e|:zz", "-s", "--text", str(locked / "b.txt"), "--no-db"])
    finally:
        locked.chmod(0o700)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.endswith(": Permission denied\n")


def test_bare_form_with_a_report_file_runs_analyse(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    txt = tmp_path / "bubble.txt"
    main(["11e|e|:zz", "--text", str(txt), "--sections", "gkz", "--no-db"])
    assert txt.exists()
