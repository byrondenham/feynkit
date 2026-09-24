"""Tests for the errors fk reports in one line, its exit statuses and the database it closes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import feynkit.cli as cli
from feynkit import FeynkitDatabase, FeynmanIntegral, ValidationError
from feynkit.cli import main

EXAMPLE = 'fk analyse "12e|2e|e|:nzz"'


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return excinfo.value.code


def _count_closes(monkeypatch: pytest.MonkeyPatch) -> list[FeynkitDatabase]:
    """Record every FeynkitDatabase.close call."""
    closed: list[FeynkitDatabase] = []
    close = FeynkitDatabase.close

    def recording_close(self: FeynkitDatabase) -> None:
        closed.append(self)
        close(self)

    monkeypatch.setattr(FeynkitDatabase, "close", recording_close)
    return closed


@pytest.mark.parametrize("bad", ["12e|2e|e|:zz", "abc", "9e|e|", ""])
def test_cnickel_that_does_not_parse_exits_1_with_the_grammar(
    bad: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = tmp_path / "x.db"
    assert _exit_code(["analyse", bad, "--db", str(db)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"fk: error: cannot parse CNickel {bad!r}: ")
    assert "TOPOLOGY:COLOURS" in lines[0]
    assert lines[0].endswith(EXAMPLE)
    assert not db.exists()


def test_compare_names_the_string_that_does_not_parse(capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(["compare", "11e|e|:zz", "abc", "--no-db"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("fk: error: cannot parse CNickel 'abc': ")


def test_bare_form_that_does_not_parse_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(["abc", "--no-db"]) == 1
    assert capsys.readouterr().err.startswith("fk: error: cannot parse CNickel 'abc': ")


def test_library_error_exits_1_without_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    # "e|e|" parses, but a graph without propagators has no Symanzik polynomials.
    assert _exit_code(["analyse", "e|e|", "-s", "--no-db"]) == 1
    assert capsys.readouterr().err == "fk: error: Graph has no internal edges.\n"


def test_validation_error_exits_1_in_one_line_and_closes_the_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    closed = _count_closes(monkeypatch)

    def fail(fi: FeynmanIntegral) -> None:
        raise ValidationError("nu_1 must be\npositive")

    monkeypatch.setitem(cli._PRINTERS, "symanzik", fail)
    assert _exit_code(["analyse", "11e|e|:zz", "-s", "--db", str(tmp_path / "x.db")]) == 1
    assert capsys.readouterr().err == "fk: error: nu_1 must be positive\n"
    assert len(closed) == 1


def test_a_bug_keeps_its_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(fi: FeynmanIntegral) -> None:
        raise RuntimeError("bug")

    monkeypatch.setitem(cli._PRINTERS, "symanzik", broken)
    with pytest.raises(RuntimeError, match="bug"):
        main(["analyse", "11e|e|:zz", "-s", "--no-db"])


def test_file_that_is_not_a_database_exits_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    bad = tmp_path / "notes.db"
    bad.write_bytes(b"not a database\n" * 64)
    assert _exit_code(["analyse", "11e|e|:zz", "-s", "--db", str(bad)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"fk: error: database {bad}: ")
    assert captured.err.count("\n") == 1


def test_database_in_a_missing_directory_exits_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "missing" / "x.db"
    assert _exit_code(["compare", "11e|e|:zz", "11e|e|:nz", "--db", str(target)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"fk: error: database {target}: ")
    assert captured.err.count("\n") == 1


def test_no_db_writes_no_database(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["analyse", "11e|e|:zz", "-t", "--no-db"])
    assert "Database" not in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_compare_with_no_db_writes_no_database(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(["compare", "12e|2e|e|:zzz", "11e|e|:zz", "--no-db"])
    assert list(tmp_path.iterdir()) == []


def test_db_and_no_db_together_exit_2(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert _exit_code(["analyse", "11e|e|:zz", "--db", str(tmp_path / "x.db"), "--no-db"]) == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_database_is_closed_after_analyse(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    closed = _count_closes(monkeypatch)
    main(["analyse", "11e|e|:zz", "-s", "--db", str(tmp_path / "x.db")])
    assert len(closed) == 1


def test_database_is_closed_when_compare_stops_early(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    closed = _count_closes(monkeypatch)
    main(["compare", "12e|2e|e|:zzz", "11e|e|:zz", "--db", str(tmp_path / "x.db")])
    assert "Ambient dimension mismatch" in capsys.readouterr().out
    assert len(closed) == 1


@pytest.mark.parametrize(
    ("argv", "status"),
    [(["--version"], 0), (["analyse", "abc", "--no-db"], 1), ([], 2)],
    ids=["version", "bad-cnickel", "no-command"],
)
def test_exit_status_seen_by_the_shell(argv: list[str], status: int, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "feynkit.cli", *argv],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == status
    assert "Traceback" not in result.stderr
