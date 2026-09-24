"""Tests for the fk subcommands, the bare form, --version and the help text."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import feynkit
import feynkit.cli
from feynkit.cli import _VALUE_OPTIONS, _build_parser, _with_command, main

_UNQUOTED_CNICKEL = re.compile(r"[0-9e]\|")


def _unquoted_cnickel_lines(text: str) -> list[str]:
    """The lines of text with a CNickel string outside double quotes."""
    return [
        line for line in text.splitlines() if _UNQUOTED_CNICKEL.search(re.sub(r'"[^"]*"', "", line))
    ]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["11e|e|:zz"], ["analyse", "11e|e|:zz"]),
        (["11e|e|:zz", "-g", "-n"], ["analyse", "11e|e|:zz", "-g", "-n"]),
        (["11e|e|:zz", "--db", "x.db"], ["analyse", "11e|e|:zz", "--db", "x.db"]),
        (["--db", "x.db", "11e|e|:zz"], ["analyse", "--db", "x.db", "11e|e|:zz"]),
        (["11e|e|:zz", "12e|2e|e|:zzz"], ["compare", "11e|e|:zz", "12e|2e|e|:zzz"]),
        (
            ["11e|e|:zz", "--db=x.db", "12e|2e|e|:zzz"],
            ["compare", "11e|e|:zz", "--db=x.db", "12e|2e|e|:zzz"],
        ),
        (["--", "11e|e|:zz"], ["analyse", "--", "11e|e|:zz"]),
        (["a", "b", "c"], ["analyse", "a", "b", "c"]),
        (["analyse", "11e|e|:zz"], ["analyse", "11e|e|:zz"]),
        (["compare", "a", "b"], ["compare", "a", "b"]),
        ([], []),
        (["--help"], ["--help"]),
        (["--version"], ["--version"]),
    ],
)
def test_bare_form_gets_its_subcommand(argv: list[str], expected: list[str]) -> None:
    assert _with_command(argv) == expected


def test_value_options_match_the_parsers() -> None:
    parsers = _build_parser()
    taking = {
        option
        for parser in (parsers.analyse, parsers.compare)
        for action in parser._actions
        if action.nargs != 0
        for option in action.option_strings
    }
    assert taking == _VALUE_OPTIONS


def test_analyse_prints_the_chosen_sections(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(["analyse", "11e|e|:zz", "-g", "--db", str(tmp_path / "x.db")])
    out = capsys.readouterr().out
    assert "Euler equations" in out
    assert "Symanzik polynomials" not in out


def test_short_section_flags_cluster(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["analyse", "11e|e|:zz", "-gn", "--db", str(tmp_path / "x.db")])
    out = capsys.readouterr().out
    assert "Euler equations" in out
    assert "Newton polytope" in out


def test_compare_runs_the_equivalence_checks(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(["compare", "12e|2e|e|:zzz", "11e|e|:zz", "--db", str(tmp_path / "x.db")])
    assert "Ambient dimension mismatch (3 vs 2)." in capsys.readouterr().out


def test_bare_single_form_runs_analyse(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["11e|e|:zz", "--db", str(tmp_path / "x.db"), "-g"])
    out = capsys.readouterr().out
    assert "Euler equations" in out
    assert "Equivalence analysis" not in out


def test_bare_pair_form_runs_compare(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["12e|2e|e|:zzz", "11e|e|:zz", "--db", str(tmp_path / "x.db")])
    assert "Equivalence analysis" in capsys.readouterr().out


def test_default_database_is_feynkit_db_in_the_working_directory(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(["analyse", "11e|e|:zz", "-t"])
    assert (tmp_path / "feynkit.db").exists()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["analyse"],
        ["compare", "11e|e|:zz"],
        ["11e|e|:zz", "12e|2e|e|:zzz", "-s"],
        ["11e|e|:zz", "12e|2e|e|:zzz", "11e|e|:nz"],
        ["analyse", "11e|e|:zz", "--sym"],
    ],
    ids=["no-command", "no-cnickel", "one-of-two", "flags-with-two", "three", "abbreviation"],
)
def test_usage_errors_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    assert "usage: fk" in capsys.readouterr().err


def test_version_prints_fk_and_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == f"fk {feynkit.__version__}\n"


@pytest.mark.parametrize("argv", [["--help"], ["analyse", "--help"], ["compare", "--help"]])
def test_help_quotes_every_cnickel_string(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "12e|2e|e|" in out
    assert _unquoted_cnickel_lines(out) == []


def test_module_docstring_quotes_every_cnickel_string() -> None:
    assert feynkit.cli.__doc__ is not None
    assert _unquoted_cnickel_lines(feynkit.cli.__doc__) == []


def test_analyse_help_describes_each_section_flag_once(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["analyse", "--help"])
    out = capsys.readouterr().out
    assert out.count("Symanzik polynomials U, F, G") == 1
    assert out.count("--symanzik") == 1
