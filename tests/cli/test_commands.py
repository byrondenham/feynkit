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
        (["11e|e|:zz", "b", "c"], ["analyse", "11e|e|:zz", "b", "c"]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["analyse", "11e|e|:zz"], ["analyse", "11e|e|:zz"]),
        (["analyze", "12e|2e|e|", "--no-db"], ["analyze", "12e|2e|e|", "--no-db"]),
        (["anlyse", "11e|e|:zz"], ["anlyse", "11e|e|:zz"]),
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
    with pytest.raises(SystemExit) as excinfo:
        main(["compare", "12e|2e|e|:zzz", "11e|e|:zz", "--db", str(tmp_path / "x.db")])
    assert excinfo.value.code == 3
    assert "Ambient dimension mismatch (3 vs 2)." in capsys.readouterr().out


def test_bare_single_form_runs_analyse(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["11e|e|:zz", "--db", str(tmp_path / "x.db"), "-g"])
    out = capsys.readouterr().out
    assert "Euler equations" in out
    assert "Equivalence analysis" not in out


def test_bare_pair_form_runs_compare(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["12e|2e|e|:zzz", "11e|e|:zz", "--db", str(tmp_path / "x.db")])
    assert excinfo.value.code == 3
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


@pytest.mark.parametrize(
    ("argv", "command", "extra"),
    [
        (["analyse", "12e|2e|e|", "--bogus", "--no-db"], "analyse", "--bogus"),
        (["12e|2e|e|", "--bogus", "--no-db"], "analyse", "--bogus"),
        (["analyse", "12e|2e|e|", "11e|e|", "--no-db"], "analyse", "11e|e|"),
        (["analyze", "12e|2e|e|", "-x", "--no-db"], "analyse", "-x"),
        (["compare", "12e|2e|e|", "11e|e|", "--bogus", "--no-db"], "compare", "--bogus"),
        (["12e|2e|e|", "11e|e|", "-s", "--no-db"], "compare", "-s"),
        (["--bogus", "analyse", "12e|2e|e|", "--no-db"], "analyse", "--bogus"),
        (["analyse", "12e|2e|e|", "--", "--no-db"], "analyse", "--no-db"),
    ],
    ids=[
        "analyse",
        "bare-analyse",
        "surplus",
        "analyze",
        "compare",
        "bare-compare",
        "before-command",
        "after-double-dash",
    ],
)
def test_unknown_arguments_get_the_usage_of_their_subcommand(
    argv: list[str], command: str, extra: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith(f"usage: fk {command} ")
    assert lines[-1] == f"fk {command}: error: unrecognized arguments: {extra}"


@pytest.mark.parametrize(
    ("argv", "command", "options"),
    [
        (["--no-db", "analyse", "12e|2e|e|"], "analyse", "--no-db"),
        (["-v", "compare", "12e|2e|e|", "11e|e|"], "compare", "-v"),
        (["--db=x.db", "analyze", "12e|2e|e|"], "analyse", "--db=x.db"),
        (["--no-db", "-v", "analyse", "12e|2e|e|"], "analyse", "--no-db -v"),
        (["--db", "x.db", "analyse", "12e|2e|e|"], "analyse", "--db x.db"),
        (["--latex", "r.tex", "-v", "analyze", "12e|2e|e|"], "analyse", "--latex r.tex -v"),
    ],
    ids=["analyse", "compare", "analyze", "two", "with-value", "value-and-flag"],
)
def test_option_of_the_command_given_before_it_must_follow_it(
    argv: list[str], command: str, options: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith(f"usage: fk {command} ")
    assert lines[-1] == (
        f"fk {command}: error: {options} must follow the command: fk {command} {options} ..."
    )


def test_option_of_the_other_command_before_it_is_not_moved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # compare has no --text, so fk's parser takes r.txt for the command.
    with pytest.raises(SystemExit) as excinfo:
        main(["--text", "r.txt", "compare", "12e|2e|e|", "11e|e|"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: fk [-h]")
    assert "invalid choice: 'r.txt'" in err
    assert "must follow" not in err


def test_analyze_runs_analyse(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyze", "12e|2e|e|", "-g", "-n", "--no-db"])
    out = capsys.readouterr().out
    assert "Euler equations" in out
    assert "Newton polytope" in out
    assert "Equivalence analysis" not in out


@pytest.mark.parametrize("argv", [["anlyse", "11e|e|:zz", "--no-db"], ["abc", "--no-db"]])
def test_first_word_without_a_bar_is_a_command(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: fk ")
    assert f"invalid choice: '{argv[0]}'" in err
    assert "CNickel" not in err


@pytest.mark.parametrize(
    ("argv", "suggestion"),
    [
        (["0:n", "--no-db"], 'fk analyse "0:n"'),
        (["0:n", "00:nn", "--no-db"], 'fk compare "0:n" "00:nn"'),
    ],
    ids=["analyse", "compare"],
)
def test_first_word_like_a_cnickel_string_without_a_bar_gets_a_hint(
    argv: list[str], suggestion: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The parser reads 0:n as the tadpole 0|:n, but the bare form needs a |.
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].startswith("usage: fk ")
    assert "invalid choice: '0:n'" in lines[-2]
    assert lines[-1] == f"a CNickel string without a | needs the command: {suggestion}"


def test_cnickel_hint_follows_only_an_invalid_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version=1", "0:n"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert err.endswith("fk: error: argument --version: ignored explicit argument '1'\n")
    assert "CNickel" not in err


def test_main_help_shows_both_bare_forms(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert 'fk "12e|2e|e|:zzz"                           the bare form of fk analyse' in out
    assert 'fk "12e|2e|e|:nzz" "12e|2e|e|:znz"           the bare form of fk compare' in out


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


def test_help_descriptions_fit_in_80_columns() -> None:
    # The descriptions are printed as written, so each line must fit a terminal.
    for parser in _build_parser():
        assert parser.description is not None
        assert max(len(line) for line in parser.description.splitlines()) < 80


def test_resonance_prints_each_facet(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:nn", "-r", "--no-db"])
    out = capsys.readouterr().out
    assert "Resonance" in out and "D = 4 - 2 epsilon" in out
    assert "Euler equations" not in out
    # F_U of the bubble: resonant on 1 + Z/2, admissible at eps = 1.
    assert "x_1 + x_2 <= 2" in out
    assert (
        "resonant: 1 + Z/2 (at epsilon = 0: yes); admissible: epsilon = 1; reducible: yes"
        in " ".join(out.split())
    )


def test_d0_sets_the_dimension(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    text = tmp_path / "r.txt"
    main(["analyse", "11e|e|:nn", "-r", "--d0", "7/2", "--text", str(text), "--no-db"])
    assert "D = 7/2 - 2 epsilon" in capsys.readouterr().out
    assert "D_0 = 7/2" in " ".join(text.read_text().split())


@pytest.mark.parametrize("value", ["four", "4.5.1", "1/0"])
def test_d0_must_be_a_number(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["analyse", "11e|e|:nn", "-r", "--d0", value, "--no-db"])
    assert excinfo.value.code == 2
    assert "--d0" in capsys.readouterr().err


def test_d0_needs_the_resonance_section(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["analyse", "11e|e|:nn", "-g", "--d0", "3", "--no-db"])
    assert excinfo.value.code == 2
    assert "--d0" in capsys.readouterr().err


def test_faces_prints_each_facet(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "12e|22e|e|:nnnn", "-f", "--no-db"])
    out = capsys.readouterr().out
    assert "Faces as graphs  (up to codimension 2)" in out
    assert "U({3,4}) G(Gamma/{3,4})" in out
    assert "Euler equations" not in out


def test_faces_are_printed_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:nn", "--no-db"])
    assert "Faces as graphs  (up to codimension 2)" in capsys.readouterr().out


def test_faces_name_the_unidentified(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "12e|3e|3e|e|:zzzz", "--kinematics", "massless_on_shell", "-f", "--no-db"])
    out = capsys.readouterr().out
    assert "unidentified; predicted G({3,4}) U(Gamma/{3,4})" in out


def test_faces_add_the_section_to_the_report(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # The default sections hold the faces; -f adds them to sections chosen without them.
    text = tmp_path / "r.txt"
    main(["analyse", "11e|e|:nz", "-f", "--text", str(text), "--sections", "gkz", "--no-db"])
    capsys.readouterr()
    report = text.read_text()
    assert "Faces as graphs" in report and "G({1}) U(Gamma/{1})" in report
    assert "\nResonance\n---" not in report


def test_faces_name_a_support_product(capsys: pytest.CaptureFixture[str]) -> None:
    import sympy as sp

    from feynkit import FeynmanIntegral
    from feynkit.cli import _print_faces

    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
    p2 = sp.Symbol("p2^2", real=True)
    fi = fi.with_(momentum_products={k: v.subs(p2, 0) for k, v in fi.momentum_products.items()})
    _print_faces(fi)
    out = capsys.readouterr().out
    assert "support product of G({2,4}) U(Gamma/{2,4})" in out
    assert "support product" in out.split("Class")[1]
