"""fk analyse --set: kinematic substitutions, their errors, the headers and the JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feynkit.cli import main

BOX = "12e|3e|3e|e|:zzzz"
SYMBOLS = "p1^2, p2^2, p3^2, p4^2, s12, s23"


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return excinfo.value.code


def _error(argv: list[str], capsys: pytest.CaptureFixture[str], code: int = 1) -> str:
    assert _exit_code(argv) == code
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err


def test_three_mass_box_shows_its_support_product_faces(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["analyse", BOX, "--set", "p4^2=0", "--json", "--no-db"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["substitutions"] == ["p4^2 = 0"]
    assert payload["summary"]["support_product_faces"] == 4
    main(["analyse", BOX, "--json", "--no-db"])
    payload = json.loads(capsys.readouterr().out)
    assert "substitutions" not in payload
    assert payload["summary"]["support_product_faces"] == 0


def test_the_terminal_header_states_the_substitutions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["analyse", BOX, "--set", "p4^2=0", "--set", "p3^2 = p1^2/2 - s12", "-n", "--no-db"])
    out = capsys.readouterr().out
    assert f"  {'Substitutions':<28} p4^2 = 0, p3^2 = p1^2/2 - s12\n" in out


def test_the_reports_state_the_substitutions(tmp_path: Path) -> None:
    tex, txt = tmp_path / "box.tex", tmp_path / "box.txt"
    argv = ["analyse", BOX, "--set", "p4^2=0", "-n", "--no-db", "--sections", "gkz"]
    main(argv + ["--latex", str(tex), "--text", str(txt)])
    assert "\\date{With \\texttt{p4\\textasciicircum{}2 = 0}.}" in tex.read_text(encoding="utf-8")
    assert "\nWith p4^2 = 0.\n" in txt.read_text(encoding="utf-8")


def test_values_are_exact_rationals_or_linear_combinations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["analyse", BOX, "--set", "p1^2=-3/2", "--set", "s12 = 2*p2^2 + (p3^2 - 1)/3", "-s"])
    out = capsys.readouterr().out
    assert "-3/2" in out and "0.5" not in out
    assert "s12" not in out.split("Symanzik polynomials", 1)[1]


def test_it_applies_after_kinematics(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["analyse", BOX, "--kinematics", "massless_on_shell", "--set", "s12=s23", "--no-db"]
    main(argv + ["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["substitutions"] == ["s12 = s23"]
    assert payload["summary"]["kinematic_class"] == "other"
    err = _error(argv[:4] + ["--set", "p1^2=0", "--no-db"], capsys)
    assert "--set 'p1^2=0': unknown kinematic symbol 'p1^2'" in err


def test_an_unknown_symbol_names_the_symbols(capsys: pytest.CaptureFixture[str]) -> None:
    err = _error(["analyse", BOX, "--set", "m^2=0", "--no-db"], capsys)
    assert err == (
        f"fk: error: --set 'm^2=0': unknown kinematic symbol 'm^2'; the symbols of "
        f"CNickel '{BOX}' are {SYMBOLS}\n"
    )


def test_an_unknown_symbol_in_the_value(capsys: pytest.CaptureFixture[str]) -> None:
    err = _error(["analyse", BOX, "--set", "s12=q+1", "--no-db"], capsys)
    assert "--set 's12=q+1': unknown kinematic symbol 'q'; the symbols" in err


@pytest.mark.parametrize("value", ["s23*p1^2", "s23/p1^2", "s23^2", "s23**2"])
def test_a_non_linear_value_is_refused(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    err = _error(["analyse", BOX, "--set", f"s12={value}", "--no-db"], capsys)
    assert err.startswith(f"fk: error: --set 's12={value}': not linear")
    assert len(err.splitlines()) == 1


@pytest.mark.parametrize("value", ["0.5", "1e3", "sqrt(2)"])
def test_a_value_that_is_not_exact_is_refused(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    err = _error(["analyse", BOX, "--set", f"p1^2={value}", "--no-db"], capsys)
    assert f"--set 'p1^2={value}': cannot read" in err


def test_a_symbol_cannot_be_set_twice_or_to_itself(capsys: pytest.CaptureFixture[str]) -> None:
    err = _error(["analyse", BOX, "--set", "p1^2=0", "--set", "p1^2=1", "--no-db"], capsys)
    assert "--set 'p1^2=1': unknown kinematic symbol 'p1^2'" in err
    err = _error(["analyse", BOX, "--set", "p1^2=p1^2+s12", "--no-db"], capsys)
    assert "--set 'p1^2=p1^2+s12': the value contains p1^2 itself" in err


@pytest.mark.parametrize("text", ["p1^2", "=0", "p1^2=", ""])
def test_a_malformed_assignment_is_a_usage_error(
    text: str, capsys: pytest.CaptureFixture[str]
) -> None:
    err = _error(["analyse", BOX, "--set", text, "--no-db"], capsys, code=2)
    assert "expected SYMBOL=VALUE, such as p4^2=0" in err
