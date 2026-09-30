"""fk analyse --kinematics: the class imposed, its errors and the header row."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feynkit import FeynkitDatabase, FeynmanIntegral
from feynkit.cli import main

BOX = "12e|3e|3e|e|"
CHOICES = "generic, massless_off_shell, massless_on_shell, equal_masses"


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return excinfo.value.code


def test_the_header_names_the_class(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", BOX + ":zzzz", "-n", "--no-db"])
    out = capsys.readouterr().out
    assert f"  {'External legs':<28} 4\n  {'Kinematic class':<28} massless_off_shell\n" in out


def test_on_shell_box(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", BOX + ":zzzz", "--kinematics", "massless_on_shell", "-n", "--no-db"])
    out = capsys.readouterr().out
    assert f"  {'Kinematic class':<28} massless_on_shell\n" in out
    assert f"  {'Monomials (A-columns)':<28} 6\n" in out
    assert "Normalised volume            3  (the holonomic rank" in out


def test_the_bare_form_reads_the_value(capsys: pytest.CaptureFixture[str]) -> None:
    main([BOX + ":zzzz", "--kinematics", "massless_on_shell", "-n", "--no-db"])
    assert f"  {'Kinematic class':<28} massless_on_shell\n" in capsys.readouterr().out


def test_json_and_the_database(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = tmp_path / "x.db"
    main(["analyse", BOX + ":nnnn", "--kinematics", "equal_masses", "--json", "--db", str(db)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["input"] == BOX + ":nnnn"
    assert payload["cnickel"] == BOX + ":aaaa"
    assert payload["summary"]["kinematic_class"] == "equal_masses"
    with FeynkitDatabase(db) as database:
        (record,) = database.all_integrals(kinematic_class="equal_masses")
        assert record.kinematics[0].internal_axis == "equal"


def test_json_counts_the_limit_surfaces_on_shell(capsys: pytest.CaptureFixture[str]) -> None:
    # The legs of the on-shell box specialise the generic ones, whose surfaces are restricted.
    argv = ["analyse", BOX + ":zzzz", "--kinematics", "massless_on_shell", "--json", "--no-db"]
    main([*argv, "--sections", "landau", "--limits"])
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert (summary["landau_surfaces"], summary["limit_surfaces"]) == (3, 0)
    assert summary["limit_candidates"] == summary["parent_skipped_faces"] == 0
    main(["analyse", BOX + ":zzzz", "--json", "--no-db", "--sections", "landau"])
    assert "limit_surfaces" not in json.loads(capsys.readouterr().out)["summary"]


@pytest.mark.parametrize("name", ["on_shell", "other"])
def test_an_unknown_class_is_a_usage_error(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(["analyse", BOX + ":zzzz", "--kinematics", name, "--no-db"]) == 2
    err = capsys.readouterr().err
    assert f"unknown kinematic class {name!r}; choose from {CHOICES}" in err


def test_a_refused_class_exits_1_in_one_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = tmp_path / "x.db"
    argv = ["analyse", BOX + ":nnnn", "--kinematics", "massless_on_shell", "--db", str(db)]
    assert _exit_code(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(
        f"fk: error: CNickel '{BOX}:nnnn': cannot impose massless_on_shell: propagators "
        "1, 2, 3, 4 are massive"
    )
    assert not db.exists()


def test_a_tadpole_cannot_be_put_on_shell(capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(["analyse", "0|:z", "--kinematics", "massless_on_shell", "--no-db"]) == 1
    assert "fewer than two external legs" in capsys.readouterr().err


def test_the_class_of_the_string_is_accepted(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "12e|2e|e|:nnn", "--kinematics", "generic", "--json", "--no-db"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["cnickel"] == FeynmanIntegral.from_cnickel("12e|2e|e|:nnn").cnickel
    assert payload["summary"]["kinematic_class"] == "generic"


def test_limits_need_a_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(["analyse", BOX + ":zzzz", "--limits", "-n", "--no-db"]) == 2
    assert "--limits" in capsys.readouterr().err
