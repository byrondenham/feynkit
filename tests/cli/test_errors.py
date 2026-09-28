"""Tests for the errors fk reports in one line, its exit statuses and the database it closes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import sympy as sp

import feynkit.cli as cli
from feynkit import FeynkitDatabase, FeynmanIntegral, ValidationError
from feynkit.a_configuration import AConfiguration, FiniteIndexResult
from feynkit.cli import main
from feynkit.types import PolytopeEquivalence

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


@pytest.mark.parametrize("bad", ["12e|2e|e|:zz", "abc", "9e|e|", "", "0|:nn"])
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


def test_integral_that_cannot_be_built_exits_1_without_the_grammar(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # No CNickel string that parses is known to fail to build, so the kinematics are made to fail.
    def fail(n_external: int, use_mandelstam: bool = False) -> None:
        raise ValueError("no kinematics")

    monkeypatch.setattr("feynkit.integral.create_momentum_products", fail)
    db = tmp_path / "x.db"
    assert _exit_code(["analyse", "11e|e|:zz", "--db", str(db)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "fk: error: cannot build the integral of CNickel '11e|e|:zz': no kinematics\n"
    )
    assert not db.exists()


def test_compare_names_the_string_that_does_not_parse(capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(["compare", "11e|e|:zz", "abc", "--no-db"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("fk: error: cannot parse CNickel 'abc': ")


def test_bare_form_that_does_not_parse_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert _exit_code(["9e|e|", "--no-db"]) == 1
    assert capsys.readouterr().err.startswith("fk: error: cannot parse CNickel '9e|e|': ")


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
    assert _exit_code(["compare", "12e|2e|e|:zzz", "11e|e|:zz", "--no-db"]) == 3
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
    assert (
        _exit_code(["compare", "12e|2e|e|:zzz", "11e|e|:zz", "--db", str(tmp_path / "x.db")]) == 3
    )
    assert "Ambient dimension mismatch" in capsys.readouterr().out
    assert len(closed) == 1


@pytest.mark.parametrize(
    ("pair", "verdict"),
    [
        (["12e|2e|e|:zzz", "11e|e|:zz"], "No affine equivalence is possible"),
        (["12e|2e|e|:nzz", "12e|2e|e|:nnn"], "No equivalence found between A and B."),
    ],
    ids=["dimension-mismatch", "no-map"],
)
def test_compare_exits_3_when_no_check_finds_an_equivalence(
    pair: list[str], verdict: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    db = tmp_path / "x.db"
    assert _exit_code(["compare", *pair, "--db", str(db)]) == 3
    captured = capsys.readouterr()
    assert verdict in captured.out
    assert "Done in" in captured.out
    assert captured.err == ""
    with FeynkitDatabase(db) as stored:
        assert len(stored.all_integrals()) == 2


@pytest.mark.parametrize(
    ("pair", "finite_index"),
    [
        pytest.param(
            ["12e|2e|e|:zzz", "e111|e|:nzz"],
            "no  (only a singular map found, det = 0)",
            id="singular-map",
            marks=pytest.mark.slow,
        ),
        pytest.param(["e111|e|:nzz", "12e|2e|e|:zzz"], "no", id="reversed", marks=pytest.mark.slow),
    ],
)
def test_singular_finite_index_map_is_no_equivalence(
    pair: list[str], finite_index: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The only map the finite_index search finds from the massless triangle to
    # the one-mass sunrise is singular, and it finds none the other way. No
    # other check finds a map in either order.
    assert _exit_code(["compare", *pair, "--no-db"]) == 3
    out = capsys.readouterr().out
    assert f"  finite_index           {finite_index}\n" in out
    assert "YES" not in out
    assert "Witness map" not in out
    assert "No equivalence found between A and B." in out


def test_a_singular_map_alone_gives_status_3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The pair above takes half a minute; here every other check is made to
    # say no and the finite_index search to return a singular map.
    def no_map(self: AConfiguration, other: AConfiguration) -> PolytopeEquivalence:
        return PolytopeEquivalence(equivalent=False, relation="unimodular")

    for method in (
        "is_unimodular_equivalent_to",
        "is_affinely_equivalent_to",
        "is_point_config_equivalent_to",
    ):
        monkeypatch.setattr(AConfiguration, method, no_map)
    singular = FiniteIndexResult(
        found=True,
        witness_matrix=sp.ImmutableMatrix([[1, 0], [0, 0]]),
        translation=sp.ImmutableMatrix([0, 0]),
        determinant=0,
    )
    monkeypatch.setattr(cli, "finite_index_map", lambda a, b: singular)
    assert _exit_code(["compare", "12e|2e|e|:znn", "12e|2e|e|:nzn", "--no-db"]) == 3
    out = capsys.readouterr().out
    assert "  finite_index           no  (only a singular map found, det = 0)\n" in out
    assert "Witness map" not in out
    assert "No equivalence found between A and B." in out


def test_singular_finite_index_map_leaves_the_other_maps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The finite_index search stops at a singular map, but point_config finds
    # a map, so main returns with status 0 and prints its identity.
    main(["compare", "12e|2e|e|:znn", "12e|2e|e|:nzn", "--no-db"])
    out = capsys.readouterr().out
    assert "  finite_index           no  (only a singular map found, det = 0)\n" in out
    assert "  point_config           YES  (det = -1)\n" in out
    assert "Witness map  [finite_index]" not in out
    assert "GKZ identity I_A(beta, z_P) = |det M| I_B(T beta, z)" in out


def test_nonsingular_finite_index_map_gives_status_0_and_its_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # main returns, so the status is 0.
    main(["compare", "12e|2e|e|:nzz", "12e|2e|e|:znz", "--no-db"])
    out = capsys.readouterr().out
    assert "  finite_index           YES  (det = 1)\n" in out
    witness = out.split("Witness map  [finite_index]")[1]
    assert "GKZ identity I_A(beta, z_P) = |det M| I_B(T beta, z)" in witness
    assert "M is singular" not in witness


@pytest.mark.parametrize(
    ("argv", "status"),
    [
        (["--version"], 0),
        (["analyse", "0|:n", "--no-db"], 0),
        (["analyse", "abc", "--no-db"], 1),
        ([], 2),
        (["compare", "12e|2e|e|:nzz", "12e|2e|e|:znz", "--no-db"], 0),
        (["compare", "12e|2e|e|:nzz", "12e|2e|e|:nnn", "--no-db"], 3),
    ],
    ids=["version", "tadpole", "bad-cnickel", "no-command", "equivalent", "not-equivalent"],
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


def test_reader_that_closes_early_gets_status_141_and_no_traceback(tmp_path: Path) -> None:
    # The reader takes one line and closes, as head -1 does. Output is flushed
    # after each section, and the box's sections take seconds, so a later
    # flush meets the closed pipe.
    err = tmp_path / "stderr.txt"
    command = [sys.executable, "-m", "feynkit.cli", "analyse", "12e|3e|3e|e|:zzzz", "--no-db"]
    with (
        err.open("w", encoding="utf-8") as stderr,
        subprocess.Popen(
            command, cwd=tmp_path, stdout=subprocess.PIPE, stderr=stderr, text=True
        ) as process,
    ):
        assert process.stdout is not None
        assert process.stdout.readline().startswith("=")
        process.stdout.close()
        status = process.wait(timeout=120)
    assert status == 141
    assert err.read_text(encoding="utf-8") == ""
