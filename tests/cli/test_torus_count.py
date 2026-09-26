"""Tests for --torus-count, --seed and --torus-budget of fk analyse."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from feynkit import FeynmanIntegral
from feynkit.cli import _with_command, main

HEADING = "  Candidate Euler characteristic from point counts\n"


def _exit(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int | str | None, str]:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return excinfo.value.code, capsys.readouterr().err


def test_alone_it_prints_the_graph_and_the_point_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["analyse", "11e|e|:nn", "--torus-count", "--no-db"])
    out = capsys.readouterr().out
    assert "  Feynman integral  11e|e|:nn\n" in out
    assert HEADING in out
    assert re.search(r"^  Candidate master count +3$", out, re.M)
    assert "evidence, not a proof" in out
    assert "Symanzik polynomials" not in out
    assert "Euler equations" not in out


def test_counts_without_a_candidate_say_why(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "12e|2e|e|:nnn", "--torus-count", "--no-db"])
    out = capsys.readouterr().out
    assert re.search(r"^  No candidate: .+\.$", out, re.M)
    assert "Candidate master count" not in out


def test_runs_without_section_flags_leave_it_out(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:zz", "--no-db"])
    out = capsys.readouterr().out
    assert "Symanzik polynomials" in out
    assert HEADING not in out


def test_it_joins_the_other_section_flags(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:nn", "-g", "--torus-count", "--no-db", "-v"])
    captured = capsys.readouterr()
    assert "Euler equations" in captured.out and HEADING in captured.out
    stages = [line.split()[1] for line in captured.err.splitlines()]
    assert stages == ["parse", "graph", "gkz", "torus"]


def test_seed_chooses_the_point(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:nn", "--torus-count", "--seed", "1", "--no-db"])
    out = capsys.readouterr().out
    count = FeynmanIntegral.from_cnickel("11e|e|:nn").torus_count(seed=1)
    point = ", ".join(f"{key} = {value}" for key, value in count.point)
    assert f"{point}  (seed 1)" in out


def test_no_point_is_drawn_without_kinematic_symbols(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "0|:z", "--torus-count", "--no-db"])
    out = capsys.readouterr().out
    assert re.search(r"^  Kinematic point +none needed: G has no kinematic symbols$", out, re.M)


def test_budget_too_small_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    code, err = _exit(
        ["analyse", "11e|e|:nn", "--torus-count", "--torus-budget", "10", "--no-db"], capsys
    )
    assert code == 1
    assert "max_evaluations=10" in err


@pytest.mark.parametrize(
    "argv",
    [["--seed", "0"], ["--torus-budget", "5"], ["-g", "--seed", "3"]],
    ids=["seed", "budget", "seed-with-another-flag"],
)
def test_seed_and_budget_need_the_point_counts(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code, err = _exit(["analyse", "11e|e|:nn", "--no-db", *argv], capsys)
    assert code == 2
    assert "applies to the point counts; add --torus-count or name torus in --sections" in err


def test_json_takes_the_point_counts_from_sections(capsys: pytest.CaptureFixture[str]) -> None:
    code, err = _exit(["analyse", "11e|e|:nn", "--json", "--torus-count", "--no-db"], capsys)
    assert code == 2
    assert "--json prints only the summary" in err
    main(["analyse", "11e|e|:nn", "--json", "--sections", "torus", "--seed", "1", "--no-db"])
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["candidate_master_count"] == 3


def test_json_gives_null_without_a_candidate(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    txt = tmp_path / "triangle.txt"
    main(
        ["analyse", "12e|2e|e|:nnn", "--json", "--text", str(txt)]
        + ["--sections", "torus", "--no-db"]
    )
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["candidate_master_count"] is None
    # Only the JSON changes: the report's summary still says none.
    assert re.search(r"^ *Candidate master count +none$", txt.read_text(encoding="utf-8"), re.M)


def test_the_count_runs_once_for_the_section_and_the_report(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[int] = []
    torus_count = FeynmanIntegral.torus_count

    def counting(self: FeynmanIntegral, **kwargs: object) -> object:
        calls.append(1)
        return torus_count(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(FeynmanIntegral, "torus_count", counting)
    txt = tmp_path / "bubble.txt"
    main(
        ["analyse", "11e|e|:nn", "--torus-count", "--text", str(txt)]
        + ["--sections", "polytope,torus", "--no-db"]
    )
    assert len(calls) == 1
    assert HEADING in capsys.readouterr().out
    assert "Candidate Euler characteristic from point counts\n" in txt.read_text(encoding="utf-8")


def test_bare_form_skips_the_values_of_seed_and_budget() -> None:
    # Read as a positional, the value would make a second diagram, and so fk compare.
    for option, value in (("--seed", "3"), ("--torus-budget", "1000")):
        argv = ["11e|e|:nn", "--torus-count", option, value]
        assert _with_command(argv) == ["analyse", *argv]
