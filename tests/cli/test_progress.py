"""Tests for the flushing, the stage timings and the headers of fk."""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest

from feynkit.cli import main

_TIMING = re.compile(r"fk: (?P<stage>[A-Za-z ]+) \d+\.\d{2}s")


def _stages(err: str) -> list[str]:
    """The stage names of the timing lines on stderr, which must be all of them."""
    stages = []
    for line in err.splitlines():
        match = _TIMING.fullmatch(line)
        assert match is not None, line
        stages.append(match["stage"])
    return stages


class _Recorder(io.StringIO):
    """A stdout that keeps what had been written at each flush."""

    def __init__(self) -> None:
        super().__init__()
        self.flushed: list[str] = []

    def flush(self) -> None:
        self.flushed.append(self.getvalue())
        super().flush()


def test_stdout_is_flushed_after_each_section(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)
    main(["analyse", "11e|e|:zz", "-s", "-g", "--no-db"])
    assert any(
        "Symanzik polynomials" in text and "GKZ hypergeometric" not in text
        for text in recorder.flushed
    )
    assert any("GKZ hypergeometric" in text and "Done in" not in text for text in recorder.flushed)


def test_verbose_times_each_analyse_stage_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:zz", "-s", "--no-db", "--verbose"])
    assert _stages(capsys.readouterr().err) == ["parse", "graph", "symanzik"]


def test_verbose_times_the_report_and_the_database(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(
        [
            "analyse",
            "11e|e|:zz",
            "--json",
            "--sections",
            "gkz",
            "-v",
            "--db",
            str(tmp_path / "x.db"),
        ]
    )
    captured = capsys.readouterr()
    assert _stages(captured.err) == ["parse", "report", "database"]
    json.loads(captured.out)


def test_stderr_is_quiet_without_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:zz", "-s", "--no-db"])
    assert capsys.readouterr().err == ""


def test_verbose_times_each_compare_stage(capsys: pytest.CaptureFixture[str]) -> None:
    main(["compare", "12e|2e|e|:zzz", "11e|e|:zz", "--no-db", "-v"])
    assert _stages(capsys.readouterr().err) == [
        "parse",
        "diagram A",
        "diagram B",
        "equivalence checks",
    ]


def test_compare_shows_each_input_with_its_canonical_form(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["compare", "11e|e|:zz", "11e|e|:zn", "--no-db"])
    out = capsys.readouterr().out
    assert "  A  =  11e|e|:zz\n" in out
    assert "  B  =  11e|e|:zn  (canonical form 11e|e|:nz)\n" in out
    assert "  Diagram A  11e|e|:zz\n" in out
    assert "  Diagram B  11e|e|:zn  (canonical form 11e|e|:nz)\n" in out


def test_analyse_header_shows_the_canonical_form_when_it_differs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["analyse", "11e|e|", "-s", "--no-db"])
    assert "  Feynman integral  11e|e|  (canonical form 11e|e|:zz)\n" in capsys.readouterr().out


def test_analyse_header_is_the_input_when_it_is_canonical(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["analyse", "11e|e|:zz", "-s", "--no-db"])
    assert "  Feynman integral  11e|e|:zz\n" in capsys.readouterr().out


def test_normalised_volume_is_the_rank_for_generic_beta(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:zz", "-n", "--no-db"])
    out = capsys.readouterr().out
    assert f"  {'Normalised volume':<28} 1  (the holonomic rank for generic beta)\n" in out
    assert "(holonomic rank)" not in out
