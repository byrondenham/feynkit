"""Tests for -D, --degeneracy of fk analyse."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from feynkit import FeynmanIntegral
from feynkit import degeneracy as degeneracy_module
from feynkit.cli import main
from feynkit.io.sections.degeneracy import torus_point
from feynkit.landau import _singular_binary

pytestmark = [
    pytest.mark.skipif(_singular_binary() is None, reason="Singular not installed"),
    pytest.mark.usefixtures("sympy_seeded"),
]

HEADING = "  Degenerate faces  (at the kinematic point of the point counts)\n"


def _row(key: str, value: object) -> str:
    return f"  {key:<28} {value}"


def _rows(out: str) -> list[str]:
    """The lines of the section, from the rule under its heading to the next rule."""
    lines = out.split(HEADING, 1)[1].splitlines()[1:]
    end = next(k for k, line in enumerate(lines) if line.startswith("--") or line.startswith("=="))
    return lines[:end]


def test_it_prints_the_degenerate_faces_at_the_point_of_the_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["analyse", "111e|e|:nnn", "-D", "--no-db"])
    out = capsys.readouterr().out
    assert "Symanzik polynomials" not in out
    fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")
    point = torus_point(fi, 0)
    assert tuple(point.items()) == fi.torus_count().point
    analysis = fi.face_degeneracy(point)
    where = ", ".join(f"{key} = {value}" for key, value in analysis.point or ())
    rows = _rows(out)
    assert rows[:3] == [
        _row("Kinematic point", f"{where}  (seed 0)"),
        _row("Normalised volume", 10),
        _row("Degenerate faces", 3),
    ]
    faces = [row for row in rows if row.startswith("    dimension")]
    assert len(faces) == 3
    assert all(re.search(r"dim Sing 0 +tau 1 +U\(\{\d,\d\}\) G\(Gamma/", row) for row in faces)
    assert rows[-1] == _row("Check over F_p", f"agrees on every face  (p = {analysis.prime})")


def test_no_degenerate_face_gives_the_volume(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "12e|2e|e|:zzz", "--degeneracy", "--no-db"])
    rows = _rows(capsys.readouterr().out)
    fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
    volume = fi.face_degeneracy(torus_point(fi, 0)).volume
    assert _row("Degenerate faces", 0) in rows
    assert f"  No face is degenerate, so |chi(X)| = N! Vol(P_z) = {volume}." in rows


def test_runs_without_the_flag_leave_it_out(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:zz", "--no-db"])
    assert HEADING not in capsys.readouterr().out


def test_the_seed_chooses_the_point(capsys: pytest.CaptureFixture[str]) -> None:
    main(["analyse", "11e|e|:nn", "-D", "--seed", "1", "--no-db"])
    point = FeynmanIntegral.from_cnickel("11e|e|:nn").torus_count(seed=1).point
    where = ", ".join(f"{key} = {value}" for key, value in point)
    assert _row("Kinematic point", f"{where}  (seed 1)") in _rows(capsys.readouterr().out)


def test_it_adds_the_section_to_the_report(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    txt = tmp_path / "bubble.txt"
    main(["analyse", "11e|e|:nn", "-D", "--text", str(txt), "--sections", "gkz", "--no-db"])
    assert HEADING in capsys.readouterr().out
    text = txt.read_text(encoding="utf-8")
    assert "\nDegenerate faces\n----------------\n" in text


def test_json_takes_it_from_sections(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["analyse", "111e|e|:nnn", "--json", "-D", "--no-db"])
    assert excinfo.value.code == 2
    assert "--json prints only the summary" in capsys.readouterr().err
    main(["analyse", "111e|e|:nnn", "--json", "--sections", "degeneracy", "--no-db"])
    data = json.loads(capsys.readouterr().out)
    assert data["summary"]["degenerate_faces"] == 3


def test_without_singular_it_says_so(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(degeneracy_module, "_singular_binary", lambda: None)
    main(["analyse", "11e|e|:nn", "-D", "--no-db"])
    rows = _rows(capsys.readouterr().out)
    assert "  Not decided: faces of dimension 2 or more need Singular, which was not found." in rows
