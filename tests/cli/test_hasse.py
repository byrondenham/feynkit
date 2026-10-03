"""Tests for fk analyse --hasse."""

from __future__ import annotations

from pathlib import Path

import pytest

from feynkit import FeynmanIntegral
from feynkit.cli import main
from feynkit.visualisation.hasse import hasse_document


def test_hasse_writes_the_standalone_document(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "box.tex"
    main(["analyse", "12e|3e|3e|e|:nnnn", "-s", "--hasse", str(target), "--no-db"])
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    expected = hasse_document(fi.face_lattice(nu={1: 1, 2: 1, 3: 1, 4: 1}), "codim", 2)
    assert target.read_text(encoding="utf-8") == expected
    assert f"  Wrote the Hasse diagram to {target}\n" in capsys.readouterr().out


def test_the_view_options_reach_the_diagram(tmp_path: Path) -> None:
    target = tmp_path / "box.tex"
    main(
        ["analyse", "12e|3e|3e|e|:nnnn", "-s", "--hasse", str(target)]
        + ["--hasse-view", "filter", "--hasse-filter", "contraction", "--no-db"]
    )
    fi = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    lattice = fi.face_lattice(nu={1: 1, 2: 1, 3: 1, 4: 1})
    assert target.read_text(encoding="utf-8") == hasse_document(
        lattice, "filter", filter="contraction"
    )
    main(
        ["analyse", "12e|3e|3e|e|:nnnn", "-s", "--hasse", str(target)]
        + ["--hasse-view", "upset", "--hasse-face", "0,1,2,3", "--hasse-highlight", "ir"]
        + ["--hasse-codim", "1", "--hasse-max-faces", "50", "--no-db"]
    )
    assert target.read_text(encoding="utf-8") == hasse_document(
        lattice, "upset", 1, face=(0, 1, 2, 3), highlight="ir", max_faces=50
    )


def test_a_view_over_the_limit_exits_1_with_the_count(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    target = tmp_path / "box.tex"
    with pytest.raises(SystemExit) as caught:
        main(
            ["analyse", "12e|3e|3e|e|:nnnn", "-s", "--hasse", str(target)]
            + ["--hasse-max-faces", "5", "--no-db"]
        )
    assert caught.value.code == 1
    assert "max_faces" in capsys.readouterr().err
    assert not target.exists()


@pytest.mark.parametrize(
    "extra",
    [["--hasse-view", "all"], ["--hasse-codim", "1"], ["--hasse-filter", "ir"]],
)
def test_the_view_options_need_hasse(extra: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["analyse", "11e|e|:nn", "-s", *extra, "--no-db"])
    assert caught.value.code == 2


def test_a_bad_face_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["analyse", "11e|e|:nn", "--hasse", "x.tex", "--hasse-face", "a,b", "--no-db"])
    assert caught.value.code == 2
