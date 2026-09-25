"""The version is stated consistently wherever it is recorded."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import feynkit

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    assert match.group(1) == feynkit.__version__


def test_citation_matches_the_version_and_licence() -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert data["cff-version"] == "1.2.0"
    assert data["title"] == "feynkit"
    assert data["authors"] == [{"family-names": "Denham", "given-names": "Byron"}]
    assert data["version"] == feynkit.__version__
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["date-released"])
    assert data["repository-code"] == "https://github.com/byrondenham/feynkit"
    assert data["license"] == "MIT"
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_changelog_opens_with_unreleased_then_this_release() -> None:
    yaml = pytest.importorskip("yaml")
    released = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))["date-released"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", changelog, re.MULTILINE)
    assert headings[:2] == ["Unreleased", f"{feynkit.__version__} ({released})"]
