"""Each fk command in the guide and the README parses and quotes CNickel strings that parse."""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pytest

from feynkit.cli import _build_parser, _load, _report_options, _with_command

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide.md"
README = ROOT / "README.md"
HEADINGS = {GUIDE: "## CLI: fk", README: "## CLI: `fk`"}


def _section(path: Path) -> str:
    """The CLI section of path, up to the next second-level heading."""
    text = path.read_text(encoding="utf-8")
    start = text.index(HEADINGS[path])
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _commands(path: Path) -> list[str]:
    """The fk commands in the bash blocks of the CLI section of path."""
    blocks = re.findall(r"```bash\n(.*?)```", _section(path), re.DOTALL)
    return [line for block in blocks for line in block.splitlines() if line.startswith("fk ")]


COMMANDS = [(path.name, line) for path in HEADINGS for line in _commands(path)]


@pytest.mark.parametrize("path", list(HEADINGS), ids=lambda path: path.name)
def test_cli_section_has_commands(path: Path) -> None:
    assert _commands(path)


@pytest.mark.parametrize(("doc", "command"), COMMANDS)
def test_documented_command_quotes_its_cnickel_strings(doc: str, command: str) -> None:
    code = command.split("#", 1)[0]
    assert not re.search(r"[0-9e]\|", re.sub(r'"[^"]*"', "", code))


def _parse(command: str) -> argparse.Namespace:
    """The arguments of command as fk parses them, with the checks analyse adds."""
    parsers = _build_parser()
    args = parsers.main.parse_args(_with_command(shlex.split(command, comments=True)[1:]))
    if args.command == "analyse":
        _report_options(parsers.analyse, args)
    return args


@pytest.mark.parametrize(("doc", "command"), COMMANDS)
def test_documented_command_parses(doc: str, command: str) -> None:
    _parse(command)


@pytest.mark.parametrize(("doc", "command"), COMMANDS)
def test_documented_cnickel_strings_parse(doc: str, command: str) -> None:
    args = _parse(command)
    strings = [args.cnickel] if args.command == "analyse" else [args.first, args.second]
    for cnickel in strings:
        _load(cnickel)


def test_guide_documents_every_option() -> None:
    section = _section(GUIDE)
    options = {
        option
        for parser in _build_parser()
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    missing = sorted(
        option
        for option in options
        if not re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", section)
    )
    assert missing == []
