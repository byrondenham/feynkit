"""
Helpers shared by the plain-text renderers of the report sections.

The text printer, prose wrapping, tables, displays, and the document that
collects the citations.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence

import sympy as sp
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter

from ._report_shared import CITATIONS, Citations

_WIDTH = 79


_INDENT = "    "


# Operators that stand as words in the prose. The wrapper keeps them on the
# line of their operands, joining them with a no-break space that textwrap
# does not split at.
_OPERATORS = frozenset({"=", "+", "-", ".", "<=", "->"})


_GLUE = "\xa0"


# Accent macros used in the bibliography, as ASCII.
_ACCENTS = {
    '\\"a': "ae",
    '\\"o': "oe",
    '\\"u': "ue",
    "\\aa ": "aa",
}


# The Lee-Pomeransky integral without its prefactor, the function the GKZ
# system annihilates.
_TEXT_EULER_MELLIN = "I_A(beta, z) = int_{R_+^N} prod_e du_e u_e^(nu_e - 1) G(z, u)^(-D/2)"


class _TextPrinter(StrPrinter):
    """``sympy.sstr`` in the notation of the report's prose.

    Powers are written with ``^``, and an exponent other than a symbol or a
    non-negative integer is bracketed: ``u_1^(nu_1 - 1)``, ``G^(-D/2)``. A
    symbol whose name contains ``^``, such as the invariant ``p1^2``, is
    bracketed as the base of a power, so that its square reads ``(p1^2)^2``
    rather than ``p1^2^2``. The gamma function is written ``Gamma``, apart
    from the constant ``gamma_E``. Products keep ``*``.
    """

    def _print_Pow(self, expr: sp.Pow, rational: bool = False) -> str:
        if expr.exp is sp.S.Half and not rational:
            return f"sqrt({self._print(expr.base)})"
        if expr.is_commutative and -expr.exp is sp.S.Half and not rational:
            return f"1/sqrt({self._print(expr.base)})"
        if expr.is_commutative and expr.exp is sp.S.NegativeOne:
            return f"1/{self._base(expr.base)}"
        return f"{self._base(expr.base)}^{self._exponent(expr.exp)}"

    def _base(self, base: sp.Expr) -> str:
        if _caret_named(base):
            return f"({self._print(base)})"
        return str(self.parenthesize(base, PRECEDENCE["Pow"], strict=False))

    def _exponent(self, exponent: sp.Expr) -> str:
        if (isinstance(exponent, sp.Symbol) and not _caret_named(exponent)) or (
            exponent.is_Integer and exponent >= 0
        ):
            return str(self._print(exponent))
        return f"({self._print(exponent)})"

    def _print_gamma(self, expr: sp.Expr) -> str:
        return f"Gamma({self._print(expr.args[0])})"


def _caret_named(expr: sp.Expr) -> bool:
    return isinstance(expr, sp.Symbol) and "^" in expr.name


_str = _TextPrinter().doprint


class TextDocument(Citations):
    """The works the sections cite, for the references at the end."""

    def cite(self, *keys: str) -> str:
        self.add(*keys)
        return " [" + ", ".join(keys) + "]"

    def references(self) -> str:
        return "\n".join(
            textwrap.fill(
                f"[{key}] {_strip_latex(CITATIONS[key])}",
                width=_WIDTH,
                subsequent_indent=_INDENT,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for key in self.keys
        )


def _strip_latex(text: str) -> str:
    """A bibliography entry in plain ASCII.

    Accent macros are transliterated (``\\"a`` to ae, ``\\aa`` to aa), the
    ``\\emph``, ``\\textit`` and ``\\mathcal`` wrappers and dollar signs are
    dropped, ``~`` becomes a space and ``--`` a hyphen.
    """
    for macro, letters in _ACCENTS.items():
        text = text.replace(macro, letters)
    # Innermost wrappers first, so that one nested in another comes off too.
    unwrapped = None
    while unwrapped != text:
        unwrapped = text
        text = re.sub(r"\\(?:emph|textit|mathcal)\{([^{}]*)\}", r"\1", text)
    return text.replace("$", "").replace("~", " ").replace("--", "-")


def _heading(text: str, rule: str) -> str:
    return f"{text}\n{rule * len(text)}"


def _paragraph(text: str) -> str:
    """Wrap prose at 79 columns without breaking inline maths.

    A line never breaks inside brackets or next to an operator standing as a
    word of its own, so ``m_j . x <= b_j`` and ``(nu_1, ..., nu_N)`` each
    stay on one line.
    """
    units: list[str] = []
    depth = 0
    joined = False
    for word in text.split(" "):
        if units and (joined or depth > 0 or word in _OPERATORS):
            units[-1] += _GLUE + word
        else:
            units.append(word)
        depth = max(0, depth + sum(map(word.count, "([{")) - sum(map(word.count, ")]}")))
        joined = word in _OPERATORS
    wrapped = textwrap.fill(
        " ".join(units), width=_WIDTH, break_long_words=False, break_on_hyphens=False
    )
    return wrapped.replace(_GLUE, " ")


def _blocks(*blocks: str) -> str:
    """Paragraphs and displays, separated by blank lines."""
    return "\n\n".join(blocks)


def _break(text: str, width: int, separators: tuple[str, ...] = (" + ", " - ")) -> list[str]:
    """Break printed maths into lines of at most ``width`` characters where it can.

    A line breaks only before one of ``separators`` standing outside every
    pair of brackets, the top-level signs of a sum by default, and the pieces
    are packed greedily. A piece longer than ``width`` keeps a line of its
    own. Every line after the first starts with its separator.
    """
    if len(text) <= width:
        return [text]
    pieces: list[str] = []
    depth = 0
    start = 0
    for i, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif depth == 0 and i > start and text.startswith(separators, i):
            pieces.append(text[start:i])
            start = i
    pieces.append(text[start:])

    lines = [pieces[0]]
    for piece in pieces[1:]:
        if len(lines[-1]) + len(piece) > width:
            lines.append(piece.strip())
        else:
            lines[-1] += piece
    return lines


def _lines(expr: sp.Expr, width: int) -> list[str]:
    """``expr`` over lines of at most ``width``: a sum at its signs, then a long
    term at its top-level products and quotients."""
    return [
        piece for line in _break(_str(expr), width) for piece in _break(line, width, ("*", "/"))
    ]


def _room(lhs: str) -> int:
    """The width left for the right-hand side of a display of ``lhs = ...``."""
    return _WIDTH - len(_INDENT) - len(lhs) - 3


def _text_equation(lhs: str, lines: Sequence[str]) -> str:
    """Display ``lhs = ...`` over the given lines, continued below the first term."""
    continued = " " * (len(_INDENT) + len(lhs) + 3)
    return "\n".join([f"{_INDENT}{lhs} = {lines[0]}", *(continued + line for line in lines[1:])])


def _expressions(expressions: Sequence[sp.Expr]) -> str:
    """Display each expression on its own line, long ones continued further in."""
    rows = []
    for expr in expressions:
        text = _str(expr)
        whole = len(_INDENT) + len(text) <= _WIDTH
        lines = [text] if whole else _lines(expr, _WIDTH - 2 * len(_INDENT))
        rows.append(_INDENT + lines[0])
        rows.extend(_INDENT * 2 + line for line in lines[1:])
    return "\n".join(rows)


def _matrix(lhs: str, matrix: sp.Matrix, after: str = "") -> str:
    """Display ``lhs = matrix``, with ``lhs`` and ``after`` on the middle row."""
    rows = sp.pretty(matrix, use_unicode=False, wrap_line=False).splitlines()
    middle = len(rows) // 2
    head = f"{lhs} = "
    lines = [
        _INDENT + (head if k == middle else " " * len(head)) + row + (after if k == middle else "")
        for k, row in enumerate(rows)
    ]
    return "\n".join(line.rstrip() for line in lines)


def _text_vector(lhs: str, entries: Sequence[sp.Expr], end: str) -> str:
    """Display ``lhs = (e_1, e_2, ...)`` then ``end``, broken after commas to fit."""
    width = _room(lhs) - len(end)
    printed = [_str(e) for e in entries]
    lines = ["(" + printed[0]]
    for entry in printed[1:]:
        if len(lines[-1]) + len(entry) + 3 > width:
            lines[-1] += ","
            lines.append(" " + entry)
        else:
            lines[-1] += ", " + entry
    lines[-1] += ")" + end
    return _text_equation(lhs, lines)


def _table(header: Sequence[str] | None, rows: Sequence[Sequence[str]]) -> str:
    """An indented table with columns padded to their widest cell.

    The last column takes what is left of the line; an entry too long for it
    is broken at its top-level signs and continued below in the same column.
    """
    cells = ([list(header)] if header else []) + [list(row) for row in rows]
    widths = [max(len(row[c]) for row in cells) for c in range(len(cells[0]) - 1)]
    lead = len(_INDENT) + sum(width + 2 for width in widths)
    lasts = [_break(row[-1], _WIDTH - lead) for row in cells]
    widths.append(max(len(line) for last in lasts for line in last))

    def line(row: Sequence[str]) -> str:
        padded = "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        return (_INDENT + padded).rstrip()

    lines = []
    for k, (row, last) in enumerate(zip(cells, lasts, strict=True)):
        lines.append(line([*row[:-1], last[0]]))
        lines.extend(" " * lead + more for more in last[1:])
        if header and k == 0:
            lines.append(line(["-" * width for width in widths]))
    return "\n".join(lines)
