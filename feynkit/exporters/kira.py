"""
Kira inputs from an integral family.

Kira (Maierhoefer, Usovitsch and Uwer, arXiv:1705.05610; Klappert, Lange, Maierhoefer and
Usovitsch, arXiv:2008.06494) reads a job directory: ``jobs.yaml``, and under ``config`` the
files ``integralfamilies.yaml`` and ``kinematics.yaml``. :func:`kira_job` writes their text
from an :class:`~feynkit.family.IntegralFamily`; nothing here runs Kira.

Signs. Kira writes a propagator ``[q, m^2]`` as 1/(q^2 - m^2) (Weinzierl, Feynman Integrals,
Springer 2022, App. J, Eq. J.259), feynkit's is 1/(-q^2 + m^2). Each function D_alpha is
exported as the Kira function D^K_alpha = s_alpha D_alpha, with s_alpha = -1 for a squared
function -q^2 + m^2 and for -l_i . p_k, and +1 for l_i . p_k, Kira's ``bilinear`` function.
Hence J(n) = prod_alpha s_alpha^(n_alpha) J^K(n), for J the integrals of the family and J^K
those Kira computes; :attr:`KiraJob.signs` holds s.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import sympy as sp

from ..core.exceptions import ValidationError
from ..family import FamilyFunction, IntegralFamily, _gram

__all__ = ["KiraJob", "kira_job"]

# Names that Kira or its algebra system read as something else.
RESERVED = frozenset({"d", "I", "Pi", "Euler", "Catalan", "pi", "e", "E", "den"})
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
MOMENTUM_NAME = re.compile(r"^[lp][0-9]+$")


def _text(expr: object) -> str:
    """An expression in Kira's syntax: ``^`` for powers."""
    return str(sp.sstr(sp.sympify(expr))).replace("**", "^")


def _momentum_text(loop: Sequence[int], external: Sequence[int]) -> str:
    """q as ``l1-l2+2*p1``, the first coefficient positive."""
    names = [f"l{i}" for i in range(1, len(loop) + 1)]
    names += [f"p{k}" for k in range(1, len(external) + 1)]
    out = ""
    for c, name in zip((*loop, *external), names, strict=True):
        if not c:
            continue
        term = name if abs(c) == 1 else f"{abs(c)}*{name}"
        out += (("" if c > 0 else "-") if not out else ("+" if c > 0 else "-")) + term
    return out


def _root(x: Fraction) -> Fraction | None:
    """The non-negative rational square root of x, or None."""
    if x < 0:
        return None
    a, b = math.isqrt(x.numerator), math.isqrt(x.denominator)
    return Fraction(a, b) if a * a == x.numerator and b * b == x.denominator else None


def _classify(f: FamilyFunction) -> tuple:
    """
    How Kira writes a function, and the sign s with D^K = s D.

    Returns ("squared", loop, external, mass_squared, -1) for -q^2 + m^2 with integer q,
    or ("bilinear", i, k, s) for s l_i . p_k.
    """
    a, b, e, c = f.form
    loops, externals = f.loops, f.externals
    if all(x == 0 for row in a for x in row):
        flat = [(i, k, b[i][k]) for i in range(loops) for k in range(externals) if b[i][k]]
        if (
            len(flat) == 1
            and abs(flat[0][2]) == Fraction(1, 2)
            and all(x == 0 for row in e for x in row)
            and c == 0
        ):
            i, k, x = flat[0]
            return ("bilinear", i, k, 1 if x > 0 else -1)
        raise ValidationError(
            f"Kira cannot express {f.label}: with no quadratic part it must be one scalar "
            "product l_i . p_k, up to a sign"
        )
    head = next((r for r in range(loops) if a[r][r]), None)
    lam_i = None if head is None else _root(-a[head][head])
    if head is None or lam_i is None or lam_i == 0:
        raise ValidationError(f"Kira cannot express {f.label}: its quadratic part is not -q^2")
    lam = [-a[head][j] / lam_i for j in range(loops)]
    lam[head] = lam_i
    kap = [-b[head][k] / lam_i for k in range(externals)]
    rebuilt = (
        tuple(tuple(-x * y for y in lam) for x in lam),
        tuple(tuple(-x * y for y in kap) for x in lam),
        tuple(tuple(-x * y for y in kap) for x in kap),
    )
    if rebuilt != (a, b, e):
        raise ValidationError(
            f"Kira cannot express {f.label}: its form is not -q^2 + m^2 for one momentum q"
        )
    if any(x.denominator != 1 for x in (*lam, *kap)):
        raise ValidationError(
            f"Kira cannot express {f.label}: its momentum has fractional coefficients"
        )
    return (
        "squared",
        tuple(int(x) for x in lam),
        tuple(int(x) for x in kap),
        c,
        -1,
    )


def _weights(expressions: Iterable[sp.Expr], symbols: list[sp.Symbol]) -> dict[sp.Symbol, int]:
    """
    The mass dimension of each symbol: 1 for one that only has even powers, else 2.

    Every expression must then be homogeneous of dimension 2.
    """
    polys = []
    for expr in expressions:
        try:
            polys.append(sp.Poly(expr, *symbols) if symbols else None)
        except sp.PolynomialError as err:
            raise ValidationError(
                f"the kinematic expression {_text(expr)} is not a polynomial in the invariants"
            ) from err
    odd: set[sp.Symbol] = set()
    for poly in polys:
        if poly is None:
            continue
        for monomial in poly.monoms():
            odd.update(s for s, p in zip(symbols, monomial, strict=True) if p % 2)
    weight = {s: (2 if s in odd else 1) for s in symbols}
    for expr, poly in zip(expressions, polys, strict=True):
        if poly is None or poly.is_zero:
            continue
        for monomial in poly.monoms():
            if sum(weight[s] * p for s, p in zip(symbols, monomial, strict=True)) != 2:
                raise ValidationError(
                    f"the kinematic expression {_text(expr)} does not have mass dimension 2"
                )
    return weight


@dataclass(frozen=True)
class KiraJob:
    """
    The text of a Kira job directory.

    Attributes
    ----------
    name
        The name of the family in the job files.
    integralfamilies, kinematics, jobs
        The text of ``config/integralfamilies.yaml``, ``config/kinematics.yaml`` and
        ``jobs.yaml``.
    integrals
        The text of ``integrals``, one integral per line as ``name[n1,...,nN]``, or None.
    signs
        s_alpha for each function, so that J(n) = prod_alpha s_alpha^(n_alpha) J^K(n).
    symbols
        The invariants whose names were changed for Kira, as (feynkit name, Kira name).
    """

    name: str
    integralfamilies: str
    kinematics: str
    jobs: str
    integrals: str | None
    signs: tuple[int, ...]
    symbols: tuple[tuple[str, str], ...] = ()

    def files(self) -> dict[str, str]:
        """The files of the job directory, by their path under it."""
        out = {
            "jobs.yaml": self.jobs,
            "config/integralfamilies.yaml": self.integralfamilies,
            "config/kinematics.yaml": self.kinematics,
        }
        if self.integrals is not None:
            out["integrals"] = self.integrals
        return out

    def write(self, directory: str | Path, *, overwrite: bool = False) -> None:
        """
        Write the files under a directory, which is created if it does not exist.

        Raises
        ------
        ValidationError
            If a file exists and overwrite is False; nothing is written then.
        """
        root = Path(directory)
        files = self.files()
        if not overwrite:
            clash = sorted(str(root / p) for p in files if (root / p).exists())
            if clash:
                raise ValidationError(
                    f"refusing to overwrite {', '.join(clash)}; pass overwrite=True"
                )
        for path, text in files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)


def _invariants(
    family: IntegralFamily, functions: Sequence[tuple]
) -> tuple[list[tuple[int, int, sp.Expr]], dict[sp.Symbol, int], dict[sp.Symbol, sp.Symbol]]:
    n = family.integral.graph.external_legs
    gram = _gram(family.integral, n)
    rules = [(j + 1, k + 1, sp.expand(gram[j][k])) for j in range(n - 1) for k in range(j, n - 1)]
    expressions = [r[2] for r in rules]
    for item in functions:
        if item[0] == "squared":
            expressions.append(sp.expand(item[3]))
    symbols: set[sp.Symbol] = set()
    for expr in expressions:
        symbols |= expr.free_symbols
    ordered = sorted(symbols, key=lambda s: s.name)
    renamed: dict[sp.Symbol, sp.Symbol] = {}
    for s in ordered:
        new = s.name.replace("^2", "sq")
        if not NAME.match(new):
            raise ValidationError(f"Kira cannot parse the symbol name {s.name!r}")
        if new in RESERVED or MOMENTUM_NAME.match(new):
            raise ValidationError(f"the symbol name {s.name!r} is reserved by Kira or the export")
        renamed[s] = sp.Symbol(new)
    if len({s.name for s in renamed.values()}) != len(renamed):
        raise ValidationError("two invariants have the same name once made fit for Kira")
    rules = [(j, k, expr.subs(renamed, simultaneous=True)) for j, k, expr in rules]
    expressions = [x.subs(renamed, simultaneous=True) for x in expressions]
    weights = _weights(expressions, [renamed[s] for s in ordered])
    return rules, weights, renamed


def kira_job(
    family: IntegralFamily,
    *,
    name: str = "family",
    top_sectors: Sequence[int] | None = None,
    replace_by_one: str | None = None,
) -> KiraJob:
    """
    The Kira job directory of a family: its functions and kinematics.

    The loop momenta are ``l1, ..., lL`` of the family's routing and all external legs are
    incoming, ``p1, ..., pn``, with ``pn`` eliminated by momentum conservation. The
    scalar-product rules are those of the integral's momentum products. The invariants are
    the free symbols of those and of the masses; a symbol that occurs only in even powers,
    such as a mass, is given mass dimension 1 and the others 2.

    Parameters
    ----------
    family
        The family.
    name
        The name of the family in Kira, a word of letters, digits and underscores.
    top_sectors
        The sector identities N_id to reduce, by default only the sector of every
        propagator.
    replace_by_one
        An invariant that Kira sets to one; its dependence is then reconstructed back.

    Raises
    ------
    ValidationError
        If a function is not one that Kira writes (-q^2 + m^2 for an integer combination q,
        or +-l_i . p_k); if a symbol name is reserved or cannot be parsed, or the
        kinematics are not homogeneous; or if an argument is malformed.
    """
    if not isinstance(name, str) or not NAME.match(name):
        raise ValidationError(f"the family name must be a word of letters and digits; got {name!r}")
    legs = family.integral.graph.external_legs
    functions = [_classify(f) for f in family.functions]
    count = family._propagator_count
    signs = tuple(item[-1] for item in functions)
    rules, weights, rename = _invariants(family, functions)
    changed = tuple((s.name, new.name) for s, new in rename.items() if s.name != new.name)

    lines = ["integralfamilies:", f'  - name: "{name}"']
    loop_names = ", ".join(f"l{i}" for i in range(1, family.routing.loops + 1))
    lines.append(f"    loop_momenta: [{loop_names}]")
    sectors = [(1 << count) - 1] if top_sectors is None else [int(x) for x in top_sectors]
    for sector in sectors:
        if not 0 < sector < 1 << count:
            raise ValidationError(f"the top sector {sector} is not a sector of {count} propagators")
    lines.append(f"    top_level_sectors: [{', '.join(map(str, sectors))}]")
    lines.append("    propagators:")
    for item in functions:
        if item[0] == "squared":
            _, loop, external, mass, _sign = item
            mass = mass.subs(rename, simultaneous=True)
            mass_text = "0" if mass == 0 else f'"{_text(mass)}"'
            lines.append(f'      - ["{_momentum_text(loop, external)}", {mass_text}]')
        else:
            _, i, k, _sign = item
            lines.append(f'      - {{ bilinear: [ ["l{i + 1}", "p{k + 1}"], 0 ] }}')
    families = "\n".join(lines) + "\n"

    legs_text = ", ".join(f"p{k}" for k in range(1, legs + 1))
    rest = "-".join(f"p{k}" for k in range(1, legs))
    kin = ["kinematics:", f"  incoming_momenta: [{legs_text}]", "  outgoing_momenta: []"]
    kin.append(f"  momentum_conservation: [p{legs}, -{rest}]")
    if weights:
        kin.append("  kinematic_invariants:")
        kin += [f"    - [{sym.name}, {w}]" for sym, w in weights.items()]
    kin.append("  scalarproduct_rules:")
    for j, k, expr in rules:
        kin.append(f"    - [[p{j}, p{k}], {_text(expr)}]")
    if replace_by_one is not None:
        if replace_by_one not in {sym.name for sym in weights}:
            raise ValidationError(f"{replace_by_one!r} is not one of the invariants")
        kin.append(f"  symbol_to_replace_by_one: {replace_by_one}")
    kinematics = "\n".join(kin) + "\n"

    return KiraJob(
        name=name,
        integralfamilies=families,
        kinematics=kinematics,
        jobs="",
        integrals=None,
        signs=signs,
        symbols=changed,
    )
