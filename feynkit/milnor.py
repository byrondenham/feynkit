"""
The Euler characteristic of the Milnor fibre of a hypersurface germ, through Lê numbers.

Let f be a polynomial in n + 1 variables with rational coefficients that
vanishes at a rational point p, and F its Milnor fibre at p. This module
computes the reduced Euler characteristic chi~(F) = chi(F) - 1 and the Lê
numbers of f at p, following the lectures of D. B. Massey, "Non-isolated
hypersurface singularities and Lê cycles", arXiv:1410.3312 (page numbers
below are that PDF's):

- If df(p) != 0, F is contractible and chi~(F) = 0.
- If p is an isolated critical point, chi~(F) = (-1)^n mu, mu the Milnor
  number, the length at p of the ring modulo the Jacobian ideal (Ex. 3.11,
  p. 21, with Thm 4.1, p. 22).
- Otherwise let s >= 1 be the dimension of the critical locus at p and
  z_0, ..., z_n generic linear coordinates centred at p. By Lê's attaching
  theorem (D. T. Lê, Ann. Inst. Fourier 23 (1973) 261-270; Massey, Thm 2.23,
  p. 14), F is obtained up to homotopy from the Milnor fibre of f restricted
  to the hyperplane V(z_0) by attaching (Gamma^1 . V(f))_p cells of dimension
  n, where Gamma^1 is the relative polar curve, the closure of
  V(df/dz_1, ..., df/dz_n) less the critical locus. So
  chi~(F) = chi~(F_1) + (-1)^n (Gamma^1 . V(f))_p, with f_1 = f|V(z_0), whose
  critical locus has dimension s - 1; after s steps the restriction has an
  isolated critical point.

Each step splits (Gamma^1 . V(f))_p into gamma^1 + lambda^0, the intersection
numbers of Gamma^1 with V(z_0) and V(df/dz_0) (the Teissier trick, Massey,
Ex. 3.3, p. 19). The first is the polar number of the step, the second its
Lê number lambda^0, and Massey's Ex. 4.6(1), p. 24, gives the Lê numbers of f
from those along the flag: lambda^0_f = lambda^0_{f_0} and
lambda^k_f = lambda^0_{f_k} - gamma^1_{f_{k-1}} for 1 <= k <= s, with f_k the
restriction to V(z_0, ..., z_{k-1}) and lambda^0_{f_s} its Milnor number. By
Massey's Thm 4.1, p. 22, and the formula after it on p. 23,
chi~(F) = sum_k (-1)^(n-k) lambda^k_f, which the attaching sum equals
once every Teissier split holds.

The coordinates are drawn from a seed as a unit upper triangular change of
the variables less p, with random integer coefficients below 2^28, applied
one shear per step so that the polynomial stays sparse. Generic coordinates
satisfy the conditions under which Lê and polar numbers exist (Massey,
Def. 3.8, p. 20, and Remark 3.9); each step checks those that concern it:
its polar curve has dimension at most 1 at p and meets V(z_k), V(df_k/dz_k)
and V(f_k) there in finitely many points, the critical locus loses one
dimension (Ex. 4.6(1)), and the Teissier split holds. If a check fails the
computation draws new coordinates once, then gives up with the reason, never
a guess.

Singular computes over F_p, at the largest primes below 2^29 that divide no
numerator or denominator of a coefficient of f centred at p. The numbers
over F_p equal those over Q for all but finitely many primes, so the results
are probabilistic, like those of
:func:`~feynkit.point_count.critical_point_count`. The polar curve is
saturated in the polynomial ring by repeated ideal quotients until it stops
growing; local orderings serve only for lengths of ideals that are finite at
p, where they are fast. Both public functions compute two independent flags,
one at each of the two largest such primes, which must give the same Lê and
polar numbers, and check the first against the Iomdine-Lê formula (Massey,
Ex. 4.6(2), p. 24), which :func:`milnor_fibre_euler_characteristic` can
skip. Generic coordinates give the same Lê numbers (p. 22), so
flags that disagree leave the germ undecided, even when their Euler
characteristics agree.

Every ComputationError raised for a germ the computation cannot decide has a
message that starts with "undecided: " and goes on with the reason.
"""

from __future__ import annotations

import numbers
import random
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import sympy as sp

from .core.exceptions import ComputationError, ValidationError
from .landau import _check_timeout, _singular_binary

__all__ = [
    "DEFAULT_TIMEOUT",
    "LeNumbers",
    "le_numbers",
    "milnor_fibre_euler_characteristic",
]

# The seconds each Singular run gets by default.
DEFAULT_TIMEOUT = 120

# Primes stay below 2^29, where Singular's prime fields work in every ring.
_PRIME_LIMIT = 2**29

# The coefficients of the random coordinates and hyperplanes lie in [1, _COEFFICIENT_LIMIT),
# below every prime used, so none vanishes modulo the prime.
_COEFFICIENT_LIMIT = 2**28

# How many sets of coordinates a flag may try before the computation gives up.
_ATTEMPTS = 2

Method = Literal["submersion", "isolated", "attaching"]


@dataclass(frozen=True)
class LeNumbers:
    """The Lê numbers of a hypersurface germ with respect to a flag of coordinates.

    Massey, "Non-isolated hypersurface singularities and Lê cycles",
    arXiv:1410.3312, Def. 3.8, p. 20, defines them; the computation follows
    Ex. 4.6(1), p. 24 (see the module docstring).

    Attributes
    ----------
    numbers
        lambda^0, ..., lambda^s at the point; (mu,) at an isolated critical
        point and () where f is a submersion.
    polar_numbers
        gamma^1 of f_0, ..., f_{s-1}, f_k the restriction of f to
        V(z_0, ..., z_{k-1}): the intersection number at the point of its
        relative polar curve with V(z_k). gamma^1 of f_0 is Massey's polar
        number gamma^1_{f,z}(0). Empty unless s >= 1.
    critical_dimension
        s, the dimension of the critical locus of f at the point; 0 at an
        isolated critical point and -1 where f is a submersion.
    flag
        The coordinates z_0, ..., z_n of the first flag, linear forms in the
        variables less their values at the point, as rows of integer
        coefficients in the order of the variables. The rows form a unit upper
        triangular matrix, and only z_0, ..., z_{s-1} differ from the
        variables. Over F_p the forms are reduced modulo ``prime``.
    prime
        The prime of the first flag; a second flag, at the next prime below
        it that divides no coefficient, gave the same numbers. None for a
        submersion, decided exactly.
    method
        "submersion" (df != 0 at the point), "isolated" (a Milnor number by a
        local standard basis) or "attaching" (Lê's attaching theorem along the
        flag).
    """

    numbers: tuple[int, ...]
    polar_numbers: tuple[int, ...]
    critical_dimension: int
    flag: tuple[tuple[int, ...], ...]
    prime: int | None
    method: Method

    @property
    def reduced_euler_characteristic(self) -> int:
        """chi~(F) = chi(F) - 1 = sum_k (-1)^(n-k) lambda^k, F the Milnor fibre and n + 1 the
        number of variables (Massey, arXiv:1410.3312, Thm 4.1, p. 22, and the formula after it
        on p. 23)."""
        n = len(self.flag) - 1
        return sum((-1) ** (n - k) * value for k, value in enumerate(self.numbers))


# --- the germ ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Germ:
    """f centred at the point: f(at + x) as exponent vectors and rational coefficients."""

    terms: tuple[tuple[tuple[int, ...], Fraction], ...]
    variables: int
    submersion: bool


def _rational(value: object) -> Fraction:
    """A coordinate of the point: an int, a Fraction or a SymPy rational, which registers
    itself as a numbers.Rational."""
    if isinstance(value, numbers.Rational) and not isinstance(value, bool):
        return Fraction(int(value.numerator), int(value.denominator))
    raise ValidationError(f"a coordinate of the point must be rational, not {value!r:.60}")


def _germ(polynomial: object, variables: Sequence[sp.Symbol], at: Sequence[object]) -> _Germ:
    """f(at + x) after validating the input.

    Raises
    ------
    ValidationError
        If the variables are not one or more distinct symbols, f is not a non-zero polynomial
        with rational coefficients in them, ``at`` is not one rational number per variable, or
        f does not vanish at ``at``.
    """
    variables = tuple(variables)
    if (
        not variables
        or not all(isinstance(x, sp.Symbol) for x in variables)
        or len(set(variables)) != len(variables)
    ):
        raise ValidationError("variables must be one or more distinct symbols")
    try:
        f = sp.sympify(polynomial)
        if f.free_symbols - set(variables):
            raise sp.PolynomialError
        poly = sp.Poly(f, *variables, domain="QQ")
    except (sp.SympifyError, sp.PolynomialError, sp.CoercionFailed, TypeError) as exc:
        raise ValidationError(
            "f must be a polynomial with rational coefficients in the variables"
        ) from exc
    if poly.is_zero:
        raise ValidationError("f must not be zero")
    if isinstance(at, (str, bytes)) or not isinstance(at, Sequence) or len(at) != len(variables):
        raise ValidationError(
            f"at must give one rational number for each of the {len(variables)} variables"
        )
    point = [_rational(value) for value in at]
    shift = {
        x: x + sp.Rational(q.numerator, q.denominator)
        for x, q in zip(variables, point, strict=True)
    }
    centred = sp.Poly(poly.as_expr().subs(shift, simultaneous=True), *variables, domain="QQ")
    terms = tuple(
        (tuple(int(e) for e in monomial), Fraction(int(c.p), int(c.q)))
        for monomial, c in centred.terms()
    )
    if any(sum(monomial) == 0 for monomial, _ in terms):
        raise ValidationError("f must vanish at the point")
    return _Germ(
        terms=terms,
        variables=len(variables),
        submersion=any(sum(monomial) == 1 for monomial, _ in terms),
    )


def _primes(germ: _Germ, count: int) -> list[int]:
    """The ``count`` largest primes below 2^29 that divide no numerator or denominator of a
    coefficient of f centred at the point."""
    values = [abs(n) for _, c in germ.terms for n in (c.numerator, c.denominator)]
    found: list[int] = []
    p = _PRIME_LIMIT
    while len(found) < count:
        p = int(sp.prevprime(p))
        if all(v % p for v in values):
            found.append(p)
    return found


# --- coordinates ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Draw:
    """Random coordinates and hyperplanes for one attempt.

    shears[k] holds the coefficients c_{k,i}, i > k, of z_k = x_k + sum_i c_{k,i} x_i;
    hyperplanes are generic linear forms through the point, for dimension tests.
    """

    shears: tuple[tuple[int, ...], ...]
    hyperplanes: tuple[tuple[int, ...], ...]


def _draw(seed: int, flag: int, attempt: int, m: int) -> _Draw:
    rng = random.Random(f"milnor:{seed}:{flag}:{attempt}")
    shears = tuple(
        tuple(rng.randrange(1, _COEFFICIENT_LIMIT) for _ in range(m - 1 - k)) for k in range(m - 1)
    )
    hyperplanes = tuple(
        tuple(rng.randrange(1, _COEFFICIENT_LIMIT) for _ in range(m)) for _ in range(m)
    )
    return _Draw(shears, hyperplanes)


def _rows(draw: _Draw, m: int, used: int) -> tuple[tuple[int, ...], ...]:
    """The coordinates z_0, ..., z_{m-1}: the first ``used`` from the shears, the rest the
    variables."""
    rows = []
    for k in range(m):
        row = [0] * m
        row[k] = 1
        if k < used:
            row[k + 1 :] = draw.shears[k]
        rows.append(tuple(row))
    return tuple(rows)


# --- Singular ---------------------------------------------------------------------------------

# satq: I : h^infinity by repeated quotients in the polynomial ring, which finishes where
# elim.lib's sat and saturation in a local ordering stall. isol: whether V(I) has dimension at
# most 0 at the origin, that is, the origin is not in the closure of V(I) less the origin.
# ldim: the dimension of V(I) at the origin, the least number of generic hyperplanes through
# the origin that leave it isolated.
_PROCS = """\
proc satq(ideal I, ideal h)
{
  ideal Q = std(I);
  ideal N = std(quotient(Q, h));
  while (size(reduce(N, Q)) != 0) { Q = N; N = std(quotient(Q, h)); }
  return(Q);
}
proc isol(ideal I)
{
  ideal K = satq(I, maxideal(1));
  int k;
  for (k = 1; k <= ncols(K); k++) { if (jet(K[k], 0) != 0) { return(1); } }
  return(0);
}
proc ldim(ideal I, ideal L)
{
  if (isol(I)) { return(0); }
  ideal C = I;
  int j;
  for (j = 1; j <= ncols(L); j++) { C = C + L[j]; if (isol(C)) { return(j); } }
  return(ncols(L) + 1);
}
"""


def _residue(c: Fraction, p: int) -> int:
    return c.numerator * pow(c.denominator, -1, p) % p


def _linear(coefficients: Sequence[int], first: int) -> str:
    """sum_i c_i x(first + i) for Singular."""
    return "+".join(f"{c}*x({first + i})" for i, c in enumerate(coefficients)) or "0"


def _prelude(germ: _Germ, p: int, draw: _Draw) -> list[str]:
    """Rings S (global) and Lc (local) in x(1), ..., x(m), the procedures, f centred at the
    origin as F in S, the shears SH and the hyperplanes L."""
    m = germ.variables
    terms = []
    for monomial, c in germ.terms:
        powers = [
            f"x({i + 1})^{e}" if e > 1 else f"x({i + 1})" for i, e in enumerate(monomial) if e
        ]
        terms.append("*".join([str(_residue(c, p)), *powers]))
    shears = [_linear(row, k + 2) for k, row in enumerate(draw.shears)] or ["0"]
    return [
        f"ring Lc = {p}, (x(1..{m})), ds;",
        "poly Fl; ideal Gl; ideal Jl;",
        f"ring S = {p}, (x(1..{m})), dp;",
        _PROCS,
        f"poly F = {'+'.join(terms)};",
        f"ideal SH = {', '.join(shears)};",
        f"ideal L = {', '.join(_linear(row, 1) for row in draw.hyperplanes)};",
        "ideal dead = 0;",
        "int m = nvars(S); int k; int i; int s; int sk; int dg; int gv; int g1; int l0; int mu;",
        "ideal J; ideal P; ideal G; poly H;",
    ]


# One level of the flag: the critical dimension sk of f_k (printed as D), then either its
# Milnor number (I) or, after the shear that makes x(k+1) the coordinate z_k, the polar curve G
# and its intersection numbers with V(f_k), V(z_k) and V(df_k/dz_k) (P). A level that breaks a
# condition stops the run; Python reads and checks every number again.
_LEVELS = """\
for (k = 0; k < m; k++) {
  J = dead;
  for (i = k + 1; i <= m; i++) { J = J + diff(F, var(i)); }
  sk = ldim(J, L);
  print("D " + string(k) + " " + string(sk));
  if (k == 0) { s = sk; }
  if (sk != s - k) { break; }
  if (sk == 0) {
    setring Lc; Fl = imap(S, F); Jl = imap(S, dead);
    for (i = k + 1; i <= m; i++) { Jl = Jl + diff(Fl, var(i)); }
    mu = vdim(std(Jl));
    setring S;
    print("I " + string(k) + " " + string(mu));
    break;
  }
  if (k == m - 1) { break; }
  F = subst(F, var(k + 1), var(k + 1) - SH[k + 1]);
  P = dead;
  for (i = k + 2; i <= m; i++) { P = P + diff(F, var(i)); }
  G = satq(P, ideal(diff(F, var(k + 1))));
  dg = 2;
  if (isol(G)) { dg = 0; } else { if (isol(G + L[1])) { dg = 1; } }
  if (dg > 1) { print("P " + string(k) + " 2 -1 -1 -1"); break; }
  setring Lc; Fl = imap(S, F); Gl = imap(S, G);
  gv = vdim(std(Gl + Fl));
  g1 = vdim(std(Gl + var(k + 1)));
  l0 = vdim(std(Gl + diff(Fl, var(k + 1))));
  setring S;
  print("P " + string(k) + " " + string(dg) + " " + string(gv) + " " + string(g1) + " " + string(l0));
  if (gv < 0 || g1 < 0 || l0 < 0) { break; }
  F = subst(F, var(k + 1), 0);
  dead = dead + var(k + 1);
}
"""


def _flag_script(germ: _Germ, p: int, draw: _Draw) -> str:
    return "\n".join([*_prelude(germ, p, draw), _LEVELS, 'print("END");', "quit;"]) + "\n"


def _check_script(
    germ: _Germ, p: int, draw: _Draw, level: int, js: Sequence[int], last: bool
) -> str:
    """The Iomdine-Lê check at ``level``: for each j, lambda^0 of f_level + z_level^j in the
    rotated flag (z_{level+1}, ..., z_n, z_level), printed as J. At the last level of the
    flag it is a Milnor number."""
    lines = _prelude(germ, p, draw)
    for k in range(level):
        lines += [
            f"F = subst(F, var({k + 1}), var({k + 1}) - SH[{k + 1}]);",
            f"F = subst(F, var({k + 1}), 0); dead = dead + var({k + 1});",
        ]
    k = level + 1
    lines.append(f"F = subst(F, var({k}), var({k}) - SH[{k}]);")
    for j in js:
        lines.append(f"H = F + var({k})^{j};")
        if last:
            lines += [
                "setring Lc; Fl = imap(S, H); Jl = imap(S, dead);",
                f"for (i = {k}; i <= m; i++) {{ Jl = Jl + diff(Fl, var(i)); }}",
                f'mu = vdim(std(Jl)); setring S; print("J {j} " + string(mu));',
            ]
            continue
        lines += [
            f"H = subst(H, var({k + 1}), var({k + 1}) - SH[{k + 1}]);",
            f"P = dead + diff(H, var({k}));",
            f"for (i = {k + 2}; i <= m; i++) {{ P = P + diff(H, var(i)); }}",
            f"G = satq(P, ideal(diff(H, var({k + 1}))));",
            "dg = 2;",
            "if (isol(G)) { dg = 0; } else { if (isol(G + L[1])) { dg = 1; } }",
            "l0 = -1;",
            "if (dg <= 1) {",
            "  setring Lc; Fl = imap(S, H); Gl = imap(S, G);",
            f"  l0 = vdim(std(Gl + diff(Fl, var({k + 1})))); setring S;",
            "}",
            f'print("J {j} " + string(l0));',
        ]
    lines += ['print("END");', "quit;"]
    return "\n".join(lines) + "\n"


def _run(script: str, binary: str, timeout: float) -> tuple[str, bool]:
    """What Singular prints for a script, and whether it ran past ``timeout`` seconds.

    Raises
    ------
    ComputationError
        If Singular exits with an error status or reports an error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "milnor.sing"
        path.write_text(script)
        try:
            result = subprocess.run(
                [binary, "-q", "--no-warn", str(path)],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or b""
            return out.decode(errors="replace") if isinstance(out, bytes) else out, True
    # Singular reports an error in the script on stdout, after a question mark.
    error = next(
        (line for line in result.stdout.splitlines() if line.lstrip().startswith("?")), None
    )
    if result.returncode != 0 or error is not None:
        reason = (
            " ".join((error or result.stderr).split())[:100] or f"exit status {result.returncode}"
        )
        raise ComputationError(f"undecided: Singular failed computing Lê numbers: {reason}")
    return result.stdout, False


_LINE = re.compile(r"([DIPJ]) (-?\d+(?: -?\d+)*)")


def _read(text: str) -> dict[str, dict[int, tuple[int, ...]]]:
    """The printed numbers by kind and level (or exponent j for J)."""
    found: dict[str, dict[int, tuple[int, ...]]] = {kind: {} for kind in "DIPJ"}
    for line in text.splitlines():
        match = _LINE.fullmatch(line.strip())
        if match:
            key, *values = (int(v) for v in match.group(2).split())
            found[match.group(1)][key] = tuple(values)
    return found


# --- one flag ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Flag:
    """The outcome of one flag: its Lê numbers, the attaching sum and lambda^0 of each f_k."""

    le: LeNumbers
    attaching: int
    lambda0: tuple[int, ...]
    draw: _Draw


@dataclass(frozen=True)
class _Levels:
    """What a run found: the critical dimension s, (gv, g1, l0) for each level k < s, the
    intersection numbers of the polar curve of f_k with V(f_k), V(z_k) and V(df_k/dz_k), and
    the Milnor number of f_s."""

    s: int
    polar: tuple[tuple[int, int, int], ...]
    mu: int


def _judge(found: dict[str, dict[int, tuple[int, ...]]], n: int) -> _Levels | str:
    """The levels of a finished run, or the condition that failed."""
    dims, polar, iso = found["D"], found["P"], found["I"]
    if 0 not in dims:
        return "Singular printed no critical dimension"
    s = dims[0][0]
    if not 0 <= s <= n:
        return f"the critical locus has dimension {s} at the point"
    levels = []
    for k in range(s + 1):
        if k not in dims:
            return f"level {k}: no critical dimension"
        if dims[k][0] != s - k:
            return (
                f"level {k}: the critical locus of the restriction has dimension {dims[k][0]}, "
                f"not {s - k}"
            )
        if k == s:
            break
        if k not in polar:
            return f"level {k}: no polar curve"
        dg, gv, g1, l0 = polar[k]
        if dg > 1:
            return f"level {k}: the polar curve has dimension 2 or more at the point"
        if min(gv, g1, l0) < 0:
            return f"level {k}: the polar curve meets V(f), V(z) or V(df/dz) in a curve"
        if gv != g1 + l0:
            return f"level {k}: the Teissier split fails, {gv} != {g1} + {l0}"
        levels.append((gv, g1, l0))
    if s not in iso or iso[s][0] < 0:
        return f"level {s}: no finite Milnor number"
    return _Levels(s, tuple(levels), iso[s][0])


def _flag(germ: _Germ, p: int, seed: int, flag: int, binary: str, timeout: float) -> _Flag:
    """Lê numbers along one flag at prime p, with a second set of coordinates if the first
    fails a condition of Def. 3.8.

    Raises
    ------
    ComputationError
        If Singular fails or runs past ``timeout``, or every set of coordinates fails.
    """
    n = germ.variables - 1
    reasons = []
    for attempt in range(_ATTEMPTS):
        draw = _draw(seed, flag, attempt, germ.variables)
        text, timed_out = _run(_flag_script(germ, p, draw), binary, timeout)
        found = _read(text)
        if timed_out:
            where = max(found["D"], default=None)
            at = "" if where is None else f" at level {where} of the flag"
            raise ComputationError(f"undecided: Singular ran past timeout={timeout} s{at}")
        result = _judge(found, n)
        if isinstance(result, str):
            reasons.append(result)
            continue
        s, levels, mu = result.s, result.polar, result.mu
        lambda0 = [l0 for _, _, l0 in levels] + [mu]
        polar = [g1 for _, g1, _ in levels]
        numbers_ = [lambda0[0]] + [lambda0[k] - polar[k - 1] for k in range(1, s + 1)]
        if min(numbers_) < 0:
            reasons.append(f"a negative Lê number {tuple(numbers_)}")
            continue
        # Lê's attaching theorem level by level. With gv = g1 + l0 at every level it equals
        # Thm 4.1's sum_k (-1)^(n-k) lambda^k term by term.
        attaching = sum((-1) ** (n - k) * gv for k, (gv, _, _) in enumerate(levels))
        attaching += (-1) ** (n - s) * mu
        le = LeNumbers(
            numbers=tuple(numbers_),
            polar_numbers=tuple(polar),
            critical_dimension=s,
            flag=_rows(draw, germ.variables, s),
            prime=p,
            method="isolated" if s == 0 else "attaching",
        )
        return _Flag(le, attaching, tuple(lambda0), draw)
    raise ComputationError(
        f"undecided: {_ATTEMPTS} flags of coordinates failed the conditions for Lê numbers ("
        + "; ".join(reasons)
        + ")"
    )


def _submersion(germ: _Germ) -> LeNumbers:
    m = germ.variables
    return LeNumbers(
        numbers=(),
        polar_numbers=(),
        critical_dimension=-1,
        flag=tuple(tuple(int(i == k) for i in range(m)) for k in range(m)),
        prime=None,
        method="submersion",
    )


def _check(germ: _Germ, p: int, found: _Flag, binary: str, timeout: float) -> None:
    """The Iomdine-Lê formula (Massey, Ex. 4.6(2), p. 24) on the flag's own coordinates.

    For f_k, with critical dimension s - k, and j > 1 + lambda^0/gamma^1 (any j >= 2 when
    gamma^1 = 0), lambda^0 of f_k + z_k^j in the rotated flag is
    lambda^0_{f_k} + (j - 1) lambda^1_{f_k}, where lambda^1_{f_k} = lambda^{k+1}_f by
    Ex. 4.6(1). At the last level, k = s - 1, the left side is a Milnor number, a cheap
    check that runs for every s >= 1. When s = 2, level 0 is checked as well if it finishes
    within ``timeout``.

    Raises
    ------
    ComputationError
        If the formula fails, or the last level runs past ``timeout``.
    """
    le = found.le
    s = le.critical_dimension
    for level in [s - 1] + ([0] if s == 2 else []):
        gamma, lam0 = le.polar_numbers[level], found.lambda0[level]
        first = 2 if gamma == 0 else lam0 // gamma + 2
        js = (first, first + 1)
        last = level == s - 1
        text, timed_out = _run(_check_script(germ, p, found.draw, level, js, last), binary, timeout)
        if timed_out:
            if last:
                raise ComputationError(
                    f"undecided: the Iomdine-Lê check at level {level} ran past timeout={timeout} s"
                )
            continue
        got = _read(text)["J"]
        for j in js:
            want = lam0 + (j - 1) * le.numbers[level + 1]
            value = got.get(j, (None,))[0]
            if value != want:
                raise ComputationError(
                    f"undecided: the Iomdine-Lê check fails at level {level}, j = {j}: lambda^0 is "
                    f"{value}, "
                    f"the Lê numbers {le.numbers} give {want}"
                )


# --- public -----------------------------------------------------------------------------------


def _validate(seed: object, timeout: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(f"seed must be an integer, not {seed!r:.60}")
    _check_timeout(timeout, optional=False)


def _binary() -> str:
    binary = _singular_binary()
    if binary is None:
        raise RuntimeError("Lê numbers at a critical point need Singular, which was not found")
    return binary


def _decide(germ: _Germ, seed: int, check: bool, timeout: float) -> _Flag:
    """The first of two flags at the two primes of the germ, which must give the same Lê and
    polar numbers, checked against the Iomdine-Lê formula if ``check``.

    Raises
    ------
    RuntimeError
        If Singular is not installed.
    ComputationError
        If either flag is undecided, the flags disagree, or the check fails.
    """
    binary = _binary()
    primes = _primes(germ, 2)
    first = _flag(germ, primes[0], seed, 0, binary, timeout)
    second = _flag(germ, primes[1], seed, 1, binary, timeout)
    a, b = first.le, second.le
    if (a.numbers, a.polar_numbers) != (b.numbers, b.polar_numbers):
        raise ComputationError(
            f"undecided: two flags give the Lê numbers {a.numbers} and {b.numbers}, polar "
            f"numbers {a.polar_numbers} and {b.polar_numbers}, modulo {primes[0]} and {primes[1]}"
        )
    if check and a.critical_dimension >= 1:
        _check(germ, primes[0], first, binary, timeout)
    return first


def le_numbers(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    at: Sequence[int | Fraction | sp.Rational],
    *,
    seed: int = 0,
    timeout: float = DEFAULT_TIMEOUT,
) -> LeNumbers:
    """The Lê numbers of f at a rational point, with respect to random coordinates.

    For generic coordinates the Lê numbers do not depend on the coordinates
    (Massey, "Non-isolated hypersurface singularities and Lê cycles",
    arXiv:1410.3312, p. 22). Two flags, drawn from ``seed``, are computed at
    the two largest primes below 2^29 that divide no numerator or denominator
    of a coefficient of f centred at the point, one at each. They must give
    the same Lê and polar numbers: flags that disagree leave the germ
    undecided, even when their Euler characteristics agree. The first flag is
    then checked against the Iomdine-Lê formula, as
    :func:`milnor_fibre_euler_characteristic` checks it. The results are
    those of computations over F_p, exact for all but finitely many primes,
    not a certificate. The module docstring describes the method.

    Parameters
    ----------
    polynomial
        f, a polynomial with rational coefficients in ``variables``.
    variables
        The n + 1 variables of f.
    at
        The point, one rational number per variable, where f must vanish.
    seed
        Seed of the random coordinates.
    timeout
        The most seconds each Singular run gets, at most 2,000,000.

    Raises
    ------
    ValidationError
        If ``variables`` are not one or more distinct symbols, f is not a
        non-zero polynomial with rational coefficients in them, ``at`` does
        not give one rational number per variable, f does not vanish there,
        ``seed`` is not an integer, or ``timeout`` is not a number of seconds
        greater than 0 and at most 2,000,000.
    RuntimeError
        If the point is critical and Singular is not installed.
    ComputationError
        If Singular fails or runs past ``timeout``, a flag fails the
        conditions under which Lê numbers exist at two sets of coordinates,
        the two flags disagree, or the Iomdine-Lê check fails; the message
        starts with "undecided: " and gives the reason.
    """
    _validate(seed, timeout)
    germ = _germ(polynomial, variables, at)
    if germ.submersion:
        return _submersion(germ)
    return _decide(germ, seed, True, timeout).le


def milnor_fibre_euler_characteristic(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    at: Sequence[int | Fraction | sp.Rational],
    *,
    seed: int = 0,
    check: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """chi~(F) = chi(F) - 1, F the Milnor fibre of f at a rational point where f vanishes.

    0 where df != 0; (-1)^n mu at an isolated critical point, n + 1 the
    number of variables; otherwise the sum of Lê's attaching theorem along a
    flag of random coordinates (see the module docstring). The flags are
    those of :func:`le_numbers`, computed and compared the same way. Since
    each level checks the Teissier split, the attaching sum equals
    sum_k (-1)^(n-k) lambda^k (Massey, "Non-isolated hypersurface
    singularities and Lê cycles", arXiv:1410.3312, Thm 4.1, p. 22, and
    p. 23). The result is that of computations over F_p,
    exact for all but finitely many primes, not a certificate.

    Parameters
    ----------
    polynomial
        f, a polynomial with rational coefficients in ``variables``.
    variables
        The n + 1 variables of f.
    at
        The point, one rational number per variable, where f must vanish.
    seed
        Seed of the random coordinates; :func:`le_numbers` with the same seed
        computes the same flags.
    check
        Whether to check the first flag against the Iomdine-Lê formula
        (Massey, Ex. 4.6(2), p. 24): at the last level of the flag, where it
        is a Milnor number, whenever the critical locus has dimension s >= 1,
        and when s = 2 also at the first level, unless that run passes
        ``timeout``.
    timeout
        The most seconds each Singular run gets, at most 2,000,000.

    Raises
    ------
    ValidationError
        As :func:`le_numbers`.
    RuntimeError
        If the point is critical and Singular is not installed.
    ComputationError
        If Singular fails or runs past ``timeout``, a flag fails the
        conditions under which Lê numbers exist at two sets of coordinates,
        the two flags disagree, or the Iomdine-Lê check fails; the message
        starts with "undecided: " and gives the reason.
    """
    _validate(seed, timeout)
    germ = _germ(polynomial, variables, at)
    if germ.submersion:
        return 0
    return _decide(germ, seed, check, timeout).attaching
