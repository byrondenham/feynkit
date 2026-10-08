"""
Euler characteristics of subvarieties of the torus.

:func:`torus_euler_characteristic` computes chi((V(I) cap T) minus V(h)), the topological
Euler characteristic of the closed subvariety V(I) of the torus T = (C^*)^d with the
hypersurface V(h) removed.

Let A be a closed irreducible subvariety of T of dimension k and h a function that does not
vanish identically on A. If A is smooth, the very affine variety A minus V(h) has
(-1)^k chi(A minus V(h)) critical points of t^b h^(-s) for generic exponents b and s, counted
with multiplicity (J. Huh, "The maximum likelihood degree of a very affine variety", Compos.
Math. 149 (2013) 1245-1266, Thm 1(iii); p. 2 of the arXiv version, arXiv:1207.0553v3). For singular A let u be a combination of the minors of the
Jacobian matrix that cut out Sing A, with random coefficients. Then A minus V(hu) is smooth, and

    chi(A minus V(h)) = (-1)^k #crit(t^b h^(-s_1) u^(-s_2)) on A minus V(hu)
                        + chi((A cap V(u)) minus V(h)),

the second term being of lower dimension (additivity of chi over the closed set V(u)). The
critical points are the points of A where the log-differential of t^b h^(-s_1) u^(-s_2) lies in
the span of the differentials of the generators of A: Lagrange multipliers where A is cut out by
codim A generators, the minors of size codim A + 1 of the Jacobian matrix, bordered by that
differential, otherwise. A Rabinowitsch variable w with 1 - w t_1 ... t_d h u = 0 removes the
points on the coordinate hyperplanes, V(h) and V(u). A closed set that is not irreducible is
cut into its prime components, and chi of the union follows by inclusion and exclusion over
their intersections. A point set counts the points off V(h).

The computation runs over a prime field F_p, at the two largest primes below 2^29 that divide
no numerator or denominator of a coefficient of the input, once for each prime, and the two
values must agree. The decompositions are Singular's (minAssGTZ), the exponents and the
coefficients of u are drawn from a seed, and the critical points are counted by the dimension of
the quotient ring, with multiplicity: by Singular's vdim, or with ``backend="msolve"`` by
msolve. Like :func:`~feynkit.point_count.critical_point_count`, the result holds for all but
finitely many primes and generic exponents, so it is a cross-check, not a certificate. The
Euler characteristics of the components of a variety over F_p stand in for those over Q.

Every ComputationError raised for an input the computation cannot decide has a message that
starts with "undecided: " and goes on with the reason.
"""

from __future__ import annotations

import math
import random
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from .core.exceptions import ComputationError, ValidationError
from .landau import _check_timeout, _singular_binary
from .point_count import _msolve_binary, _msolve_count, _singular_polynomial

__all__ = ["torus_euler_characteristic"]

# Primes stay below 2^29, where Singular's prime fields work in every ring and msolve 0.10.1
# (which fails above about 1.5e9) runs in little memory.
_PRIME_LIMIT = 2**29

_BACKENDS = ("singular", "msolve")

_HEADER = 'LIB "primdec.lib";\nLIB "elim.lib";\n'


@dataclass(frozen=True)
class _Component:
    """A prime component of a closed set of the torus, over F_p."""

    gens: str
    dim: int
    count: int  # the number of generators in ``gens``
    key: str  # the reduced Gröbner basis, the same for equal sets


def _undecided(reason: str) -> ComputationError:
    return ComputationError(f"undecided: {reason}")


class _Run:
    """The computation at one prime."""

    def __init__(
        self,
        *,
        prime: int,
        dimension: int,
        removed: str,
        seed: int,
        backend: str,
        timeout: float,
        singular: str,
        msolve: str | None,
    ) -> None:
        self.prime = prime
        self.d = dimension
        self.names = [f"v{i}" for i in range(dimension)]
        self.product = "*".join(self.names)
        self.removed = removed
        self.rng = random.Random(f"{seed}:{prime}")
        self.backend = backend
        self.timeout = timeout
        self.singular = singular
        self.msolve = msolve

    # -- running the solvers ------------------------------------------------------------------

    def ring(self) -> str:
        return f"ring R = {self.prime}, ({','.join(self.names)}), dp;\n"

    def singular_output(self, script: str) -> str:
        """What Singular prints for the script.

        Raises
        ------
        ComputationError
            If Singular runs past the timeout, exits with an error status or reports an error.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strata.sing"
            path.write_text(_HEADER + self.ring() + script + "quit;\n")
            try:
                run = subprocess.run(
                    [self.singular, "-q", "--no-warn", str(path)],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                    timeout=self.timeout,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired as exc:
                raise _undecided(f"Singular ran past timeout={self.timeout} s") from exc
        # Singular reports an error in the script on stdout, after a question mark.
        error = next(
            (line for line in run.stdout.splitlines() if line.lstrip().startswith("?")), None
        )
        if run.returncode != 0 or error is not None:
            reason = (
                " ".join((error or run.stderr).split())[:100] or f"exit status {run.returncode}"
            )
            raise _undecided(f"Singular failed: {reason}")
        return run.stdout

    def msolve_count(self, names: Sequence[str], system: str) -> int:
        """The number of solutions of the system modulo the prime, with multiplicity.

        Raises
        ------
        ComputationError
            If msolve runs past the timeout, the solutions are not a finite set, or msolve
            fails or prints something else.
        """
        assert self.msolve is not None
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "strata.ms"
            target = Path(tmp) / "strata.out"
            lines = system.replace(",", ",\n")
            source.write_text(f"{','.join(names)}\n{self.prime}\n{lines}\n")
            try:
                run = subprocess.run(
                    [self.msolve, "-t", "1", "-f", str(source), "-o", str(target)],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                    timeout=self.timeout,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired as exc:
                raise _undecided(f"msolve ran past timeout={self.timeout} s") from exc
            output = target.read_text() if target.exists() else ""
        if run.returncode != 0 or not output.strip():
            shown = " ".join((run.stderr or run.stdout or "nothing").split())[:100]
            raise _undecided(f"msolve failed: {shown}")
        try:
            count = _msolve_count(output)
        except RuntimeError as exc:
            raise _undecided(f"msolve printed an unexpected result: {exc}") from exc
        if count < 0:
            raise _undecided("the critical points do not form a finite set")
        return count

    # -- decomposition ------------------------------------------------------------------------

    def components(self, gens: str) -> list[_Component]:
        """The prime components over F_p of V(gens) in the torus."""
        out = self.singular_output(
            f"ideal Z = {gens};\n"
            f"Z = sat(Z, ideal({self.product}));\n"
            'if (dim(std(Z)) < 0) { print("EMPTY"); quit; }\n'
            "list L = minAssGTZ(Z);\n"
            "ideal C;\n"
            "option(redSB);\n"
            "for (int i = 1; i <= size(L); i++) {\n"
            "  C = simplify(L[i], 2);\n"
            '  print("COMP " + string(dim(std(C))) + " " + string(size(C)) + " " + string(std(C)) + " ; " + string(C));\n'
            "}\n"
        )
        if "EMPTY" in out:
            return []
        found = [
            _Component(m.group(4).strip(), int(m.group(1)), int(m.group(2)), m.group(3).strip())
            for m in re.finditer(r"^COMP (-?\d+) (\d+) (.*) ; (.*)$", out, re.MULTILINE)
        ]
        if not found:
            raise _undecided(f"Singular printed {out.strip()[:100]!r} instead of components")
        return found

    # -- Euler characteristics ----------------------------------------------------------------

    def closed(self, gens: str) -> int:
        """chi(V(gens) minus V(h)) in the torus."""
        return self.union(self.components(gens))

    def union(self, family: Sequence[_Component]) -> int:
        """chi of the union of the sets V(A) in the family, minus V(h).

        The union is the disjoint union of the sets A_i minus the earlier A_j, and chi is
        additive over disjoint unions of constructible pieces, so
        chi(union) = sum_i [chi(A_i) - chi(A_i cap (A_1 cup ... cup A_{i-1}))], the last set
        being the union of the components of the intersections A_i cap A_j, handled the same
        way. A member inside an earlier one adds nothing. Larger sets come first, so each
        intersection has smaller dimension than A_i unless A_i lies inside A_j.
        """
        ordered = sorted({c.key: c for c in family}.values(), key=lambda c: -c.dim)
        total = 0
        for i, comp in enumerate(ordered):
            pool: dict[str, _Component] = {}
            inside = False
            for earlier in ordered[:i]:
                found = self.components(f"{comp.gens},{earlier.gens}")
                if any(c.key == comp.key for c in found):
                    inside = True
                    break
                pool.update((c.key, c) for c in found)
            if inside:
                continue
            total += self.open(comp)
            if pool:
                total -= self.union(list(pool.values()))
        return total

    def open(self, comp: _Component) -> int:
        """chi(A minus V(h)) for the irreducible A = V(comp) of the torus."""
        d, k = self.d, comp.dim
        c = d - k
        prime, rng = self.prime, self.rng
        exponents = [rng.randrange(1, prime) for _ in range(d)]
        s1, s2 = rng.randrange(1, prime), rng.randrange(1, prime)
        script = (
            f"ideal A = {comp.gens};\nint m = size(A);\npoly h = {self.removed};\nint i;\nint j;\n"
            "ideal sA = std(A);\n"
            'if (reduce(h, sA) == 0) { print("VANISH"); quit; }\n'
        )
        if k == 0:
            script += 'ideal Zh = sat(A, ideal(h));\nprint("POINTS " + string(vdim(std(Zh))));\n'
            return self._read_points(self.singular_output(script))
        # u, a random combination of the minors that cut out the singular locus.
        if c == 0:
            script += "ideal MI = 1;\n"
        else:
            script += (
                f"matrix JC[m][{d}];\n"
                f"for (i = 1; i <= m; i++) {{ for (j = 1; j <= {d}; j++) "
                "{ JC[i, j] = diff(A[i], var(j)); } }\n"
                f"ideal MI = minor(JC, {c});\n"
            )
        most = math.comb(comp.count, c) * math.comb(d, c) if c else 1
        coefficients = ",".join(str(rng.randrange(1, prime)) for _ in range(most))
        script += (
            f"ideal SA = sat(A + MI, ideal({self.product}));\n"
            "poly u = 1;\n"
            "if (dim(std(SA)) >= 0) {\n"
            f"  intvec cf = {coefficients};\n"
            "  u = 0;\n"
            "  for (i = 1; i <= ncols(MI); i++) { u = u + cf[i] * MI[i]; }\n"
            "  u = reduce(u, sA);\n"
            '  if (u == 0) { print("UZERO"); quit; }\n'
            "}\n"
            "ideal AU = 1;\n"
            "if (u != 1) { AU = sat(A + ideal(u), ideal(" + self.product + ")); }\n"
            'print("AU " + string(AU));\n'
            'print("AUDIM " + string(dim(std(AU))));\n'
        )
        # The critical system, in a ring that adds the multipliers and the Rabinowitsch variable.
        lagrange = comp.count == c
        extra = [f"l{i}" for i in range(c)] if lagrange else []
        system_names = [*self.names, *extra, "w"]
        script += f"ring S = {prime}, ({','.join(system_names)}), dp;\n"
        script += "ideal A = imap(R, A);\npoly h = imap(R, h);\npoly u = imap(R, u);\n"
        script += "ideal G = A;\n"
        border = [
            f"{exponents[j]}*h*u - {s1}*{self.names[j]}*diff(h, {self.names[j]})*u"
            f" - {s2}*{self.names[j]}*diff(u, {self.names[j]})*h"
            for j in range(d)
        ]
        if lagrange:
            for j in range(d):
                terms = "".join(
                    f" - {extra[i]}*{self.names[j]}*diff(A[{i + 1}], {self.names[j]})"
                    for i in range(c)
                )
                script += f"G = G + ideal({border[j]}{terms});\n"
        else:
            script += f"matrix M[{comp.count + 1}][{d}];\n"
            for i in range(comp.count):
                for j in range(d):
                    script += f"M[{i + 1}, {j + 1}] = {self.names[j]}*diff(A[{i + 1}], {self.names[j]});\n"
            for j in range(d):
                script += f"M[{comp.count + 1}, {j + 1}] = {border[j]};\n"
            script += f"G = G + minor(M, {c + 1});\n"
        script += f"G = G + ideal(1 - w*{self.product}*h*u);\n"
        if self.backend == "singular":
            script += (
                'ideal GS = std(G);\nprint("CRIT " + string(dim(GS)) + " " + string(vdim(GS)));\n'
            )
        else:
            script += 'print("SYSTEM " + string(simplify(G, 2)));\n'
        out = self.singular_output(script)
        if "VANISH" in out:
            return 0
        if "UZERO" in out:
            raise _undecided("the random combination of minors vanished on a component")
        count = self._read_count(out, system_names)
        value = -count if k % 2 else count
        rest = re.search(r"^AU (.*)$", out, re.MULTILINE)
        rest_dim = re.search(r"^AUDIM (-?\d+)$", out, re.MULTILINE)
        if rest is None or rest_dim is None:
            raise _undecided(f"Singular printed {out.strip()[:100]!r} instead of the cut by u")
        if int(rest_dim.group(1)) >= k:
            raise _undecided("the random combination of minors vanished on a component")
        if rest.group(1).strip() != "1":
            value += self.closed(rest.group(1).strip())
        return value

    def _read_points(self, out: str) -> int:
        if "VANISH" in out:
            return 0
        found = re.search(r"^POINTS (\d+)$", out, re.MULTILINE)
        if found is None:
            raise _undecided(f"Singular printed {out.strip()[:100]!r} instead of a point count")
        return int(found.group(1))

    def _read_count(self, out: str, names: Sequence[str]) -> int:
        """The number of critical points in the output of the system script."""
        if self.backend == "singular":
            found = re.search(r"^CRIT (-?\d+) (\d+)$", out, re.MULTILINE)
            if found is None:
                raise _undecided(f"Singular printed {out.strip()[:100]!r} instead of a count")
            if found.group(1) not in ("0", "-1"):
                raise _undecided("the critical points do not form a finite set")
            return int(found.group(2))
        system = re.search(r"^SYSTEM (.*)$", out, re.MULTILINE)
        if system is None:
            raise _undecided(f"Singular printed {out.strip()[:100]!r} instead of a system")
        return self.msolve_count(names, system.group(1).strip())


def _validate(
    generators: object, variables: object, remove: object, seed: object, backend: object
) -> tuple[list[sp.Poly], tuple[sp.Symbol, ...], sp.Poly | None]:
    if backend not in _BACKENDS:
        raise ValidationError(f"backend must be 'singular' or 'msolve', not {backend!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError(f"seed must be an integer, not {seed!r:.60}")
    if (
        not isinstance(variables, Sequence)
        or isinstance(variables, str)
        or not variables
        or not all(isinstance(x, sp.Symbol) for x in variables)
        or len(set(variables)) != len(variables)
    ):
        raise ValidationError("variables must be one or more distinct symbols")
    names = tuple(variables)
    if isinstance(generators, str) or not isinstance(generators, Sequence):
        raise ValidationError("generators must be a sequence of polynomials")

    def poly(expr: object, what: str) -> sp.Poly:
        try:
            value = sp.sympify(expr)
        except (sp.SympifyError, TypeError) as exc:
            raise ValidationError(f"{what} must be a polynomial, not {expr!r:.60}") from exc
        if value.free_symbols - set(names):
            raise ValidationError(f"{what} has symbols other than the variables")
        try:
            return sp.Poly(sp.expand(value), *names, domain="QQ")
        except (sp.PolynomialError, sp.CoercionFailed) as exc:
            raise ValidationError(
                f"{what} must be a polynomial with rational coefficients in the variables"
            ) from exc

    polys = [poly(g, "a generator") for g in generators]
    removed = None if remove is None else poly(remove, "remove")
    return polys, names, removed


def torus_euler_characteristic(
    generators: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    *,
    remove: sp.Expr | None = None,
    seed: int = 0,
    backend: str = "singular",
    timeout: float = 300,
) -> int:
    """chi((V(I) cap T) minus V(h)), the Euler characteristic of a subvariety of the torus.

    T = (C^*)^d, I is the ideal generated by ``generators`` and h is ``remove``. The Euler
    characteristic is the topological one, with compact support or not alike for complex
    varieties. With no generators, or generators that are all 0, V(I) is the torus. Where
    nothing of V(I) lies in the torus, or all of it lies in V(h), the result is 0.

    The computation follows the formula in the module docstring, after J. Huh, Compos. Math.
    149 (2013) 1245-1266, Thm 1(iii) (p. 2 of the arXiv version): the number of critical points of a generic master
    function on the smooth part, plus the Euler characteristic of the lower-dimensional locus
    where a combination of the minors of the Jacobian matrix vanishes. For almost all Laurent
    polynomials f in d variables whose support lies in a polytope Gamma, A. G. Kouchnirenko,
    Invent. Math. 32 (1976) 1-31, Thm IV, p. 30, gives chi(V(f) cap T) = (-1)^(d-1) d!
    Vol(Gamma), the normalised volume of Gamma.

    Singular decomposes and counts over F_p at the two largest primes below 2^29 that divide no
    numerator or denominator of a coefficient, and the two values must agree. With
    ``backend="msolve"`` the critical points are counted by msolve instead, at the same primes;
    the decompositions stay in Singular.

    Parameters
    ----------
    generators
        Polynomials in ``variables`` with rational coefficients.
    variables
        The d coordinates of the torus, distinct symbols.
    remove
        The polynomial h, or None to remove nothing.
    seed
        Seed of the random exponents and combinations.
    timeout
        The most seconds to give each Singular or msolve run, at most 2,000,000.
    backend
        "singular" or "msolve".

    Raises
    ------
    RuntimeError
        If the backend, or Singular, is not installed.
    ValidationError
        If ``backend`` is neither "singular" nor "msolve", ``variables`` are not one or more
        distinct symbols, ``seed`` is not an integer, ``timeout`` is not a number greater than
        0 and at most 2,000,000, or a generator or ``remove`` is not a polynomial with
        rational coefficients in the variables.
    ComputationError
        With a message that starts with "undecided: ": if a solver runs past the timeout or
        fails, the two primes give different values, or the critical points do not form a
        finite set.
    """
    polys, names, removed = _validate(generators, variables, remove, seed, backend)
    _check_timeout(timeout, optional=False)
    singular = _singular_binary()
    if singular is None:
        raise RuntimeError("torus_euler_characteristic needs Singular, which was not found")
    msolve = _msolve_binary() if backend == "msolve" else None
    if backend == "msolve" and msolve is None:
        raise RuntimeError("torus_euler_characteristic with backend='msolve' needs msolve")
    if removed is not None and removed.is_zero:
        return 0
    if removed is not None and removed.is_ground:
        removed = None
    polys = [poly for poly in polys if not poly.is_zero]
    if any(poly.is_ground for poly in polys):
        return 0
    integral = [poly.clear_denoms(convert=True)[1] for poly in polys]
    h_integral = None if removed is None else removed.clear_denoms(convert=True)[1]
    avoided = {
        abs(int(n))
        for q in [c for poly in [*polys, *([removed] if removed else [])] for c in poly.coeffs()]
        for n in (q.p, q.q)
    }
    primes: list[int] = []
    p = _PRIME_LIMIT
    while len(primes) < 2:
        p = sp.prevprime(p)
        if all(n % p for n in avoided):
            primes.append(p)

    values = []
    for prime in primes:
        run = _Run(
            prime=prime,
            dimension=len(names),
            removed="1" if h_integral is None else _singular_polynomial(h_integral, prime),
            seed=seed,
            backend=backend,
            timeout=timeout,
            singular=singular,
            msolve=msolve,
        )
        gens = ",".join(_singular_polynomial(poly, prime) for poly in integral) or "0"
        values.append(run.closed(gens))
    if values[0] != values[1]:
        raise _undecided(
            f"the Euler characteristic is {values[0]} modulo {primes[0]} and {values[1]} "
            f"modulo {primes[1]}"
        )
    return values[0]
