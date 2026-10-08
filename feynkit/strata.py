"""
Euler characteristics of subvarieties of the torus, and the strata of the singular locus of a
face polynomial.

:func:`singular_strata` and :func:`singular_strata_from_polynomial` cut the singular locus of
the polynomial on the orbit of every face of the Newton polytope into locally closed pieces on
which the torus-cut Milnor number is constant; see :class:`StrataAnalysis`. Faces whose
normal cone is not smooth are handled through a smooth subdivision of the cone.
:func:`stratum_sum` and :func:`stratum_sum_from_polynomial` add the Euler characteristics of the
pieces, cut with a generic hypersurface, weigh them by mu^T and compare the sum with the drop of
the number of critical points below the volume of the Newton polytope; see :class:`StratumSum`.

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

import dataclasses
import itertools
import math
import random
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import sympy as sp

from .core.exceptions import ComputationError, ValidationError
from .degeneracy import _support
from .landau import _check_timeout, _singular_binary
from .lattice_invariants import lattice_points
from .milnor import (
    NumberField,
    TorusCutMilnorNumber,
    _field,
    _smooth_chart,
    torus_cut_milnor_number,
)
from .point_count import (
    _msolve_binary,
    _msolve_count,
    _singular_polynomial,
    critical_point_count,
)
from .polytope import PolytopeData, lattice_coordinates, polytope_data
from .toric import (
    NormalCone,
    OrbitChart,
    normal_cone,
    orbit_chart,
    orbit_cones,
    smooth_subdivision,
)

if TYPE_CHECKING:
    from .integral import FeynmanIntegral

__all__ = [
    "ConditionFailure",
    "StrataAnalysis",
    "Stratum",
    "StratumSum",
    "singular_strata",
    "singular_strata_from_polynomial",
    "stratum_sum",
    "stratum_sum_from_polynomial",
    "torus_euler_characteristic",
]

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


def _singular_run(text: str, singular: str, timeout: float) -> str:
    """What Singular prints for the script text.

    Raises
    ------
    ComputationError
        If Singular runs past the timeout, exits with an error status or reports an error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "strata.sing"
        path.write_text(text)
        try:
            run = subprocess.run(
                [singular, "-q", "--no-warn", str(path)],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired as exc:
            raise _undecided(f"Singular ran past timeout={timeout} s") from exc
    # Singular reports an error in the script on stdout, after a question mark.
    error = next((line for line in run.stdout.splitlines() if line.lstrip().startswith("?")), None)
    if run.returncode != 0 or error is not None:
        reason = " ".join((error or run.stderr).split())[:100] or f"exit status {run.returncode}"
        raise _undecided(f"Singular failed: {reason}")
    return run.stdout


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
        return _singular_run(
            _HEADER + self.ring() + script + "quit;\n", self.singular, self.timeout
        )

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

    def transverse(self, gens: str, dimension: int, removed: Sequence[str]) -> bool:
        """Whether d(h) restricted to the piece never vanishes on the piece and V(h).

        The piece is V(gens), of the given dimension, less the sets V(R) for R in ``removed``,
        which lie inside it. It is smooth, so its Jacobian matrix has rank equal to its
        codimension c at its points, and d(h) restricted to it vanishes at a point of V(h)
        exactly when the Jacobian matrix of the generators and h has rank at most c there:
        when its minors of size c + 1 vanish. The piece is transverse to V(h) when no point
        of the piece satisfies this.
        """
        codim = self.d - dimension
        script = (
            f"ideal G = {gens};\npoly h = {self.removed};\nideal I = G, h;\n"
            "matrix J = jacob(I);\n"
            f"if (nrows(J) >= {codim + 1} && ncols(J) >= {codim + 1}) "
            f"{{ I = I, minor(J, {codim + 1}); }}\n"
            f"I = sat(I, ideal({self.product}));\n"
        )
        for ideal in removed:
            script += f"I = sat(I, ideal({ideal}));\n"
        script += 'if (dim(std(I)) < 0) { print("TRANSVERSE"); } else { print("NOT"); }\n'
        out = self.singular_output(script)
        if "TRANSVERSE" in out:
            return True
        if "NOT" in out:
            return False
        raise _undecided(f"Singular printed {out.strip()[:100]!r} instead of a verdict")

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


def _two_primes(polys: Sequence[sp.Poly]) -> list[int]:
    """The two largest primes below 2^29 that divide no numerator or denominator of a
    coefficient of the polynomials."""
    avoided = {abs(int(n)) for poly in polys for q in poly.coeffs() for n in (q.p, q.q)}
    primes: list[int] = []
    p = _PRIME_LIMIT
    while len(primes) < 2:
        p = sp.prevprime(p)
        if all(n % p for n in avoided):
            primes.append(p)
    return primes


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
    primes = _two_primes([*polys, *([removed] if removed else [])])

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


# --- singular strata ---------------------------------------------------------------------------

# The most members a face may have before the closure gives up.
_MAX_MEMBERS = 120

# How many random fibres are tried when looking for rational points on a member.
_POINT_TRIES = 15


@dataclass(frozen=True)
class Stratum:
    """A locally closed piece of the singular locus of a face polynomial.

    Let F be a face of the Newton polytope P of G, O_F its torus orbit, with coordinates
    t_1, ..., t_d in the chart of :class:`~feynkit.toric.OrbitChart`, and Sing_F the singular
    locus of the face polynomial in O_F. The piece is the closure V(generators) in O_F, less
    the closures V(removed) of the smaller pieces inside it, so that mu^T is constant on it.

    Attributes
    ----------
    face
        The point indices of F, as in :attr:`StrataAnalysis.points`.
    dimension
        The dimension of the closure, or None if the face itself is undecided.
    generators
        Generators over Q of the prime ideal of the closure in the coordinates t_1, ..., t_d,
        as symbols t1, ..., td; empty if the face is undecided.
    degree
        The degree of the closure: the number of points of a closure of dimension 0, and for
        higher dimension the multiplicity of its ideal for a degree ordering. None if the face
        is undecided.
    removed
        The generators of each closed set removed from the closure.
    mu_t
        mu^T on the piece, (-1)^(N-1) beta, N the dimension of P, and None if undecided. See
        :class:`~feynkit.milnor.TorusCutMilnorNumber`.
    euler
        chi of the piece minus V(H), in the chart torus, as :func:`stratum_sum` fills it in
        for the pieces on which mu^T is decided and not 0; None otherwise, and always None in
        a :class:`StrataAnalysis`.
    chart
        The chart of the face in which the piece was computed, None if there is none.
    reason
        Why the piece is undecided, or None. When set, ``mu_t`` is None.
    method
        "smooth chart" if the normal cone of the face is smooth and mu^T comes from the chart
        of the face, "subdivision" if it is not and mu^T is the sum over the orbits of a
        smooth subdivision of the cone, pushed forward to the orbit of the face; None where the
        face is undecided before either is chosen. On a subdivided face only the pieces of
        dimension 0 can be decided: the candidates for the jumps of beta come from the corner
        charts of the subdivision only, and jumps inside the exceptional fibre are not yet
        located, so every piece of positive dimension carries a reason.
    """

    face: tuple[int, ...]
    dimension: int | None
    generators: tuple[sp.Expr, ...]
    degree: int | None
    removed: tuple[tuple[sp.Expr, ...], ...]
    mu_t: int | None
    euler: int | None
    chart: OrbitChart | None
    reason: str | None
    method: str | None = None


@dataclass(frozen=True)
class ConditionFailure:
    """A member and coordinate subset at which the criterion of :class:`StrataAnalysis` fails.

    Attributes
    ----------
    face
        The point indices of the face.
    dimension
        The dimension of the member.
    generators
        Generators of the prime ideal of the member, as in :class:`Stratum`.
    coordinates
        The set I of the normal coordinates, counted from 0, that are set to 0.
    reason
        What fails.
    """

    face: tuple[int, ...]
    dimension: int
    generators: tuple[sp.Expr, ...]
    coordinates: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class StrataAnalysis:
    """The singular strata of every face of the Newton polytope, from
    :func:`singular_strata_from_polynomial`.

    For each face F of the Newton polytope P of G, the top face included, the singular locus of
    the face polynomial in the orbit O_F is cut into locally closed pieces on which the
    torus-cut Milnor number mu^T of :func:`~feynkit.milnor.torus_cut_milnor_number` is
    constant, with its value on each. Faces whose singular locus is empty have none.

    The pieces come from members: the prime components of the singular locus, and, level by
    level, the components of the jump candidates of the members of positive dimension and of
    the intersections of the members. For a member A and a subset I of the normal coordinates
    y of the chart, let f_I be the local equation g restricted to y_I = 0. The candidates
    follow the jump loci of the Euler characteristic of the Milnor fibre of f_I along A, after
    Massey's Le cycles: the singular locus of A; A meets the critical locus of f_I; where A is
    inside it, the singular loci of the components of the critical locus through A, the other
    components, and the traces on A of the polar varieties and Le cycles of f_I for a random
    flag of coordinates. beta is computed at two points of each member outside the smaller members, which must
    agree: rational points, and where a member has fewer than two, points with coordinates in
    a number field (a root of a component of a slice, as in :class:`~feynkit.milnor.NumberField`);
    a member that is a single point is computed twice, with two seeds, and a member joins the piece of the smallest member containing it, when that is
    unique, if they have the same value. Where the critical locus has a component through A
    that is singular along A, or the polar variety of that component contains A, the
    condition below fails.

    ``complete`` is True exactly when every member of positive dimension satisfies the
    following condition for every subset I of the normal coordinates, every chart used is
    smooth, and no piece is undecided. Let Crit(f_I) be the critical locus of f_I in the
    torus. The member A, of dimension k, satisfies it for I when
    A is not inside Crit(f_I); or A is an irreducible component of Crit(f_I); or every
    irreducible component of Crit(f_I) through A is smooth at the generic point of A and
    either has dimension at most k + 1, or has dimension s > k + 1 with A outside the polar
    variety of f_I of dimension s for two random flags. This is feynkit's sufficient criterion
    for the candidate set to contain every jump of the Euler characteristic of the Milnor
    fibre of f_I along A; with it beta is constant on every piece. It rests on the Le cycles
    and Le numbers of D. B. Massey (Le Cycles and Hypersurface Singularities, Lecture Notes in
    Mathematics 1615, Springer 1995, Def. 1.26, p. 26, on prepolar coordinates and Thm 1.28,
    p. 27, on their existence for generic coordinates; Non-isolated hypersurface singularities
    and Le cycles, arXiv:1410.3312, Thm 2.23, p. 14, and Thm 4.1, p. 22, on the Milnor fibre
    as a complex with cells attached in numbers given by the Le numbers). It is a criterion for
    generic flags over Q, applied with one random flag per pair.

    Attributes
    ----------
    points
        The exponent vectors of the support, in the order of the point indices that name faces.
    strata
        The pieces, by face and then by decreasing dimension.
    primes
        The primes of the Le computations behind the values of mu^T, in increasing order. The
        decompositions are exact, over Q.
    seed
        The seed.
    complete
        As above.
    failures
        The members and subsets at which the condition fails, and so why ``complete`` is
        False when it is not because a piece is undecided.
    subdivided
        The faces whose normal cone is not smooth and whose face polynomial is singular in
        the orbit, which were computed through a smooth subdivision of the cone. For them
        the candidate set is the union of the candidates of the charts of the pieces, the
        condition above is not checked, and ``complete`` is False.
    """

    points: tuple[tuple[int, ...], ...]
    strata: tuple[Stratum, ...]
    primes: tuple[int, ...]
    seed: int
    complete: bool
    failures: tuple[ConditionFailure, ...] = ()
    subdivided: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _Comp:
    """A prime component over Q of a closed set of a torus."""

    gens: str
    dim: int
    degree: int
    key: str


@dataclass
class _Member:
    comp: _Comp
    beta: int | None = None
    primes: tuple[int, ...] = ()
    reason: str | None = None
    points_used: int = 0


def _poly(terms: Mapping[tuple[int, ...], Fraction], names: Sequence[str]) -> str:
    """The polynomial with the given exponents and coefficients, as Singular reads it."""
    pieces = []
    for exponent, c in terms.items():
        monomial = "*".join(f"{n}^{e}" for n, e in zip(names, exponent, strict=True) if e)
        coefficient = f"({c.numerator}/{c.denominator})"
        pieces.append(f"{coefficient}*{monomial}" if monomial else coefficient)
    return "+".join(pieces) or "0"


def _parse(gens: str, names: Sequence[str]) -> tuple[sp.Expr, ...]:
    """The comma-separated polynomials Singular printed, as expressions in symbols of names."""
    local = {n: sp.Symbol(n) for n in names}
    out = []
    depth, start = 0, 0
    for i, ch in enumerate(gens + ","):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            text = gens[start:i].strip()
            start = i + 1
            if text:
                out.append(sp.parse_expr(text.replace("^", "**"), local_dict=local))
    return tuple(out)


@dataclass(frozen=True)
class _Point:
    """A point of a member: rational coordinates, or polynomials in the generator a of a
    number field."""

    coordinates: tuple[Fraction | sp.Expr, ...]
    field: NumberField | None = None


_GENERATOR = sp.Dummy("a")

_CORNER_ONLY = (
    "on a subdivided face the candidates come from the corner charts only, and jumps of the "
    "fibre integral inside the exceptional fibre are not yet located"
)

# The most random linear forms tried when looking for a primitive element of a set of points.
_SHAPE_TRIES = 6


def _random_flag(rng: random.Random, n: int) -> str:
    return ",".join(str(rng.randint(1, 997)) for _ in range(n * n))


class _Face:
    """The strata of one face in a smooth chart C^r_y x (C^*)^d_t, or in several charts over
    the same torus t, the pieces of a subdivision.

    ``sections`` are pairs (r, local): the number of normal coordinates and the local equation
    of each chart. The face polynomial and the members come from the first, the candidates from
    all of them. ``evaluate(point, seed)`` gives beta at a point of a member.
    """

    def __init__(
        self,
        *,
        key: tuple[int, ...],
        chart: OrbitChart | None,
        d: int,
        sections: Sequence[tuple[int, Mapping[tuple[int, ...], Fraction]]],
        evaluate: Callable[[_Point, int], TorusCutMilnorNumber],
        sign: int,
        seed: int,
        timeout: float,
        singular: str,
        salt: str = "",
        method: str = "smooth chart",
    ) -> None:
        self.method = method
        self.key = key
        self.chart = chart
        self.d = d
        self.sections = []
        for r, local in sections:
            shift = [max(0, -min(e[a] for e in local)) for a in range(d)] if local else [0] * d
            self.sections.append(
                (
                    r,
                    {
                        tuple(e[a] + shift[a] for a in range(d)) + tuple(e[d:]): c
                        for e, c in local.items()
                    },
                )
            )
        self.r, self.g = self.sections[0]
        self.tnames = [f"t{i + 1}" for i in range(self.d)]
        self.product = "*".join(self.tnames)
        self.evaluate = evaluate
        self.salt = salt
        self.sign = sign
        self.seed = seed
        self.timeout = timeout
        self.singular = singular
        self.failures: list[ConditionFailure] = []
        self.members: list[_Member] = []
        self.candidate_reasons: dict[int, list[str]] = {}

    # -- Singular ------------------------------------------------------------------------------

    def run(self, script: str, names: Sequence[str] | None = None) -> str:
        ring = f"ring R = 0, ({','.join(names or self.tnames)}), dp;\n"
        return _singular_run(_HEADER + ring + script + "quit;\n", self.singular, self.timeout)

    def components(self, gens: str, prefix: str = "") -> list[_Comp]:
        """The prime components over Q of V(gens) in the torus of t."""
        out = self.run(
            f"{prefix}ideal Z = {gens};\n"
            f"Z = sat(Z, ideal({self.product}));\n"
            'if (dim(std(Z)) < 0) { print("EMPTY"); quit; }\n'
            "list L = minAssGTZ(Z);\n"
            "ideal C; ideal S;\n"
            "option(redSB);\n"
            "for (int i = 1; i <= size(L); i++) {\n"
            "  C = simplify(L[i], 2); S = std(C);\n"
            '  print("COMP " + string(dim(S)) + " " + string(mult(S)) + " " + string(S)'
            ' + " ; " + string(C));\n'
            "}\n"
        )
        if "EMPTY" in out:
            return []
        found = [
            _Comp(m.group(4).strip(), int(m.group(1)), int(m.group(2)), m.group(3).strip())
            for m in re.finditer(r"^COMP (-?\d+) (\d+) (.*) ; (.*)$", out, re.MULTILINE)
        ]
        if not found:
            raise _undecided(f"Singular printed {out.strip()[:100]!r} instead of components")
        return found

    def inside(self, comps: Sequence[_Comp]) -> set[tuple[int, int]]:
        """The pairs (i, j), i != j, with V(comps[i]) inside V(comps[j])."""
        if len(comps) < 2:
            return set()
        lines = ["list G; list S; ideal Gj;"]
        for i, c in enumerate(comps, start=1):
            lines.append(f"G[{i}] = ideal({c.gens}); S[{i}] = std(G[{i}]);")
        lines += [
            f"int n = {len(comps)}; int i; int j; int jj; int ok;",
            "for (i = 1; i <= n; i++) { for (j = 1; j <= n; j++) { if (i != j) {",
            "  ok = 1; Gj = G[j];",
            "  for (jj = 1; jj <= ncols(Gj); jj++) { if (reduce(Gj[jj], S[i]) != 0) { ok = 0; } }",
            '  if (ok == 1) { print("IN " + string(i) + " " + string(j)); } } } }',
        ]
        out = self.run("\n".join(lines) + "\n")
        return {
            (int(m.group(1)) - 1, int(m.group(2)) - 1)
            for m in re.finditer(r"^IN (\d+) (\d+)$", out, re.MULTILINE)
        }

    # -- the members -------------------------------------------------------------------------

    def singular_locus(self) -> list[_Comp]:
        terms = {e[: self.d]: c for e, c in self.g.items() if not any(e[self.d :])}
        partials = ", ".join(f"diff(f, t{i + 1})" for i in range(self.d))
        return self.components(f"f, {partials}", f"poly f = {_poly(terms, self.tnames)};\n")

    def singular_part(self, member: _Comp) -> list[str]:
        """The singular locus of the member, as ideals."""
        if member.dim == self.d:
            return []
        out = self.run(
            f"ideal A = {member.gens};\n"
            f"ideal SC = sat(A + minor(jacob(A), {self.d - member.dim}), ideal({self.product}));\n"
            'if (dim(std(SC)) >= 0) { print("CAND " + string(SC)); }\n'
        )
        return re.findall(r"^CAND (.*)$", out, re.MULTILINE)

    def pair(
        self, member: _Comp, subset: tuple[int, ...], flag: int
    ) -> tuple[list[str], str, list[str], dict[int, bool]]:
        """The jump candidates of the member for the coordinate subset, as ideals in t.

        Returns the candidate ideals, the kind of the pair ("N" if the member is not inside the
        critical locus, "K" if it is a component, "E" otherwise), the reasons the condition
        fails, and, for each dimension j of a polar variety computed, whether the member lies
        inside it.
        """
        d, r, k = self.d, self.r, member.dim
        free = [j for j in range(r) if j not in subset]
        names = [*self.tnames, *(f"y{j + 1}" for j in free)]
        n = len(names)
        terms = {
            e[:d] + tuple(e[d + j] for j in free): c
            for e, c in self.g.items()
            if not any(e[d + j] for j in subset)
        }
        rng = random.Random(f"{self.seed}:{self.key}{self.salt}:{member.key}:{subset}:{flag}")
        yprod = "*".join(f"y{j + 1}" for j in free)
        partials = ", ".join(f"diff(f, var({c}))" for c in range(1, n + 1))
        center = ", ".join([member.gens, *(f"y{j + 1}" for j in free)])

        def candidate(expression: str) -> str:
            eliminate = f"PT = eliminate(PT, {yprod}); " if free else ""
            return (
                f"PT = sat({expression}, ideal({self.product}));\n"
                f'if (dim(std(PT)) >= 0) {{ {eliminate}print("CAND " + string(PT)); }}\n'
            )

        script = (
            f"poly f = {_poly(terms, names)};\n"
            f"ideal JI = {partials};\n"
            f"ideal IC = {center};\nideal ICs = std(IC);\n"
            "int i; int jj; int inC = 1; ideal PT;\n"
            "for (i = 1; i <= ncols(JI); i++) { if (reduce(JI[i], ICs) != 0) { inC = 0; } }\n"
            'if (inC == 0) {\n  print("CASE N");\n  ' + candidate("IC + JI") + "  quit;\n}\n"
            'print("CASE C");\n'
            f"ideal CR = sat(JI, ideal({self.product}));\n"
            "list CC = minAssGTZ(CR);\n"
            "int s = -1; int ncont = 0; int insing = 0; int dc; int ok2; int a; int h;\n"
            "ideal Ca; ideal SMa; matrix JC;\n"
            "for (a = 1; a <= size(CC); a++) {\n"
            "  Ca = CC[a]; ok2 = 1;\n"
            "  for (jj = 1; jj <= ncols(Ca); jj++) { if (reduce(Ca[jj], ICs) != 0) { ok2 = 0; } }\n"
            "  if (ok2 == 1) {\n"
            "    ncont++; dc = dim(std(Ca)); if (dc > s) { s = dc; }\n"
            f"    if (dc < {n}) {{\n"
            f"      JC = jacob(Ca); SMa = Ca + minor(JC, {n} - dc); ok2 = 1;\n"
            "      for (jj = 1; jj <= ncols(SMa); jj++) {"
            " if (reduce(SMa[jj], ICs) != 0) { ok2 = 0; } }\n"
            "      if (ok2 == 1) { insing = 1; } else {\n        "
            + candidate("IC + SMa")
            + "      }\n"
            "    }\n"
            "  } else {\n    " + candidate("IC + Ca") + "  }\n"
            "}\n"
            'print("S " + string(s) + " " + string(ncont));\n'
            'print("INSING " + string(insing));\n'
            f"matrix RW[{n}][{n}] = {_random_flag(rng, n)};\n"
            "intvec JJ;\n"
            f"if (s == {k}) {{ JJ = {k}; }} else {{ if (s == {k + 1}) {{ JJ = {k}, {k + 1}; }}"
            " else { JJ = s; } }\n"
            "int jv; int b; int c; int q; ideal GM; ideal GC; list GL;\n"
            "for (q = 1; q <= size(JJ); q++) {\n"
            f"  jv = JJ[q]; matrix MP[jv + 1][{n}];\n"
            f"  for (c = 1; c <= {n}; c++) {{ MP[1, c] = diff(f, var(c)); }}\n"
            f"  for (b = 1; b <= jv; b++) {{ for (c = 1; c <= {n}; c++) {{ MP[b + 1, c] = RW[b, c]; }} }}\n"
            f"  GM = sat(minor(MP, jv + 1), JI); GM = sat(GM, ideal({self.product}));\n"
            "  ok2 = 1;\n"
            "  for (jj = 1; jj <= ncols(GM); jj++) { if (reduce(GM[jj], ICs) != 0) { ok2 = 0; } }\n"
            '  print("POLAR " + string(jv) + " " + string(ok2));\n'
            "  if (ok2 == 0) {\n    " + candidate("IC + GM") + "  }\n"
            f"  if (s == {k + 1} && jv == {k + 1}) {{\n"
            f"    GC = sat(GM + JI, ideal({self.product}));\n"
            "    if (dim(std(GC)) >= 0) {\n"
            "      GL = minAssGTZ(GC);\n"
            "      for (h = 1; h <= size(GL); h++) {\n"
            "        Ca = GL[h]; ok2 = 1;\n"
            "        for (jj = 1; jj <= ncols(Ca); jj++) {"
            " if (reduce(Ca[jj], ICs) != 0) { ok2 = 0; } }\n"
            "        if (ok2 == 0) {\n          " + candidate("IC + Ca") + "        }\n"
            "      }\n"
            "      kill GL;\n"
            "    }\n"
            "  }\n"
            "  kill MP;\n"
            "}\n"
        )
        out = self.run(script, names)
        parts = re.findall(r"^CAND (.*)$", out, re.MULTILINE)
        polar = {
            int(a): b == "1" for a, b in re.findall(r"^POLAR (\d+) ([01])$", out, re.MULTILINE)
        }
        if "CASE N" in out:
            return parts, "N", [], polar
        found = re.search(r"^S (-?\d+) (\d+)$", out, re.MULTILINE)
        if found is None or "CASE C" not in out:
            raise _undecided(
                f"Singular printed {out.strip()[:100]!r} instead of the critical locus"
            )
        s, count = int(found.group(1)), int(found.group(2))
        reasons: list[str] = []
        if s == k and count == 1:
            kind = "K"
        else:
            kind = "E"
            if re.search(r"^INSING 1$", out, re.MULTILINE):
                reasons.append(
                    "the member lies in the singular locus of a component of the critical locus"
                )
        if polar.get(k):
            reasons.append("the member lies in the polar variety of its own dimension")
        if s >= k + 2:
            if polar.get(s) is None:
                raise _undecided("no polar variety was computed")
            if polar[s]:
                reasons.append(f"the member lies in the polar variety of dimension {s}")
        return parts, kind, reasons, polar

    def candidates(self, index: int) -> list[_Comp]:
        """The components of the jump candidates of the member.

        The union runs over the sections of the face. On a subdivided face these are the corner
        charts of the subdivision only: jumps of the fibre integral that come from inside the
        exceptional fibre are not located, and :meth:`strata` marks the positive-dimensional
        pieces undecided for that reason.
        """
        member = self.members[index]
        comp = member.comp
        parts = self.singular_part(comp)
        failed: list[str] = []
        for section in self.sections:
            self.r, self.g = section
            for size in range(self.r + 1):
                for subset in itertools.combinations(range(self.r), size):
                    try:
                        found, kind, reasons, polar = self.pair(comp, subset, 0)
                        big = [s for s in polar if s >= comp.dim + 2]
                        if kind == "E" and big and not any(polar[s] for s in big):
                            # Case (ii): a second flag must keep the member outside the polar
                            # variety as well.
                            again = self.pair(comp, subset, 1)
                            if any(again[3].get(s) for s in big):
                                reasons = [
                                    *reasons,
                                    "a second random flag puts the member in the polar variety",
                                ]
                    except ComputationError as exc:
                        if not str(exc).startswith("undecided: "):
                            raise
                        failed.append(
                            f"coordinates {list(subset)}: {str(exc)[len('undecided: ') :]}"
                        )
                        continue
                    parts += found
                    if len(self.sections) > 1:
                        continue  # the criterion is for a single smooth chart
                    for text in reasons:
                        self.failures.append(
                            ConditionFailure(
                                self.key, comp.dim, _parse(comp.gens, self.tnames), subset, text
                            )
                        )
        self.r, self.g = self.sections[0]
        if failed:
            self.candidate_reasons.setdefault(index, []).extend(failed)
        out: list[_Comp] = []
        for part in parts:
            try:
                out += self.components(part)
            except ComputationError as exc:
                if not str(exc).startswith("undecided: "):
                    raise
                self.candidate_reasons.setdefault(index, []).append(
                    f"a candidate set was not decomposed: {str(exc)[len('undecided: ') :]}"
                )
        return out

    def closure(self) -> None:
        """Build the members level by level."""
        index: dict[str, int] = {}

        def add(comp: _Comp) -> bool:
            if comp.key in index:
                return False
            index[comp.key] = len(self.members)
            self.members.append(_Member(comp))
            return True

        for comp in self.singular_locus():
            add(comp)
        frontier = list(range(len(self.members)))
        while frontier:
            pending: list[_Comp] = []
            for i in frontier:
                if self.members[i].comp.dim >= 1:
                    pending += self.candidates(i)
            relations = self.inside([m.comp for m in self.members])
            for i in frontier:
                for j in range(len(self.members)):
                    if j == i or (j in frontier and j < i):
                        continue
                    if (i, j) in relations or (j, i) in relations:
                        continue
                    pending += self.components(
                        f"{self.members[i].comp.gens}, {self.members[j].comp.gens}"
                    )
            frontier = []
            for comp in pending:
                if add(comp):
                    frontier.append(len(self.members) - 1)
            if len(self.members) > _MAX_MEMBERS:
                raise _undecided(f"the face has more than {_MAX_MEMBERS} members")

    # -- beta ----------------------------------------------------------------------------------

    def shape(self, comp: _Comp, rng: random.Random) -> _Point | None:
        """One point of a prime component of dimension 0 and degree above 1, with coordinates in
        the number field Q(a) of its points: a minimal polynomial m of a primitive element a,
        and each coordinate as a polynomial in a (the shape lemma, from a lexicographic
        Groebner basis of the component with u = a). None if no primitive element is found."""
        names = [*self.tnames, "u"]
        symbols = [sp.Symbol(n) for n in names]
        for _ in range(_SHAPE_TRIES):
            form = "+".join(f"{rng.randint(1, 97)}*{n}" for n in self.tnames)
            ring = f"ring R = 0, ({','.join(names)}), lp;\n"
            script = (
                f"option(redSB); ideal I = {comp.gens}, u - ({form}); ideal G = std(I);\n"
                'print("SHAPE " + string(G));\n'
            )
            out = _singular_run(_HEADER + ring + script + "quit;\n", self.singular, self.timeout)
            found = re.search(r"^SHAPE (.*)$", out, re.MULTILINE)
            if found is None:
                continue
            basis = _parse(found.group(1), names)
            if len(basis) != self.d + 1:
                continue
            rows: dict[sp.Symbol, sp.Expr] = {}
            minimal: sp.Expr | None = None
            for g in basis:
                if g.free_symbols == {symbols[-1]}:
                    minimal = g
                    continue
                for t in symbols[:-1]:
                    poly = sp.Poly(g, t)
                    if poly.degree() == 1 and g.free_symbols - {t} <= {symbols[-1]}:
                        rows[t] = sp.solve(g, t)[0]
            if minimal is None or set(rows) != set(symbols[:-1]):
                continue
            if sp.Poly(minimal, symbols[-1]).degree() != comp.degree:
                continue
            a = _GENERATOR
            field = NumberField(sp.expand(minimal.subs(symbols[-1], a)), a)
            coordinates = tuple(sp.expand(rows[t].subs(symbols[-1], a)) for t in symbols[:-1])
            return _Point(coordinates, field)
        return None

    def points(self, comp: _Comp, avoid: Sequence[_Comp], want: int) -> list[_Point]:
        """Points on the member outside the members inside it, or fewer: rational points when
        there are enough, and otherwise, for the others, points with coordinates in a number
        field, from components of the slices that are not rational."""
        rng = random.Random(f"{self.seed}:{self.key}{self.salt}:{comp.key}:points")
        symbols = [sp.Symbol(n) for n in self.tnames]
        avoiders = [_parse(a.gens, self.tnames) for a in avoid]
        found: list[tuple[Fraction, ...]] = []
        irrational: dict[str, _Comp] = {}
        values = [Fraction(a, b) for a in range(-9, 10) if a for b in (1, 2, 3)]
        for _ in range(_POINT_TRIES):
            if len(found) >= want:
                break
            if comp.dim == 0:
                fixed: list[str] = []
            else:
                chosen = rng.sample(range(self.d), comp.dim)
                fixed = [
                    f"t{j + 1} - ({(c := rng.choice(values)).numerator}/{c.denominator})"
                    for j in chosen
                ]
            gens = ", ".join([comp.gens, *fixed])
            for found_comp in self.components(gens):
                if found_comp.dim != 0:
                    continue
                if found_comp.degree != 1:
                    irrational.setdefault(found_comp.key, found_comp)
                    continue
                solutions = sp.solve(list(_parse(found_comp.key, self.tnames)), symbols, dict=True)
                if len(solutions) != 1 or set(solutions[0]) != set(symbols):
                    continue
                point = [Fraction(int(solutions[0][s].p), int(solutions[0][s].q)) for s in symbols]
                exact = {
                    s: sp.Rational(v.numerator, v.denominator)
                    for s, v in zip(symbols, point, strict=True)
                }
                if any(all(g.subs(exact) == 0 for g in gens_) for gens_ in avoiders):
                    continue
                if tuple(point) not in found and all(v != 0 for v in point):
                    found.append(tuple(point))
        out = [_Point(p) for p in found]
        # The points of a prime component are conjugate: none lies on a smaller member unless
        # all do, so one test of the first point decides, as for a rational point.
        for found_comp in sorted(irrational.values(), key=lambda c: (c.degree, c.key)):
            if len(out) >= want:
                break
            point_ = self.shape(found_comp, rng)
            if point_ is None or point_.field is None:
                continue
            field = point_.field
            minimal = sp.Poly(field.minimal_polynomial, field.generator)
            exact = dict(zip(symbols, point_.coordinates, strict=True))
            if any(
                all(sp.rem(sp.expand(g.subs(exact)), minimal, field.generator) == 0 for g in gens_)
                for gens_ in avoiders
            ):
                continue
            if any(sp.rem(sp.expand(c), minimal, field.generator) == 0 for c in point_.coordinates):
                continue
            out.append(point_)
        return out

    def beta(self, index: int, avoid: Sequence[_Comp]) -> None:
        member = self.members[index]
        comp = member.comp
        want = 1 if comp.dim == 0 else 2
        try:
            points = self.points(comp, avoid, want)
        except ComputationError as exc:
            if not str(exc).startswith("undecided: "):
                raise
            member.reason = f"no point was found: {str(exc)[len('undecided: ') :]}"
            return
        if len(points) < want:
            member.reason = (
                f"only {len(points)} points off the smaller members were found over Q or a "
                f"number field, {want} are needed"
            )
            return
        values: list[int] = []
        primes: list[int] = []
        # A point is computed twice, with two seeds, when the member is a point.
        for n, point in enumerate(points if comp.dim else points * 2):
            result = self.evaluate(point, self.seed + n)
            if result.reason is not None or result.beta is None:
                member.reason = f"beta at a point is undecided: {result.reason}"
                return
            values.append(result.beta)
            if result.prime is not None:
                primes.append(result.prime)
        if len(set(values)) != 1:
            member.reason = f"beta differs between points of the member: {values}"
            return
        member.beta = values[0]
        member.primes = tuple(primes)

    # -- the pieces --------------------------------------------------------------------------

    def strata(self) -> list[Stratum]:
        comps = [m.comp for m in self.members]
        relation = self.inside(comps)
        count = len(comps)
        above = [{j for j in range(count) if (i, j) in relation} for i in range(count)]
        for i in range(count):
            avoid = [comps[j] for j in range(count) if (j, i) in relation]
            self.beta(i, avoid)
        parent: dict[int, int] = {}
        for i in range(count):
            minimal = [j for j in above[i] if not any(j in above[m] for m in above[i])]
            if len(minimal) == 1:
                j = minimal[0]
                a, b = self.members[i], self.members[j]
                if a.beta is not None and a.beta == b.beta:
                    parent[i] = j

        def root(i: int) -> int:
            while i in parent:
                i = parent[i]
            return i

        owner = [root(i) for i in range(count)]
        out = []
        for i in sorted(set(owner), key=lambda i: (-comps[i].dim, comps[i].key)):
            absorbed = [j for j in range(count) if owner[j] == i]
            reasons: list[str] = []
            for j in absorbed:
                if (own := self.members[j].reason) is not None:
                    reasons.append(own)
                reasons += self.candidate_reasons.get(j, [])
            lower = [j for j in range(count) if (j, i) in relation and owner[j] != i]
            removed = [j for j in lower if not any((j, m) in relation for m in lower if m != j)]
            if self.method == "subdivision" and comps[i].dim > 0:
                reasons.append(_CORNER_ONLY)
            reason = "; ".join(reasons) or None
            beta = self.members[i].beta
            out.append(
                Stratum(
                    face=self.key,
                    dimension=comps[i].dim,
                    generators=_parse(comps[i].gens, self.tnames),
                    degree=comps[i].degree,
                    removed=tuple(_parse(comps[j].gens, self.tnames) for j in sorted(removed)),
                    mu_t=None if reason or beta is None else self.sign * beta,
                    euler=None,
                    chart=self.chart,
                    reason=reason if reason or beta is not None else "beta was not computed",
                    method=self.method,
                )
            )
        return out


class _Cover:
    """The orbits of a smooth subdivision of the normal cone of a face that lie over the
    orbit of the face, with the local equation of G on each.

    The charts of the maximal cones share the torus coordinates t of the orbit of the face
    (:func:`~feynkit.toric.orbit_chart`). The orbit of a cone tau of the subdivision whose
    relative interior lies in that of the cone of the face has torus coordinates (t, z), z the
    coordinates of the rays of a maximal cone containing tau that are not in tau, and normal
    coordinates the rays of tau: it is the chart of the face tau, with the torus basis
    extended by those rays. Over a point q of the orbit of the face it is the torus z, the
    fibre of the orbit of tau over q.
    """

    def __init__(
        self,
        data: PolytopeData,
        key: tuple[int, ...],
        base: NormalCone,
        terms: Mapping[tuple[int, ...], Fraction],
        seed: int,
        timeout: float,
        pieces: Sequence[NormalCone] | None = None,
    ) -> None:
        if pieces is None:
            pieces = smooth_subdivision(base, seed=seed, timeout=timeout)
        coefficients: dict[Sequence[int], Fraction] = {
            tuple(data.chart.coordinates[i]): terms[data.points[i]] for i in range(len(data.points))
        }
        self.charts = [orbit_chart(data, key, cone=piece) for piece in pieces]
        self.sections = [(c.normal_dimension, c.local_equation(coefficients)) for c in self.charts]
        self.orbits: list[tuple[OrbitChart, dict[tuple[int, ...], Fraction]]] = []
        self.strata: dict[int, _OrbitStrata] = {}
        for rays in orbit_cones(data, pieces):
            if not rays:
                continue
            chart = next(c for c in self.charts if set(rays) <= set(c.rays))
            rest = tuple(u for u in chart.rays if u not in rays)
            orbit = OrbitChart(chart.face, rays, (*chart.torus_basis, *rest), chart.vertex)
            self.orbits.append((orbit, orbit.local_equation(coefficients)))


class _OrbitStrata:
    """The strata of constant beta of a smooth chart of an orbit of the subdivision, in its torus
    coordinates (t, z): t the coordinates of the orbit of the face, z those of the fibre.

    beta of the chart is the one of :func:`~feynkit.milnor.torus_cut_milnor_number`, with the
    normal coordinates of the orbit. The strata are computed once for all points q, and the
    integral over the fibre {t = q} sums beta times the Euler characteristic of each stratum
    cut with the fibre.
    """

    def __init__(
        self,
        local: dict[tuple[int, ...], Fraction],
        k: int,
        extra: int,
        r: int,
        *,
        seed: int,
        timeout: float,
        singular: str,
        salt: str,
    ) -> None:
        self.k, self.extra = k, extra
        self.timeout = timeout
        self.seed = seed
        self.primes: list[int] = []
        shift = [max(0, -min(e[a] for e in local)) for a in range(k + extra)] if local else []

        def evaluate(point: _Point, n: int) -> TorusCutMilnorNumber:
            result = _smooth_chart(
                local, k + extra, r, point.coordinates, shift, n, timeout, 1, _field(point.field)
            )
            if result.prime is not None:
                self.primes.append(result.prime)
            return result

        self.face = _Face(
            key=(),
            chart=None,
            d=k + extra,
            sections=[(r, local)],
            evaluate=evaluate,
            sign=1,
            seed=seed,
            timeout=timeout,
            singular=singular,
            salt=salt,
        )
        self.face.closure()
        self.strata = self.face.strata()
        self.names = [sp.Symbol(f"t{i + 1}") for i in range(k + extra)]

    def integral(self, point: Sequence[Fraction]) -> int:
        """The integral over the fibre {t = point} of beta, by Euler characteristic.

        Raises
        ------
        ComputationError
            Starting with "undecided: ", if a stratum that meets the fibre is undecided or an
            Euler characteristic is, or if the condition (H) failed in the orbit.
        """
        if self.face.failures:
            raise _undecided(
                "the condition on the polar varieties failed in the orbit: "
                f"{self.face.failures[0].reason}"
            )
        fixed = {
            self.names[i]: sp.Rational(q.numerator, q.denominator) for i, q in enumerate(point)
        }
        z = self.names[self.k :]
        total = 0
        for stratum in self.strata:
            pieces = [
                [sp.expand(g.subs(fixed)) for g in generators]
                for generators in (stratum.generators, *stratum.removed)
            ]
            closure = pieces[0]
            if any(g.is_number and g != 0 for g in closure):
                continue  # the closure misses the fibre
            if stratum.reason is not None or stratum.mu_t is None:
                raise _undecided(f"the fibre is undecided: {stratum.reason}")
            euler = 0
            removed = pieces[1:]
            for size in range(len(removed) + 1):
                for chosen in itertools.combinations(removed, size):
                    generators = [*closure, *itertools.chain.from_iterable(chosen)]
                    euler += (-1) ** size * torus_euler_characteristic(
                        generators, z, seed=self.seed, timeout=self.timeout
                    )
            total += stratum.mu_t * euler
        return total


def _pushed_forward_beta(
    cover: _Cover,
    point: Sequence[Fraction],
    *,
    seed: int,
    timeout: float,
    singular: str,
) -> tuple[int, list[int]]:
    """beta at a point of the orbit of the face, as the sum over the orbits of the subdivision
    above it of the integral over the fibre of beta of the smooth chart.

    Nearby cycles commute with the push-forward along the proper toric morphism pi from the
    subdivided variety (A. Dimca, Sheaves in Topology, Springer 2004, Prop. 4.2.11, p. 109, with
    Prop. 4.1.31 to 4.1.33, pp. 98-99), and pi is an isomorphism over the torus. So the Euler
    characteristic of the Milnor fibre of G at q, cut by the torus, is the integral over pi^-1(q)
    of the same number computed in a smooth chart at the point of pi^-1(q); the fibre over q of the
    orbit O_tau is a torus, and the integral is the sum over the orbits and over the pieces of
    the torus on which the number is constant.

    Raises
    ------
    ComputationError
        Starting with "undecided: ", if a term is undecided.
    """
    k = len(point)
    total = 0
    primes: list[int] = []
    for index, (orbit, local) in enumerate(cover.orbits):
        extra = orbit.torus_dimension - k
        r = orbit.normal_dimension
        if extra == 0:
            shift = [max(0, -min(e[a] for e in local)) for a in range(k)] if local else [0] * k
            result = _smooth_chart(local, k, r, list(point), shift, seed, timeout, 1)
            if result.reason is not None or result.beta is None:
                raise _undecided(str(result.reason))
            total += result.beta
            if result.prime is not None:
                primes.append(result.prime)
            continue
        if index not in cover.strata:
            cover.strata[index] = _OrbitStrata(
                local,
                k,
                extra,
                r,
                seed=seed,
                timeout=timeout,
                singular=singular,
                salt=f":{orbit.rays}",
            )
        total += cover.strata[index].integral(point)
        primes += cover.strata[index].primes
    return total, primes


def _prepare(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction] | None,
    scale: sp.Symbol | None,
) -> tuple[dict[tuple[int, ...], Fraction], PolytopeData, sp.Expr]:
    """The terms of G at the point, the polytope data of their support and G as an expression."""
    support = _support(polynomial, tuple(variables), {} if point is None else point, scale)
    terms = {
        tuple(p): Fraction(int(c.p), int(c.q))
        for p, c in zip(support.points, support.coefficients, strict=True)
    }
    data = polytope_data(sorted(terms))
    expression = sum(
        (
            sp.Rational(c.numerator, c.denominator)
            * sp.Mul(*(x**e for x, e in zip(variables, p, strict=True)))
            for p, c in terms.items()
        ),
        start=sp.Integer(0),
    )
    return terms, data, expression


def _analyse(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction] | None,
    scale: sp.Symbol | None,
    seed: int,
    timeout: float,
    faces: Sequence[Sequence[int]] | None,
) -> StrataAnalysis:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError(f"seed must be an integer, not {seed!r:.60}")
    _check_timeout(timeout, optional=False)
    terms, data, expression = _prepare(polynomial, variables, point, scale)
    singular = _singular_binary()
    if singular is None:
        raise RuntimeError("singular_strata needs Singular, which was not found")
    sign = (-1) ** (data.dimension - 1)
    strata: list[Stratum] = []
    failures: list[ConditionFailure] = []
    subdivided: list[tuple[int, ...]] = []
    primes: set[int] = set()
    wanted: set[tuple[int, ...]] | None = None
    if faces is not None:
        if isinstance(faces, (str, bytes)) or not isinstance(faces, Sequence):
            raise ValidationError("faces must be a sequence of faces, each a sequence of indices")
        known = {tuple(sorted(indices)) for _, indices in data.faces}
        wanted = set()
        for entry in faces:
            try:
                key = tuple(sorted(int(i) for i in entry))
            except (TypeError, ValueError):
                raise ValidationError("faces must be sequences of point indices") from None
            if key not in known:
                raise ValidationError(f"{list(key)} are not the point indices of a face")
            wanted.add(key)
    for dimension, indices in data.faces:
        if dimension < 1 or (wanted is not None and tuple(sorted(indices)) not in wanted):
            continue
        key = tuple(indices)
        if len(key) == dimension + 1:
            continue  # affinely independent points: the face polynomial is smooth
        base = normal_cone(data, key)
        exps = [data.points[i] for i in key]
        if not base.smooth:
            found, used, used_primes = _non_smooth(
                data, key, base, terms, exps, sign, seed, singular, timeout
            )
            strata += found
            primes.update(used_primes)
            if used:
                subdivided.append(key)
            continue
        chart = orbit_chart(data, key)
        local = chart.local_equation(
            {
                tuple(data.chart.coordinates[i]): terms[data.points[i]]
                for i in range(len(data.points))
            }
        )

        def evaluate(
            point: _Point, n: int, chart: OrbitChart = chart, exps: list = exps
        ) -> TorusCutMilnorNumber:
            return torus_cut_milnor_number(
                expression,
                variables,
                exps,
                list(point.coordinates),
                seed=n,
                timeout=timeout,
                chart=chart,
                field=point.field,
            )

        face = _Face(
            key=key,
            chart=chart,
            d=chart.torus_dimension,
            sections=[(chart.normal_dimension, local)],
            evaluate=evaluate,
            sign=sign,
            seed=seed,
            timeout=timeout,
            singular=singular,
        )
        try:
            face.closure()
            pieces = face.strata()
        except ComputationError as exc:
            if not str(exc).startswith("undecided: "):
                raise
            strata.append(
                Stratum(key, None, (), None, (), None, None, chart, str(exc)[len("undecided: ") :])
            )
            continue
        strata += pieces
        failures += face.failures
        for member in face.members:
            primes.update(member.primes)
    complete = not failures and not subdivided and all(s.reason is None for s in strata)
    return StrataAnalysis(
        points=data.points,
        strata=tuple(strata),
        primes=tuple(sorted(primes)),
        seed=seed,
        complete=complete,
        failures=tuple(failures),
        subdivided=tuple(subdivided),
    )


def _non_smooth(
    data: PolytopeData,
    key: tuple[int, ...],
    base: NormalCone,
    terms: Mapping[tuple[int, ...], Fraction],
    exps: Sequence[tuple[int, ...]],
    sign: int,
    seed: int,
    singular: str,
    timeout: float,
) -> tuple[list[Stratum], bool, set[int]]:
    """The pieces of a face whose normal cone is not smooth, whether a subdivision was used, and
    the primes of the Le computations.

    None if the face polynomial has no singular point in its orbit. Otherwise the cone is
    subdivided into smooth cones (``smooth_subdivision`` with a time limit, undecided past it),
    the members come from the candidates of the charts of all the pieces, and beta at a point of
    a member is the push-forward sum of :func:`_pushed_forward_beta`, with the subdivision of
    seed ``seed + n`` for the n-th point, so that the two points of a member also compare two
    subdivisions. ``complete`` is not claimed on such a face. The candidates come from the corner
    charts only, so every piece of positive dimension is undecided and only points carry a value.
    """
    coordinates = lattice_coordinates(list(exps))
    d = len(coordinates[0])
    names = [f"t{i + 1}" for i in range(d)]
    f = _poly(
        {tuple(c): terms[e] for c, e in zip(coordinates, exps, strict=True)},
        names,
    )
    partials = ", ".join(f"diff(f, t{i + 1})" for i in range(d))
    ring = f"ring R = 0, ({','.join(names)}), dp;\n"
    script = (
        f"poly f = {f};\nideal Z = f, {partials};\n"
        f"Z = sat(Z, ideal({'*'.join(names)}));\n"
        'print("DIM " + string(dim(std(Z))));\n'
    )

    def undecided(reason: str) -> tuple[list[Stratum], bool, set[int]]:
        return [Stratum(key, None, (), None, (), None, None, None, reason)], False, set()

    try:
        out = _singular_run(_HEADER + ring + script + "quit;\n", singular, timeout)
    except ComputationError as exc:
        if not str(exc).startswith("undecided: "):
            raise
        return undecided(str(exc)[len("undecided: ") :])
    found = re.search(r"^DIM (-?\d+)$", out, re.MULTILINE)
    if found is None:
        raise ComputationError(f"Singular printed {out.strip()[:100]!r} instead of a dimension")
    if int(found.group(1)) < 0:
        return [], False, set()
    covers: dict[int, _Cover] = {}

    def cover(n: int) -> _Cover:
        if n not in covers:
            covers[n] = _Cover(data, key, base, terms, seed + n, timeout)
        return covers[n]

    try:
        first = cover(0)
    except ComputationError as exc:
        if not str(exc).startswith("undecided: "):
            raise
        return undecided(str(exc)[len("undecided: ") :])

    def evaluate(point: _Point, n: int) -> TorusCutMilnorNumber:
        def refused(reason: str) -> TorusCutMilnorNumber:
            return TorusCutMilnorNumber(None, None, "subdivision", (), None, reason)

        if point.field is not None:
            return refused(
                "the point has coordinates in a number field and the subdivision path takes "
                "rational points only"
            )
        try:
            beta, primes = _pushed_forward_beta(
                cover(n - seed),
                [c for c in point.coordinates if isinstance(c, Fraction)],
                seed=n,
                timeout=timeout,
                singular=singular,
            )
        except ComputationError as exc:
            if not str(exc).startswith("undecided: "):
                raise
            return refused(str(exc)[len("undecided: ") :])
        return TorusCutMilnorNumber(
            sign * beta, beta, "subdivision", (), min(primes, default=None), None
        )

    face = _Face(
        key=key,
        chart=first.charts[0],
        d=first.charts[0].torus_dimension,
        sections=first.sections,
        evaluate=evaluate,
        sign=sign,
        seed=seed,
        timeout=timeout,
        singular=singular,
        method="subdivision",
    )
    try:
        face.closure()
        pieces = face.strata()
    except ComputationError as exc:
        if not str(exc).startswith("undecided: "):
            raise
        return undecided(str(exc)[len("undecided: ") :])
    return pieces, True, {p for member in face.members for p in member.primes}


def singular_strata_from_polynomial(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    *,
    scale: sp.Symbol | None = None,
    seed: int = 0,
    timeout: float = 120,
    faces: Sequence[Sequence[int]] | None = None,
) -> StrataAnalysis:
    """The strata of constant mu^T on the singular locus of every face of the Newton polytope.

    For each face F of the Newton polytope P of G, the top face included, the singular locus
    of the face polynomial G_F in the orbit O_F is cut into locally closed pieces on which the
    torus-cut Milnor number mu^T = (-1)^(N-1) beta is constant, N the dimension of P, with
    its value on each; see :class:`StrataAnalysis` for the construction, the criterion behind
    ``complete`` and the fields. A face whose polynomial is smooth in its orbit has no piece. Where
    the normal cone of F is smooth, the chart of :func:`~feynkit.toric.orbit_chart` is
    used and beta comes from :func:`~feynkit.milnor.torus_cut_milnor_number`, at two points of
    each member, which must agree: rational points, or, where a member has fewer than two,
    points with coordinates in a number field. The decompositions are exact, over Q, by
    Singular; a member with fewer than two points outside the smaller members is undecided.

    Where the cone is not smooth and the polynomial is singular in the orbit, the cone is
    subdivided into smooth cones (:func:`~feynkit.toric.smooth_subdivision`, within
    ``timeout``; past it the face is undecided). The members come from the candidates of the
    charts of all the pieces, and beta at a rational point q of a member is the sum, over the
    orbits of the subdivision above the orbit of the face, of the integral over the fibre of
    q of beta in the smooth chart of the orbit: the fibre is a torus, cut into the pieces
    on which beta is constant by the same construction in the chart of the orbit, and
    integrated by the Euler characteristic of each piece cut with the fibre. This is the
    push-forward of the nearby cycles (A. Dimca, Sheaves in Topology, Springer 2004,
    Prop. 4.2.11, p. 109). The two points of a member use the subdivisions of seeds
    ``seed`` and ``seed + 1``, which must agree; the pieces are then ``method="subdivision"``,
    listed in :attr:`StrataAnalysis.subdivided`, and ``complete`` is False. A member with
    fewer than two rational points, or a point over a number field, is undecided there.
    On a subdivided face the candidate set comes from the corner charts only, and jumps of the
    fibre integral that come from inside the exceptional fibre are not yet located; every
    piece of positive dimension is therefore undecided (it has a ``reason``), and only the
    pieces of dimension 0 carry a value. The same holds if the condition on the polar
    varieties fails inside an orbit of the subdivision: beta is then undecided.

    Parameters
    ----------
    polynomial
        G, with coefficients that are rational functions, with rational coefficients, of the
        symbols other than ``variables``.
    variables
        The variables of G.
    point
        A rational kinematic point, keyed as for
        :func:`~feynkit.degeneracy.face_degeneracy_from_polynomial`; None, the default, when G has
        no other symbols.
    scale
        The energy scale, set to 1.
    seed
        Seed of the random flags, points and Le computations.
    timeout
        The most seconds each Singular run gets, at most 2,000,000.
    faces
        The point indices of the faces to analyse, as in :attr:`StrataAnalysis.points`; None,
        the default, analyses every face of dimension 1 or more. ``complete`` then speaks of
        these faces only.

    Raises
    ------
    RuntimeError
        If Singular is not installed.
    ValidationError
        If the arguments are not as for :func:`~feynkit.degeneracy.face_degeneracy_from_polynomial`,
        ``seed`` is not an integer, or ``faces`` are not faces of the Newton polytope.
    """
    return _analyse(polynomial, variables, point, scale, seed, timeout, faces)


def singular_strata(
    integral: FeynmanIntegral,
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    *,
    seed: int = 0,
    timeout: float = 120,
    faces: Sequence[Sequence[int]] | None = None,
) -> StrataAnalysis:
    """The strata of constant mu^T for G = U + F of an integral at a rational kinematic point.

    The point is keyed as
    :attr:`~feynkit.point_count.TorusCount.point` gives it, with the energy scale set to 1.
    See :func:`singular_strata_from_polynomial` for the arguments and errors. At a generic
    point no face polynomial is singular in its orbit (A. G. Kouchnirenko, Invent. Math. 32
    (1976) 1-31, Thm IV, p. 30, and :func:`~feynkit.degeneracy.face_degeneracy`), so there are no
    pieces.

    Raises
    ------
    ValidationError
        Also if the integral has kinematic constraints, which are not applied.
    """
    if integral.kinematic_constraints:
        raise ValidationError(
            "singular_strata does not apply kinematic_constraints; substitute them in the "
            "momentum products"
        )
    sym = integral.symanzik
    return _analyse(
        sym.g, list(sym.lp_parameters), point, integral.graph.energy_scale, seed, timeout, faces
    )


# --- the stratum sum ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratumSum:
    """The stratum sum of a polynomial G beside its volume and its number of critical points.

    Let P be the Newton polytope of G, of dimension N equal to the number of variables, T the
    torus, and H a polynomial supported on the lattice points of P with generic coefficients.
    The identity is

        vol(P) - |chi| = sum over the strata S of the singular loci of mu^T_S chi(S minus V(H)),

    where vol is the volume normalised to the lattice Z^N, |chi| = |chi(T minus V(G))| and
    S runs over the pieces of :class:`StrataAnalysis`, of every face of P, the top face
    included, on each of which the torus-cut Milnor number mu^T of
    :func:`~feynkit.milnor.torus_cut_milnor_number` is constant. It is the toric, torus-cut
    form of the formula of A. Parusinski and P. Pragacz (J. Algebraic Geom. 4 (1995) 337-351,
    Prop. 7, p. 8 of the preprint); equivalently, it follows from J. Schuermann,
    arXiv:math/0202175, Cor. 0.2 (p. 8), or from S. M. Gusein-Zade, I. Luengo and
    A. Melle-Hernandez, Proc. Steklov Inst. Math. 225 (1999) 156-164, Thm 2 (p. 4 of
    arXiv:math/9804071). The volume is the Euler characteristic of a generic hypersurface in the
    torus up to sign (A. G. Kouchnirenko, Invent. Math. 32 (1976) 1-31, Thm IV, p. 30), and
    J. Huh (Compos. Math. 149 (2013) 1245-1266, Thm 1(iii)) reads |chi| as a number of critical
    points. The left side is the drop of the number of master integrals below its generic
    value, vol.

    The sum is computed in the coordinates of the lattice that the differences of the exponents
    span, in which every Euler characteristic is of a subvariety of a torus of the strata's
    charts, and multiplied by the index of that lattice in Z^N, 1 for every Feynman integral
    with a full-dimensional polytope. The coefficients of H are drawn from ``seed``, and H is
    a section of the ample line bundle of P, not a generic polynomial: (G2) below is a
    probability-one condition on it.

    ``agrees`` compares ``total`` with ``drop``. It is None, never True, whenever the sum may
    be partial or unproved: when any piece of the strata is undecided (it has a ``reason``) or
    ``complete`` of the analysis is False, because the pieces are not known to cover every
    locus where mu^T jumps (a missed jump would change the total silently); when an Euler
    characteristic could not be decided; and when H is not transverse to a piece. The cause is
    then in ``reason``. A False means the sum is complete in this sense and still differs from
    the drop, so a stratum or a value is wrong.

    Attributes
    ----------
    volume
        The volume of P normalised to Z^N.
    master_count
        |chi|, the number of critical points of a generic master function on T minus V(G), by
        :func:`~feynkit.point_count.critical_point_count` at the seeds ``seed`` and
        ``seed + 1``, which must agree.
    strata
        The pieces of :func:`singular_strata_from_polynomial`, with ``euler`` set to
        chi(S minus V(H)) in the chart torus of the piece, for the pieces on which mu^T is
        decided and not 0, which are the only ones that count. The others keep ``euler`` None.
    total
        The sum, or None if it is partial because a piece or an Euler characteristic is
        undecided.
    agrees
        Whether ``total`` equals ``drop``, or None where that is undecided; see above.
    transverse
        Condition (G2) for the pieces that count: whether d(h) restricted to a piece is non-zero
        at every point of the piece and V(h), where h is H in the chart of the face. True
        when it holds for every piece, False when it fails for one, None if a check is
        undecided. Failing is a reason to draw H again with another seed.
    primes
        The primes of the computations, in increasing order: those of the analysis and those of
        the Euler characteristics and transversality checks.
    seed
        The seed.
    complete
        ``complete`` of the analysis.
    reason
        Why ``agrees`` or ``total`` is None, or None.
    """

    volume: int
    master_count: int
    strata: tuple[Stratum, ...]
    total: int | None
    agrees: bool | None
    transverse: bool | None
    primes: tuple[int, ...]
    seed: int
    complete: bool = False
    reason: str | None = None

    @property
    def drop(self) -> int:
        """vol - |chi|, the number of master integrals lost to the degeneracy."""
        return self.volume - self.master_count


def _face_h(
    data: PolytopeData,
    stratum: Stratum,
    h_terms: Mapping[tuple[int, ...], int],
    names: Sequence[sp.Symbol],
) -> sp.Poly:
    """H on the face of the stratum, as a polynomial in the chart coordinates t of the face.

    A lattice point of P lies on the face F when every inner normal of a facet through F is 0
    on its difference from a vertex of F. The monomials t^a of the face's polynomial are then
    multiplied by a monomial so that the exponents are non-negative, which does not change
    the zero set in the torus.
    """
    chart = stratum.chart
    assert chart is not None
    rays = normal_cone(data, stratum.face).rays
    out: dict[tuple[int, ...], int] = {}
    for c, coefficient in h_terms.items():
        shifted = [a - b for a, b in zip(c, chart.vertex, strict=True)]
        if any(sum(u * s for u, s in zip(ray, shifted, strict=True)) for ray in rays):
            continue
        key = tuple(
            sum(w * s for w, s in zip(row, shifted, strict=True)) for row in chart.torus_basis
        )
        out[key] = out.get(key, 0) + coefficient
    shift = [max(0, -min(e[a] for e in out)) for a in range(len(names))]
    expression = sum(
        (
            sp.Integer(c) * sp.Mul(*(t ** (e[a] + shift[a]) for a, t in enumerate(names)))
            for e, c in out.items()
        ),
        start=sp.Integer(0),
    )
    return sp.Poly(expression, *names, domain="QQ")


def _piece(
    stratum: Stratum,
    h: sp.Poly,
    *,
    seed: int,
    backend: str,
    timeout: float,
    singular: str,
    msolve: str | None,
) -> tuple[int, bool, list[int]]:
    """chi(S minus V(h)) for the piece S of the stratum, whether h is transverse to S, and the
    primes used.

    S is the closure V(generators) less the closures V(removed), which lie inside it, so
    chi(S minus V(h)) = chi(V(generators) minus V(h)) - chi(union of V(removed) minus V(h)),
    computed at the two primes, which must agree.

    Raises
    ------
    ComputationError
        With a message that starts with "undecided: ", as :func:`torus_euler_characteristic`.
    """
    chart = stratum.chart
    assert chart is not None and stratum.dimension is not None
    d = chart.torus_dimension
    names = h.gens

    def poly(expr: sp.Expr) -> sp.Poly:
        return sp.Poly(sp.expand(expr), *names, domain="QQ")

    generators = [q for q in map(poly, stratum.generators) if not q.is_zero]
    removed = [[q for q in map(poly, r) if not q.is_zero] for r in stratum.removed]
    primes = _two_primes([*generators, *(q for r in removed for q in r), h])
    values: list[tuple[int, bool]] = []
    for prime in primes:
        run = _Run(
            prime=prime,
            dimension=d,
            removed=_singular_polynomial(h.clear_denoms(convert=True)[1], prime),
            seed=seed,
            backend=backend,
            timeout=timeout,
            singular=singular,
            msolve=msolve,
        )

        def ideal(polys: Sequence[sp.Poly], prime: int = prime) -> str:
            integral = [q.clear_denoms(convert=True)[1] for q in polys]
            return ",".join(_singular_polynomial(q, prime) for q in integral) or "0"

        gens = ideal(generators)
        family = [c for r in removed for c in run.components(ideal(r))]
        value = run.closed(gens) - (run.union(family) if family else 0)
        values.append((value, run.transverse(gens, stratum.dimension, [ideal(r) for r in removed])))
    if values[0][0] != values[1][0]:
        raise _undecided(
            f"the Euler characteristic of a piece is {values[0][0]} modulo {primes[0]} and "
            f"{values[1][0]} modulo {primes[1]}"
        )
    return values[0][0], values[0][1] and values[1][1], primes


def _random_h(data: PolytopeData, seed: int) -> dict[tuple[int, ...], int]:
    """Random non-zero integer coefficients on every lattice point of P, in chart coordinates."""
    rng = random.Random(f"H:{seed}")
    points = lattice_points(data.chart.coordinates, lattice="ambient")
    return {p: rng.randint(1, 1000) for p in points}


def _stratum_sum(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction] | None,
    scale: sp.Symbol | None,
    seed: int,
    timeout: float,
    backend: str,
) -> StratumSum:
    if backend not in _BACKENDS:
        raise ValidationError(f"backend must be 'singular' or 'msolve', not {backend!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError(f"seed must be an integer, not {seed!r:.60}")
    _check_timeout(timeout, optional=False)
    terms, data, expression = _prepare(polynomial, variables, point, scale)
    if not data.is_full_dimensional:
        raise ValidationError(
            f"the Newton polytope has dimension {data.dimension} in {data.ambient_dimension} "
            "variables; the stratum sum needs it to be full-dimensional"
        )
    singular = _singular_binary()
    if singular is None:
        raise RuntimeError("stratum_sum needs Singular, which was not found")
    msolve = _msolve_binary() if backend == "msolve" else None
    if backend == "msolve" and msolve is None:
        raise RuntimeError("stratum_sum with backend='msolve' needs msolve")
    index = data.sublattice_index
    volume = data.normalized_volume * index
    counts = [
        critical_point_count(
            expression, variables, {}, seed=seed + k, timeout=timeout, backend=backend
        )
        for k in (0, 1)
    ]
    if counts[0] != counts[1]:
        raise ComputationError(
            f"the number of critical points is {counts[0]} at seed {seed} and {counts[1]} at "
            f"seed {seed + 1}"
        )
    analysis = _analyse(polynomial, variables, point, scale, seed, timeout, None)

    h_terms = _random_h(data, seed)
    names = tuple(sp.Symbol(f"t{i + 1}") for i in range(data.dimension))
    h_faces: dict[tuple[int, ...], sp.Poly] = {}
    reasons: list[str] = []
    primes = set(analysis.primes)
    transverse: bool | None = True
    total = 0
    strata: list[Stratum] = []
    partial = False
    for stratum in analysis.strata:
        if stratum.reason is not None or stratum.mu_t is None:
            partial = True
            reasons.append(f"a piece of the face {list(stratum.face)}: {stratum.reason}")
            strata.append(stratum)
            continue
        if stratum.mu_t == 0 or stratum.chart is None or stratum.dimension is None:
            strata.append(stratum)
            continue
        key = stratum.face
        if key not in h_faces:
            h_faces[key] = _face_h(data, stratum, h_terms, names[: stratum.chart.torus_dimension])
        try:
            euler, is_transverse, used = _piece(
                stratum,
                h_faces[key],
                seed=seed,
                backend=backend,
                timeout=timeout,
                singular=singular,
                msolve=msolve,
            )
        except ComputationError as exc:
            if not str(exc).startswith("undecided: "):
                raise
            partial = True
            transverse = None if transverse else transverse
            reasons.append(f"a piece of the face {list(key)}: {str(exc)[len('undecided: ') :]}")
            strata.append(stratum)
            continue
        primes.update(used)
        if not is_transverse:
            transverse = False
        total += stratum.mu_t * euler
        strata.append(dataclasses.replace(stratum, euler=euler))
    total *= index

    if partial:
        reasons.insert(0, "the sum is partial")
    elif not analysis.complete:
        reasons.append(
            "the strata are not known to be complete: "
            + (
                f"{len(analysis.failures)} condition(s) fail"
                if analysis.failures
                else "a face was computed through a subdivision"
            )
        )
    if transverse is False:
        reasons.append("H is not transverse to a piece; draw it again with another seed")
    undecided = partial or not analysis.complete or transverse is not True
    return StratumSum(
        volume=volume,
        master_count=counts[0],
        strata=tuple(strata),
        total=None if partial else total,
        agrees=None if undecided else total == volume - counts[0],
        transverse=transverse,
        primes=tuple(sorted(primes)),
        seed=seed,
        complete=analysis.complete,
        reason="; ".join(reasons) or None,
    )


def stratum_sum_from_polynomial(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    *,
    scale: sp.Symbol | None = None,
    seed: int = 0,
    timeout: float = 120,
    backend: str = "singular",
) -> StratumSum:
    """The stratum sum of a polynomial: the drop of its number of critical points, and the sum.

    Computes the volume of the Newton polytope P of G, |chi| by
    :func:`~feynkit.point_count.critical_point_count` at the seeds ``seed`` and ``seed + 1``
    (which must agree), the strata of :func:`singular_strata_from_polynomial` and
    sum mu^T_S chi(S minus V(H)) over them, H a polynomial with random coefficients on every
    lattice point of P, and compares the sum with vol - |chi|. The identity and its sources
    are in :class:`StratumSum`. A piece of the strata with mu^T = 0 does not count, so its
    Euler characteristic is not computed.

    ``agrees`` is None, never True, when the strata may be incomplete or partial: when
    :func:`singular_strata_from_polynomial` leaves any piece undecided (a ``reason`` set) or
    ``complete`` is False, when an Euler characteristic is undecided, and when H is not
    transverse to a piece. ``reason`` then says why, and ``total`` is None where the sum is
    partial.

    Parameters
    ----------
    polynomial, variables, point, scale
        As for :func:`singular_strata_from_polynomial`. The Newton polytope of G at the point
        must be full-dimensional.
    seed
        Seed of the strata, the random exponents of the counts and the coefficients of H.
    timeout
        The most seconds each Singular or msolve run gets, at most 2,000,000.
    backend
        "singular" or "msolve", for the counts of critical points of the Euler characteristics
        and of |chi|; the decompositions stay in Singular.

    Raises
    ------
    RuntimeError
        If Singular, or msolve for ``backend="msolve"``, is not installed.
    ValidationError
        If the arguments are not as for :func:`singular_strata_from_polynomial`, ``backend``
        is neither "singular" nor "msolve", or the Newton polytope is not full-dimensional.
    ComputationError
        If the counts of critical points at the two seeds differ, or a solver fails in a way
        other than being undecided.
    """
    return _stratum_sum(polynomial, variables, point, scale, seed, timeout, backend)


def stratum_sum(
    integral: FeynmanIntegral,
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    *,
    seed: int = 0,
    timeout: float = 120,
    backend: str = "singular",
) -> StratumSum:
    """The stratum sum for G = U + F of an integral at a rational kinematic point.

    The point is keyed as :attr:`~feynkit.point_count.TorusCount.point` gives it, with the
    energy scale set to 1. See :func:`stratum_sum_from_polynomial` for the arguments, the
    errors and when ``agrees`` is None, and :class:`StratumSum` for the identity.

    Raises
    ------
    ValidationError
        Also if the integral has kinematic constraints, which are not applied.
    """
    if integral.kinematic_constraints:
        raise ValidationError(
            "stratum_sum does not apply kinematic_constraints; substitute them in the "
            "momentum products"
        )
    sym = integral.symanzik
    return _stratum_sum(
        sym.g, list(sym.lp_parameters), point, integral.graph.energy_scale, seed, timeout, backend
    )
