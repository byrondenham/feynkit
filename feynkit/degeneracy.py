"""
Degenerate faces of the Newton polytope of G.

Let z be a rational kinematic point, G_z the polynomial G = U + F there with
the energy scale set to 1, and P_z its Newton polytope, the convex hull of the
exponents whose coefficients do not vanish at z. A face F of P_z, P_z itself
included, is degenerate at z when the restriction G_z|_F, the sum of the terms
of G_z whose exponents lie on F, has a singular point in the torus:
G_z|_F = u_1 dG_z|_F/du_1 = ... = u_N dG_z|_F/du_N = 0 has a solution in
(C^*)^N. When P_z is full-dimensional and no face is degenerate, |chi| of the
complement of {G_z = 0} in the torus is N! Vol(P_z), the value Bitoun, Bogner,
Klausen and Panzer (arXiv:1712.09215, Theorem 44, after Kouchnirenko) give for
almost all coefficients. For the principal A-determinant E_A of Gelfand,
Kapranov and Zelevinsky (Discriminants, Resultants, and Multidimensional
Determinants, 1994) is the A-resultant of the u_i dG_z/du_i and G_z (Ch. 10,
(1.1)), which vanishes exactly where they have a common zero on the toric
variety of P_z (Ch. 8, Prop. 2.1, and Ch. 3, (2.2)); that variety is the union
of one orbit for each face F (Ch. 5, Prop. 1.9), and a common zero on the
orbit of F is a singular point of G_z|_F in the torus. So E_A does not vanish
at z, and Fevola, Mizera and Telen (arXiv:2311.16219, Theorem 2.3) give the
equality. The design note docs/design/2026-10-01-face-degeneracy.md spells
the argument out. Where
coefficients vanish at z, P_z is smaller than the Newton polytope P of G for
generic kinematics, and the difference of the normalised volumes is reported as
support loss, not as degeneracy.

On the kinematic family itself, a face is generically degenerate when it is
degenerate on a Zariski-dense set of kinematic points. That holds exactly when
it is degenerate at the generic point, over the field Q(E) of rational
functions in the kinematic symbols, and exactly when a component of the face's
incidence variety projects onto a dense subset of the kinematic space, a
component that the principal Landau determinant leaves out (Fevola, Mizera and
Telen, Definition 3.5); :class:`~feynkit.landau.FaceDiscriminant` calls such a
face ``dominant``.

Each face is decided exactly, in its lattice coordinates t in (C^*)^d, d its
dimension, where G restricts to a monomial times a polynomial g_F(t); F is
degenerate exactly when g_F has a singular point in (C^*)^d.

- A vertex never is, since its coefficient does not vanish.
- Nor is a face whose points are affinely independent.
- An edge is degenerate exactly when the discriminant of g_F vanishes; its
  total Tjurina number is the degree of g_F less that of its square-free part.
- Any other face is degenerate exactly when the saturation
  S = (g_F, dg_F/dt_1, ..., dg_F/dt_d) : (t_1 ... t_d)^infinity is not the unit
  ideal, computed with Singular's Gröbner bases over Q, or over Q(E) in
  generic mode. dim S is the dimension of the singular locus in the torus and,
  when it is 0, the vector-space dimension of the quotient by S is the total
  Tjurina number.

The faces of an analysis go to Singular in one run with a time limit. When the
run passes the limit, the faces it decided are kept, the face it was on is run
alone, unless it already had the whole limit, and the rest run together again;
a face that passes the limit alone is left undecided, never guessed.

The same runs over F_p, at the largest prime below 2^29 that divides no
numerator or denominator of a coefficient, give a second verdict for every face
that needed Singular. A certificate S = (1) over Q reduces modulo all but
finitely many primes, so the two differ only at finitely many primes; the F_p
verdict is a probabilistic cross-check and never decides.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import sympy as sp

from .core.exceptions import ComputationError, ValidationError
from .landau import _check_timeout, _singular_binary, _unit_scale_is_exact
from .point_count import _given_point, _key, _volume_bound
from .polytope import lattice_coordinates, polytope_data
from .systems.monomial import extract_monomial_support

if TYPE_CHECKING:
    from .integral import FeynmanIntegral

__all__ = [
    "DEFAULT_TIMEOUT",
    "DegeneracyAnalysis",
    "FaceDegeneracy",
    "face_degeneracy",
    "face_degeneracy_from_polynomial",
]

# The seconds Singular gets for the faces of an analysis, and for a face run alone.
DEFAULT_TIMEOUT = 120

# Singular's prime fields stop at 2^29 in rings with parameters.
_PRIME_LIMIT = 2**29

Method = Literal["vertex", "simplex", "edge", "groebner"]

# The generator of the polynomials of a coefficient without kinematic symbols.
_NO_PARAMETER = sp.Dummy("c")


@dataclass(frozen=True)
class FaceDegeneracy:
    """Whether one face F of the Newton polytope P_z has a singular point in the torus.

    Attributes
    ----------
    point_indices
        The points of P_z on F, as indices into ``DegeneracyAnalysis.points``.
    dimension
        The dimension of F.
    degenerate
        Whether G restricted to F has a singular point in the torus; None
        when Singular ran past the time limit on F (see ``reason``).
    method
        How it was decided: "vertex", "simplex" (affinely independent
        points), "edge" (the discriminant in the lattice coordinate) or
        "groebner" (Singular).
    singular_dimension
        The dimension of the singular locus in the torus of F's lattice
        coordinates, -1 when it is empty; None when undecided. It does not
        depend on the choice of lattice.
    tjurina
        The total Tjurina number of that locus when it is finite, 0 when it
        is empty; None when it is not finite or F is undecided. It is taken
        in the lattice coordinates of F, those of the lattice that the
        differences of its points span: the sum over the singular points p
        of dim C[t]_p / (g_F, dg_F/dt_1, ..., dg_F/dt_d). In the coordinates
        of the saturated lattice aff(F) cap Z^N it is that number times the
        index of the one lattice in the other, since the change of
        coordinates is then an isogeny of that degree, a covering of the
        torus that keeps the local type of each singular point.
    modular
        The verdict over F_p at ``DegeneracyAnalysis.prime``, a probabilistic
        cross-check that never decides; None for faces decided without
        Singular, without the check, or past the time limit over F_p.
    reason
        Why F is undecided, or None.
    """

    point_indices: tuple[int, ...]
    dimension: int
    degenerate: bool | None
    method: Method
    singular_dimension: int | None
    tjurina: int | None
    modular: bool | None
    reason: str | None = None


@dataclass(frozen=True)
class DegeneracyAnalysis:
    """The degenerate faces of the Newton polytope of G, at a point or on the family.

    Attributes
    ----------
    mode
        "point" at a rational kinematic point, or "generic" at the generic
        point of the kinematic family, over the field of rational functions
        in its symbols.
    point
        The kinematic point as (quantity, value) pairs in the order of the
        symbol names, a symbol occurring only to even powers, such as a
        mass, given by its square, as :attr:`~feynkit.point_count.TorusCount.point`
        gives them; None in generic mode.
    points
        The points of P_z, the exponents whose coefficients do not vanish, in
        the order of the support of G; in generic mode, all of them, the
        order of :attr:`~feynkit.integral.FeynmanIntegral.newton_polytope`.
    volume
        N! Vol(P_z), 0 when P_z is not full-dimensional.
    support_loss
        N! Vol(P) - N! Vol(P_z), P the Newton polytope for generic
        kinematics; 0 in generic mode.
    faces
        Every face of P_z, P_z included, in the order of
        :attr:`~feynkit.polytope.PolytopeData.faces`.
    prime
        The prime of the F_p check, or None when it did not run.
    """

    mode: Literal["point", "generic"]
    point: tuple[tuple[sp.Expr, Fraction], ...] | None
    points: tuple[tuple[int, ...], ...]
    volume: int
    support_loss: int
    faces: tuple[FaceDegeneracy, ...]
    prime: int | None

    @property
    def degenerate_faces(self) -> tuple[FaceDegeneracy, ...]:
        """The faces decided degenerate, in the order of ``faces``."""
        return tuple(face for face in self.faces if face.degenerate)

    @property
    def undecided_faces(self) -> tuple[FaceDegeneracy, ...]:
        """The faces Singular did not decide within the time limit."""
        return tuple(face for face in self.faces if face.degenerate is None)

    @property
    def non_degenerate(self) -> bool | None:
        """True when no face is degenerate, False when one is, and None when none is found
        degenerate but some face is undecided."""
        if self.degenerate_faces:
            return False
        return None if self.undecided_faces else True


# --- the polynomial at the point -------------------------------------------------------


@dataclass(frozen=True)
class _Support:
    """The terms of G with non-zero coefficients, at the point or on the family."""

    points: tuple[tuple[int, ...], ...]
    coefficients: tuple[sp.Expr, ...]
    parameters: tuple[sp.Symbol, ...]
    point: tuple[tuple[sp.Expr, Fraction], ...] | None
    family_volume: int


def _fraction(c: sp.Expr, parameters: Sequence[sp.Symbol]) -> tuple[sp.Poly, sp.Poly]:
    """The numerator and denominator of a coefficient as polynomials over QQ in the parameters,
    or in a dummy generator when there are none."""
    num, den = sp.fraction(sp.together(c))
    gens = list(parameters) or [_NO_PARAMETER]
    try:
        return sp.Poly(num, *gens, domain="QQ"), sp.Poly(den, *gens, domain="QQ")
    except (sp.PolynomialError, sp.CoercionFailed) as exc:
        raise ValidationError(
            f"the coefficient {c} is not a rational function with rational coefficients of "
            "the kinematic symbols"
        ) from exc


def _even(polys: Sequence[sp.Poly], k: int) -> bool:
    """Whether the k-th generator occurs only to even powers."""
    return all(m[k] % 2 == 0 for poly in polys for m in poly.monoms())


def _nonzero(
    terms: Sequence[tuple[tuple[int, ...], sp.Expr]],
) -> list[tuple[tuple[int, ...], sp.Expr]]:
    """The terms whose coefficient is not identically zero as a rational function."""
    return [(e, c) for e, c in terms if sp.fraction(sp.together(sp.sympify(c)))[0].expand() != 0]


def _support(
    polynomial: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    point: Mapping[sp.Expr, int | Fraction] | None,
    scale: sp.Symbol | None,
) -> _Support:
    distinct = len(set(variables)) == len(variables)
    if not variables or not distinct or not all(isinstance(v, sp.Symbol) for v in variables):
        raise ValidationError("variables must be one or more distinct symbols")
    g = sp.expand(sp.sympify(polynomial))
    if g == 0:
        raise ValidationError("the polynomial is zero")
    everything = extract_monomial_support(g, list(variables))
    terms = _nonzero(everything)
    unit = scale is not None and (point is not None or _unit_scale_is_exact(terms, scale))
    if unit and scale in g.free_symbols:
        terms = _nonzero([(e, sp.sympify(c).subs(scale, 1)) for e, c in terms])
    if not terms:
        raise ValidationError("the polynomial is zero")
    # The symbols of a coefficient that cancels still count as parameters of the polynomial.
    listed = [(e, sp.sympify(c).subs(scale, 1) if unit else sp.sympify(c)) for e, c in everything]
    parameters = tuple(
        sorted(set().union(*(c.free_symbols for _, c in listed)), key=lambda x: x.name)
    )
    fractions = [_fraction(c, parameters) for _, c in terms]
    family_volume = _volume_bound(polytope_data([e for e, _ in terms]))
    if point is None:
        return _Support(
            points=tuple(e for e, _ in terms),
            coefficients=tuple(sp.together(c) for _, c in terms),
            parameters=parameters,
            point=None,
            family_volume=family_volume,
        )
    even = frozenset(
        x for k, x in enumerate(parameters) if _even([p for pair in fractions for p in pair], k)
    )
    values = _given_point(point, parameters, even)
    exact = [sp.Rational(v.numerator, v.denominator) for v in values]
    at = {x: sp.sqrt(v) if x in even else v for x, v in zip(parameters, exact, strict=True)}
    kept_points: list[tuple[int, ...]] = []
    kept: list[sp.Expr] = []
    for (exps, _), (num, den) in zip(terms, fractions, strict=True):
        bottom = sp.Rational(sp.expand(den.as_expr().subs(at)))
        if bottom == 0:
            raise ValidationError("a coefficient of the polynomial is undefined at the point")
        value = sp.Rational(sp.expand(num.as_expr().subs(at))) / bottom
        if value != 0:
            kept_points.append(exps)
            kept.append(value)
    if not kept:
        raise ValidationError("the polynomial vanishes identically at the point")
    return _Support(
        points=tuple(kept_points),
        coefficients=tuple(kept),
        parameters=(),
        point=tuple((_key(x, even), v) for x, v in zip(parameters, values, strict=True)),
        family_volume=family_volume,
    )


# --- edges -------------------------------------------------------------------------------


def _edge(coefficients: Sequence[sp.Expr], positions: Sequence[int]) -> tuple[bool, int]:
    """Whether sum_i c_i t^k_i has a multiple root, and its total Tjurina number.

    The lowest position is 0 and carries a vertex, so 0 is not a root and
    every root lies in the torus.
    """
    t = sp.Dummy("t")
    poly = sp.Poly(sum(c * t**k for c, k in zip(coefficients, positions, strict=True)), t)
    if poly.degree() < 2:
        return False, 0
    numerator, _ = sp.fraction(sp.together(sp.discriminant(poly)))
    if sp.expand(numerator) != 0:
        return False, 0
    return True, poly.degree() - poly.sqf_part().degree()


# --- Singular --------------------------------------------------------------------------------


def _render(c: sp.Expr, names: Mapping[sp.Symbol, str]) -> str:
    """A coefficient for Singular: a quotient of polynomials in the parameters p1, p2, ...."""

    def poly(p: sp.Poly) -> str:
        terms = []
        for monomial, value in p.terms():
            q = sp.Rational(value)
            powers = [
                f"{names[x]}^{e}" if e > 1 else names[x]
                for x, e in zip(p.gens, monomial, strict=True)
                if e
            ]
            terms.append("*".join([f"({q.p}/{q.q})", *powers]))
        return "+".join(terms) or "0"

    num, den = _fraction(c, list(names))
    if den.is_one:
        return f"({poly(num)})"
    return f"(({poly(num)})/({poly(den)}))"


def _rationals(c: sp.Expr, parameters: Sequence[sp.Symbol]) -> list[int]:
    """The numerators and denominators of the rational coefficients of a coefficient."""
    found = []
    for part in _fraction(c, parameters):
        for value in part.coeffs():
            q = sp.Rational(value)
            found += [abs(int(q.p)), int(q.q)]
    return found


def _prime(numbers: Sequence[int]) -> int:
    """The largest prime below 2^29 dividing none of the non-zero numbers."""
    p = _PRIME_LIMIT
    while True:
        p = int(sp.prevprime(p))
        if all(n % p for n in numbers if n):
            return p


def _block(k: int, face: tuple[int, str], parameters: Sequence[str], characteristic: int) -> str:
    """The Singular commands deciding one face: dimension and polynomial g_F in t1, ..., td."""
    d, g = face
    ts = [f"t{i}" for i in range(1, d + 1)]
    field = f"({characteristic},{','.join(parameters)})" if parameters else str(characteristic)
    return (
        f"//F {k}\n"
        f"ring r{k} = {field}, ({','.join(ts)}), dp;\n"
        f"poly g = {g};\n"
        f"def s0 = sat(ideal(g) + jacob(g), ideal({'*'.join(ts)}));\n"
        'ideal s; if (typeof(s0) == "list") { s = s0[1]; } else { s = s0; }\n'
        "s = std(s); ds = dim(s); tau = -1; if (ds == 0) { tau = vdim(s); }\n"
        f'print("F {k} " + string(ds) + " " + string(tau));\n'
        f"kill r{k};\n"
    )


def _script(
    faces: Mapping[int, tuple[int, str]], parameters: Sequence[str], characteristic: int
) -> str:
    blocks = [_block(k, face, parameters, characteristic) for k, face in faces.items()]
    return 'LIB "elim.lib";\nint ds; int tau;\n' + "".join(blocks) + "quit;\n"


def _run(script: str, binary: str, timeout: float) -> tuple[str, bool]:
    """What Singular prints for a script, and whether it ran past ``timeout`` seconds.

    Raises
    ------
    ComputationError
        If Singular exits with an error status.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "faces.sing"
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
    if result.returncode != 0:
        reason = " ".join(result.stderr.split())[:100] or f"exit status {result.returncode}"
        raise ComputationError(f"Singular failed deciding faces: {reason}")
    return result.stdout, False


_VERDICT = re.compile(r"F (\d+) (-?\d+) (-?\d+)")


def _read(text: str, faces: Sequence[int]) -> dict[int, tuple[int, int]]:
    wanted = set(faces)
    found = {}
    for line in text.splitlines():
        match = _VERDICT.fullmatch(line.strip())
        if match and int(match.group(1)) in wanted:
            k, ds, tau = (int(x) for x in match.groups())
            found[k] = (ds, tau)
    return found


def _decide(
    faces: Mapping[int, tuple[int, str]],
    parameters: Sequence[str],
    characteristic: int,
    binary: str,
    timeout: float,
) -> dict[int, tuple[int, int] | None]:
    """(dim S, total Tjurina number or -1) for each face, None for a face past the limit.

    Raises
    ------
    ComputationError
        If Singular fails or prints no verdict for a face within the limit.
    """
    results: dict[int, tuple[int, int] | None] = {}
    pending = list(faces)

    def run(keys: list[int]) -> tuple[dict[int, tuple[int, int]], bool]:
        text, timed_out = _run(
            _script({k: faces[k] for k in keys}, parameters, characteristic), binary, timeout
        )
        found = _read(text, keys)
        if not timed_out and len(found) != len(keys):
            first = next((line for line in text.splitlines() if "?" in line), text[:100])
            raise ComputationError(
                f"Singular printed verdicts for {len(found)} of {len(keys)} faces: "
                f"{' '.join(first.split())[:100]}"
            )
        return found, timed_out

    while pending:
        found, timed_out = run(pending)
        results.update(found)
        rest = [k for k in pending if k not in found]
        if not timed_out or not rest:
            break
        # Faces print in order, so the first face without a verdict is the one Singular was on.
        stuck, pending = rest[0], rest[1:]
        if not found:
            # It was first in the run, and so had the whole limit.
            results[stuck] = None
            continue
        alone, _ = run([stuck])
        results[stuck] = alone.get(stuck)
    return results


# --- the analysis ---------------------------------------------------------------------------


def _shortcut(
    indices: Sequence[int], dimension: int, method: Method, degenerate: bool, tau: int
) -> FaceDegeneracy:
    """A face decided without Singular, whose singular locus is finite."""
    return FaceDegeneracy(
        point_indices=tuple(indices),
        dimension=dimension,
        degenerate=degenerate,
        method=method,
        singular_dimension=0 if degenerate else -1,
        tjurina=tau,
        modular=None,
    )


def _analysis(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction] | None,
    *,
    scale: sp.Symbol | None,
    check: bool,
    timeout: float,
    shortcuts: bool = True,
) -> DegeneracyAnalysis:
    """The analysis of :func:`face_degeneracy_from_polynomial`; without ``shortcuts`` every
    face but a vertex goes to Singular, which the tests compare with the shortcuts."""
    _check_timeout(timeout, optional=False)
    support = _support(polynomial, tuple(variables), point, scale)
    data = polytope_data(list(support.points))
    names = {x: f"p{i}" for i, x in enumerate(support.parameters, start=1)}

    decided: dict[int, FaceDegeneracy] = {}
    pending: dict[int, tuple[int, str]] = {}
    numbers: list[int] = []
    for k, (dimension, indices) in enumerate(data.faces):
        coefficients = [support.coefficients[i] for i in indices]
        if dimension == 0:
            decided[k] = _shortcut(indices, dimension, "vertex", False, 0)
            continue
        if shortcuts and len(indices) == dimension + 1:
            decided[k] = _shortcut(indices, dimension, "simplex", False, 0)
            continue
        coordinates = lattice_coordinates([support.points[i] for i in indices])
        if shortcuts and dimension == 1:
            found, tau = _edge(coefficients, [c[0] for c in coordinates])
            decided[k] = _shortcut(indices, dimension, "edge", found, tau)
            continue
        terms = [
            _render(c, names) + "".join(f"*t{j}^{e}" for j, e in enumerate(exps, start=1) if e)
            for c, exps in zip(coefficients, coordinates, strict=True)
        ]
        pending[k] = (dimension, "+".join(terms))
        for c in coefficients:
            numbers += _rationals(c, support.parameters)

    prime = None
    exact: dict[int, tuple[int, int] | None] = {}
    modular: dict[int, tuple[int, int] | None] = {}
    if pending:
        binary = _singular_binary()
        if binary is None:
            raise RuntimeError(
                "face_degeneracy needs Singular for faces of dimension 2 or more that are not "
                "simplices, and it was not found"
            )
        parameters = list(names.values())
        exact = _decide(pending, parameters, 0, binary, timeout)
        if check:
            prime = _prime(numbers)
            modular = _decide(pending, parameters, prime, binary, timeout)
    for k, (dimension, _) in pending.items():
        verdict = exact[k]
        other = modular.get(k)
        indices = data.faces[k][1]
        if verdict is None:
            decided[k] = FaceDegeneracy(
                point_indices=tuple(indices),
                dimension=dimension,
                degenerate=None,
                method="groebner",
                singular_dimension=None,
                tjurina=None,
                modular=None if other is None else other[0] >= 0,
                reason=f"Singular ran past timeout={timeout} s",
            )
            continue
        ds, tau = verdict
        decided[k] = FaceDegeneracy(
            point_indices=tuple(indices),
            dimension=dimension,
            degenerate=ds >= 0,
            method="groebner",
            singular_dimension=ds,
            tjurina=0 if ds < 0 else (tau if ds == 0 else None),
            modular=None if other is None else other[0] >= 0,
        )
    volume = _volume_bound(data)
    return DegeneracyAnalysis(
        mode="generic" if point is None else "point",
        point=support.point,
        points=support.points,
        volume=volume,
        support_loss=0 if point is None else support.family_volume - volume,
        faces=tuple(decided[k] for k in range(len(data.faces))),
        prime=prime,
    )


def face_degeneracy_from_polynomial(
    polynomial: sp.Expr,
    variables: Sequence[sp.Symbol],
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    *,
    scale: sp.Symbol | None = None,
    check: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> DegeneracyAnalysis:
    """Decide which faces of the Newton polytope of a polynomial are degenerate.

    Parameters
    ----------
    polynomial
        G, polynomial in ``variables`` with coefficients that are rational
        functions, with rational coefficients, of the other symbols.
    variables
        The N variables of G.
    point
        A rational kinematic point, keyed as for
        :func:`~feynkit.point_count.critical_point_count`: by symbol or, for a
        symbol occurring only to even powers, such as a mass, by its square.
        None, the default, decides each face at the generic point of the
        family, over the field of rational functions in the symbols.
    scale
        The energy scale mu, if G carries one. A point sets it to 1. In
        generic mode it is set to 1 when every coefficient is a power of mu
        times an expression free of it, with the power an affine function of
        the exponents, as for G = U + F: rescaling mu then acts on the
        coefficients through the torus, which maps the singular locus of
        every face onto that at mu = 1. Otherwise mu stays a symbol of the
        family.
    check
        Whether to decide the faces that need Singular a second time over
        F_p, as a probabilistic cross-check.
    timeout
        The most seconds Singular gets for the faces together, and for a face
        run alone, at most 2,000,000; 120 by default. Each check takes as
        long again.

    Raises
    ------
    ValidationError
        If ``variables`` are not one or more distinct symbols, the polynomial
        is zero or vanishes at the point, a coefficient is not a rational
        function with rational coefficients of the other symbols or is
        undefined at the point, ``point`` misses a symbol, has another key or
        gives a value that is not rational, or ``timeout`` is not a number of
        seconds greater than 0 and at most 2,000,000.
    RuntimeError
        If a face needs Singular and it is not installed.
    ComputationError
        If Singular fails or prints no verdict for a face.
    """
    return _analysis(polynomial, variables, point, scale=scale, check=check, timeout=timeout)


def face_degeneracy(
    integral: FeynmanIntegral,
    point: Mapping[sp.Expr, int | Fraction] | None = None,
    *,
    check: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> DegeneracyAnalysis:
    """Decide which faces of the Newton polytope of G = U + F are degenerate.

    At a rational kinematic point, keyed as
    :attr:`~feynkit.point_count.TorusCount.point` gives it, with the energy
    scale set to 1, or, by default, at the generic point of the integral's
    kinematics. See :func:`face_degeneracy_from_polynomial` for the arguments
    and errors.

    Raises
    ------
    ValidationError
        Also if the integral has kinematic constraints, which are not applied.
    """
    if integral.kinematic_constraints:
        raise ValidationError(
            "face_degeneracy does not apply kinematic_constraints; substitute them in the "
            "momentum products"
        )
    sym = integral.symanzik
    return face_degeneracy_from_polynomial(
        sym.g,
        list(sym.lp_parameters),
        point,
        scale=integral.graph.energy_scale,
        check=check,
        timeout=timeout,
    )
