"""
Cayley GKZ system of the Schwinger representation.

Substituting alpha = (t u_1, ..., t u_{N-1}, t) into the Schwinger
representation and integrating out t gives a generalised Euler integral in
two polynomials, the dehomogenised Symanzik polynomials U~ and F~
(Jimenez-Santacruz, Lopez-Arcos and Quintero Velez, arXiv:2609.16107,
section 3; the same configuration is the UF Cayley form of Klausen,
arXiv:2302.13184, section 3.4):

    Phi = int prod_{i<N} du_i u_i^(nu_i - 1) U~(u)^(nu - (L+1) D/2) F~(u)^(L D/2 - nu),

with nu the sum of all N propagator exponents. The A-matrix is the Cayley
configuration: two rows of ones, one per polynomial, above the exponent
blocks of U~ and F~. In the convention sum_j A_rj z_j d/dz_j Phi = beta_r Phi
the parameter vector is

    beta = (nu - (L+1) D/2, L D/2 - nu, -nu_1, ..., -nu_{N-1}),

each entry the exponent of the corresponding polynomial or minus the
exponent of u_i. This is what the paper's worked examples use (eqs. 53, 54
and the triangle and box); its section 2.3 writes the first two entries
with the opposite sign, and its eq. 46 drops L from Gamma(nu - L D/2) and
F~^(L D/2 - nu); the general-L form is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ..algebra.toric import compute_toric_ideal_generators
from .complete import GKZSystem
from .euler import create_euler_equations
from .monomial import extract_monomial_support

__all__ = ["CayleyGKZSystem", "cayley_matrix", "create_cayley_system"]

Support = list[tuple[tuple[int, ...], sp.Expr]]


def _sorted_support(poly: sp.Expr, variables: list[sp.Symbol]) -> Support:
    support = extract_monomial_support(sp.expand(poly), variables)
    return sorted(support, key=lambda item: (-sum(item[0]), tuple(-e for e in item[0])))


def cayley_matrix(
    u_tilde: sp.Expr, f_tilde: sp.Expr, u_variables: list[sp.Symbol]
) -> tuple[sp.Matrix, Support, Support]:
    """Cayley matrix of the pair (U~, F~) and the sorted supports of the two blocks.

    Columns are the U~ monomials followed by the F~ monomials, each block in
    descending total degree then descending lexicographic order.
    """
    u_support = _sorted_support(u_tilde, u_variables)
    f_support = _sorted_support(f_tilde, u_variables)
    n, m = len(u_support), len(f_support)
    d = len(u_variables)
    rows = [[1] * n + [0] * m, [0] * n + [1] * m]
    for i in range(d):
        rows.append([e[i] for e, _ in u_support] + [e[i] for e, _ in f_support])
    return sp.Matrix(rows), u_support, f_support


@dataclass(frozen=True)
class CayleyGKZSystem:
    """GKZ system of the Schwinger representation as a two-polynomial Euler integral.

    Attributes
    ----------
    a_matrix
        Cayley matrix of shape (N + 1) x (n + m).
    w_variables, z_variables
        Coefficient variables of U~ (the paper's w_j) and F~ (its z_j).
    u_support, f_support
        (exponent, coefficient) pairs of the two blocks, in column order.
    u_polynomial, f_polynomial, u_variables
        U~, F~ and the variables u_1 ... u_{N-1}.
    beta_parameters
        (nu - (L+1) D/2, L D/2 - nu, -nu_1, ..., -nu_{N-1}).
    euler_equations
        One equation per row of ``a_matrix``.
    prefactor
        Factor relating the Feynman integral to the Euler integral Phi.
    dimension, propagator_exponents, loop_count
        The inputs, kept for reference. ``propagator_exponents`` has all N
        entries; the last belongs to the parameter set to one.
    """

    a_matrix: sp.Matrix
    w_variables: list[sp.Symbol]
    z_variables: list[sp.Symbol]
    u_support: Support
    f_support: Support
    u_polynomial: sp.Expr
    f_polynomial: sp.Expr
    u_variables: list[sp.Symbol]
    beta_parameters: list[sp.Expr]
    euler_equations: list[sp.Equality]
    prefactor: sp.Expr
    dimension: sp.Expr
    propagator_exponents: list[sp.Expr]
    loop_count: int

    @property
    def variables(self) -> list[sp.Symbol]:
        """All coefficient variables in column order."""
        return list(self.w_variables) + list(self.z_variables)

    def toric_ideal(self, backend: str = "auto") -> list[sp.Expr]:
        """Generators of the toric ideal of the Cayley matrix in the w and z variables."""
        generators = compute_toric_ideal_generators(self.a_matrix, backend=backend)
        standard = sp.symbols(f"z_1:{self.a_matrix.cols + 1}")
        return [
            g.subs(dict(zip(standard, self.variables, strict=True)), simultaneous=True)
            for g in generators
        ]

    def restrict_to_f_block(self) -> GKZSystem:
        """The face subsystem on the F~ block, the paper's eq. 54.

        Returns the ordinary GKZ system with A = (1 ... 1; A_F) and
        beta = (L D/2 - nu, -nu_1, ..., -nu_{N-1}). Britto, Grimm and
        Hoefnagels (arXiv:2606.09978, section 2.2) show that solutions of a
        face subsystem are solutions of the full system, not the converse:
        rescaling the F~ coefficients along an exponent row rescales u,
        which changes U~^(nu - (L+1) D/2) unless that exponent vanishes. The
        reduced system therefore annihilates the Feynman integral itself only
        when nu = (L+1) D/2 (their section 8.1) or on cut contours (Vanhove,
        arXiv:1807.11466, section 3.2). The paper's comparison of its rank
        with the number of master integrals is the paper's observation, not
        a bound.
        """
        n = len(self.u_support)
        a_f = self.a_matrix[2:, n:]
        a_matrix = sp.Matrix.vstack(sp.Matrix([[1] * a_f.cols]), a_f)
        beta = [self.beta_parameters[1]] + list(self.beta_parameters[2:])
        return GKZSystem(
            a_matrix=a_matrix,
            z_variables=list(self.z_variables),
            support=list(self.f_support),
            beta_parameters=beta,
            euler_equations=create_euler_equations(a_matrix, beta, list(self.z_variables)),
            dimension=self.dimension,
            propagator_exponents=list(self.propagator_exponents[:-1]),
        )


def create_cayley_system(
    u_tilde: sp.Expr,
    f_tilde: sp.Expr,
    u_variables: list[sp.Symbol],
    *,
    dimension: sp.Expr,
    propagator_exponents: list[sp.Expr],
    loop_count: int,
    prefactor: sp.Expr | None = None,
) -> CayleyGKZSystem:
    """Build the Cayley system from the dehomogenised Symanzik polynomials.

    ``propagator_exponents`` lists all N exponents; the last one belongs to
    the Schwinger parameter that was set to one. ``prefactor`` is the
    Schwinger prefactor exp(L epsilon gamma_E) / prod Gamma(nu_i); the
    Gamma(nu - L D/2) from the t integration is multiplied on here.
    """
    if len(propagator_exponents) != len(u_variables) + 1:
        raise ValueError("Expected one propagator exponent more than there are u variables")
    a_matrix, u_support, f_support = cayley_matrix(u_tilde, f_tilde, u_variables)
    n, m = len(u_support), len(f_support)
    w = list(sp.symbols(f"w_1:{n + 1}"))
    z = list(sp.symbols(f"z_1:{m + 1}"))
    nu = sum(propagator_exponents)
    half_d = dimension / 2
    beta = [nu - (loop_count + 1) * half_d, loop_count * half_d - nu] + [
        -e for e in propagator_exponents[:-1]
    ]
    gamma = sp.gamma(nu - loop_count * half_d)
    return CayleyGKZSystem(
        a_matrix=a_matrix,
        w_variables=w,
        z_variables=z,
        u_support=u_support,
        f_support=f_support,
        u_polynomial=sp.expand(u_tilde),
        f_polynomial=sp.expand(f_tilde),
        u_variables=list(u_variables),
        beta_parameters=beta,
        euler_equations=create_euler_equations(a_matrix, beta, w + z),
        prefactor=(prefactor if prefactor is not None else sp.Integer(1)) * gamma,
        dimension=dimension,
        propagator_exponents=list(propagator_exponents),
        loop_count=loop_count,
    )
