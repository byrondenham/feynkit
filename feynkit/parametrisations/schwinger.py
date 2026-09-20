"""
Schwinger parametrisation for Feynman integrals.

The Schwinger parametrisation uses parameters alpha_i in [0, infty) for each propagator,
transforming the Feynman integral into an integral over Schwinger parameters.
"""

from __future__ import annotations

import sympy as sp

from ..core.constants import (
    ALPHA_PARAM_PREFIX,
    DEFAULT_EPSILON,
    DEFAULT_GAMMA_E,
    LEE_POMERANSKY_PARAM_PREFIX,
)
from ..core.exceptions import ComputationError
from ..core.graph import Graph
from ..systems.cayley import cayley_matrix
from .base import Parametrisation, ParametrisationResult


class SchwingerParametrisation(Parametrisation):
    """
    Schwinger parametrisation of Feynman integrals.

    In Schwinger parametrisation, each propagator 1/(p^2 - m^2)^nu is represented as:
        int_0^infty d alpha alpha^(nu-1) exp(-alpha(p^2 - m^2))

    The full integral becomes an integral over all alpha parameters with each alpha in [0, infty).

    Attributes
    ----------
    graph : Graph
        The Feynman graph to parametrise.
    dimension : sp.Expr
        Spacetime dimension D (typically symbolic, e.g., 4 - 2 epsilon).
    loop_count : int
        Number of independent loops L.
    propagator_exponents : Dict[int, sp.Expr]
        Propagator exponents {edge_idx: nu}.
    u_polynomial : sp.Expr
        First Symanzik polynomial U(alpha).
    f_polynomial : sp.Expr
        Second Symanzik polynomial F(alpha).

    References
    ----------
    .. [1] Schwinger, J. (1951). "On gauge invariance and vacuum polarization."
            Phys. Rev. 82, 664.
    .. [2] Weinzierl, S. (2022). "Feynman Integrals." Springer.
    """

    def __init__(
        self,
        graph: Graph,
        dimension: sp.Expr,
        loop_count: int,
        propagator_exponents: dict[int, sp.Expr],
        u_polynomial: sp.Expr,
        f_polynomial: sp.Expr,
    ):
        super().__init__(
            graph, dimension, loop_count, propagator_exponents, u_polynomial, f_polynomial
        )

    @property
    def name(self) -> str:
        """Return the name of this parametrisation."""
        return "Schwinger"

    @property
    def prefactor(self) -> sp.Expr:
        """exp(L epsilon gamma_E) / prod_i Gamma(nu_i), without computing the integrand."""
        internal_edges = self.graph.get_internal_edges()
        epsilon = sp.Symbol(DEFAULT_EPSILON, real=True)
        gamma_e = sp.Symbol(DEFAULT_GAMMA_E, real=True)
        gamma_product = sp.prod(
            [sp.gamma(self.propagator_exponents[e.idx]) for e in internal_edges]
        )
        return sp.exp(sp.Integer(self.loop_count) * epsilon * gamma_e) / gamma_product

    def compute(self) -> ParametrisationResult:
        """
        Compute the Schwinger parametrisation.

        Returns
        -------
        ParametrisationResult
            Complete Schwinger parametrisation.

        Notes
        -----
        The Schwinger representation is:
            I = prefactor x int_0^infty [prod_i d alpha_i alpha_i^(nu_i-1)] U^(-D/2) exp(-F/U)

        where:
        - prefactor = exp(L*epsilon*gamma_E) / prod_i[Gamma(nu_i)]
        - measure = prod_i[d alpha_i alpha_i^(nu_i-1)]
        - integrand = U^(-D/2) * exp(-F/U)

        Examples
        --------
        >>> schwinger = SchwingerParametrisation(graph, D, L, nus, U, F)
        >>> result = schwinger.compute()
        >>> print(result.integrand)
        """
        try:
            internal_edges = self.graph.get_internal_edges()
            num_internal = len(internal_edges)

            # Create Schwinger parameters alpha_i for each internal edge
            alpha_params = [
                sp.Symbol(f"{ALPHA_PARAM_PREFIX}_{e.idx}", nonnegative=True, real=True)
                for e in internal_edges
            ]

            # Substitute alpha parameters into Symanzik polynomials
            param_substitution = {
                self.graph.schwinger_parameters[e.idx]: alpha_params[i]
                for i, e in enumerate(internal_edges)
            }

            u_alpha = self.u_polynomial.subs(param_substitution)
            f_alpha = self.f_polynomial.subs(param_substitution)

            # === Compute prefactor ===
            # prefactor = exp(L*epsilon*gamma_E) / prod_i[Gamma(nu_i)]
            prefactor = self.prefactor

            # === Compute measure ===
            # measure = prod_i[d alpha_i alpha_i^(nu_i-1)]
            # We represent d alpha_i implicitly, storing only the weight alpha_i^(nu_i-1)
            measure = sp.prod(
                [
                    alpha_params[i] ** (self.propagator_exponents[internal_edges[i].idx] - 1)
                    for i in range(num_internal)
                ]
            )

            # === Compute integrand ===
            # integrand = U^(-D/2) * exp(-F/U)
            integrand = sp.simplify(u_alpha ** (-self.dimension / 2) * sp.exp(-f_alpha / u_alpha))

            # === Constraints ===
            # All alpha_i >= 0 (implicit in symbol definition)
            constraints: list[sp.Expr] = []

            description = (
                f"Schwinger parametrisation with {num_internal} alpha parameters. "
                f"Each $\\alpha \\in [0, \\infty)$. Integration domain is $\\mathbb{{R}}_+^{{{num_internal}}}$."
            )

            return ParametrisationResult(
                name=self.name,
                prefactor=prefactor,
                measure=measure,
                integrand=integrand,
                parameters=alpha_params,
                constraints=constraints,
                description=description,
            )

        except Exception as e:
            raise ComputationError(f"Failed to compute Schwinger parametrisation: {e}") from e

    def dehomogenised_symanzik_polynomials(
        self,
    ) -> tuple[tuple[sp.Expr, sp.Expr], list[sp.Symbol]]:
        """
        Dehomogenise the Symanzik polynomials for the GKZ form of the Schwinger integral.

        The substitution alpha = (t u_1, ..., t u_{N-1}, t) factors the
        homogeneous Symanzik polynomials as U(alpha) = t^L U~(u) and
        F(alpha) = t^{L+1} F~(u), which brings the Schwinger representation into
        the form of a GKZ integral (arXiv:2609.16107, section 3). Because U and
        F are homogeneous, U~ and F~ are just U and F with alpha_N = 1 and
        alpha_i = u_i for i < N.

        Returns
        -------
        tuple[sp.Expr, sp.Expr]
            Dehomogenised Symanzik polynomials U~ and F~ in the new variables.
        list[sp.Symbol]
            The new variables u_1, ..., u_{N-1}.
        """
        internal_edges = self.graph.get_internal_edges()
        new_vars = [
            sp.Symbol(f"{LEE_POMERANSKY_PARAM_PREFIX}_{e.idx}", nonnegative=True, real=True)
            for e in internal_edges[:-1]
        ]
        substitution: dict[sp.Symbol, sp.Expr] = {
            self.graph.schwinger_parameters[e.idx]: v
            for e, v in zip(internal_edges[:-1], new_vars, strict=True)
        }
        substitution[self.graph.schwinger_parameters[internal_edges[-1].idx]] = sp.Integer(1)

        u_tilde = sp.expand(self.u_polynomial.subs(substitution))
        f_tilde = sp.expand(self.f_polynomial.subs(substitution))
        return (u_tilde, f_tilde), new_vars

    def get_A_matrix(self) -> sp.Matrix:
        """
        GKZ A-matrix of the Schwinger representation written as an A-hypergeometric integral.

        See :mod:`feynkit.systems.cayley` for the block structure and conventions.

        Returns
        -------
        sp.Matrix
            Integer matrix of shape (N + 1) x (n + m) for N propagators, n
            monomials in U~ and m monomials in F~.
        """
        (u_tilde, f_tilde), u = self.dehomogenised_symanzik_polynomials()
        matrix, _, _ = cayley_matrix(u_tilde, f_tilde, u)
        return matrix
