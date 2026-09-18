"""
Schwinger parametrisation for Feynman integrals.

The Schwinger parametrisation uses parameters alpha_i in [0, infty) for each propagator,
transforming the Feynman integral into an integral over Schwinger parameters.
"""

import sympy as sp
import numpy as np

from ..core.constants import ALPHA_PARAM_PREFIX, DEFAULT_EPSILON, DEFAULT_GAMMA_E
from ..core.exceptions import ComputationError
from ..core.graph import Graph
from .base import Parametrisation, ParametrisationResult
from ..systems.gkz import construct_gkz_matrix


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
            I = prefactor × int_0^infty [prod_i d alpha_i alpha_i^(nu_i-1)] U^(-D/2) exp(-F/U)

        where:
        - prefactor = exp(L·epsilon·gamma_E) / prod_i[Gamma(nu_i)]
        - measure = prod_i[d alpha_i alpha_i^(nu_i-1)]
        - integrand = U^(-D/2) · exp(-F/U)

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

            # Define regularisation symbols
            epsilon = sp.Symbol(DEFAULT_EPSILON, real=True)
            gamma_e = sp.Symbol(DEFAULT_GAMMA_E, real=True)
            loop_order = sp.Integer(self.loop_count)

            # === Compute prefactor ===
            # prefactor = exp(L·epsilon·gamma_E) / prod_i[Gamma(nu_i)]
            gamma_product = sp.prod(
                [sp.gamma(self.propagator_exponents[e.idx]) for e in internal_edges]
            )
            prefactor = sp.exp(loop_order * epsilon * gamma_e) / gamma_product

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
            # integrand = U^(-D/2) · exp(-F/U)
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
    ) -> tuple[tuple[sp.Expr, sp.Expr], list[sp.Symbols]]:
        """
        Apply the substitution needed to obtain the dehomogenised Symanzik polynomials. This can be used to bring the Schwinger parametrisation into the form of a GKZ integral. Specifically, the substitution is {α_1, ..., α_N} -> {tu_1, ..., tu_{N-1}, t}, based on work from https://arxiv.org/abs/2609.16107v1.

        Returns
        ---
        tuple[sp.Expr, sp.Expr]
            Dehomomgenised Symanzik polynomials Ũ and F̃ in the new variables

        list[sp.Symbols]
            List of the new variables {u_i}
        """

        t = sp.symbols("t")
        alpha = [
            sp.Symbol(f"{ALPHA_PARAM_PREFIX}_{e.idx}", nonnegative=True, real=True)
            for e in self.internal_edges
        ]
        u = [sp.Symbol(f"{u}_{e.idx}", nonnegative=True, real=True) for e in self.internal_edges]

        new_vars = tuple(t * ui for ui in new_vars)
        substitution = dict(zip(alpha, new_vars))

        u_tilde_polynomial = self.u_polynomial.subs(substitution)
        f_tilde_polynomial = self.f_polynomial.subs(substitution)

        return [u_tilde_polynomial, f_tilde_polynomial], u

    def get_A_matrix(self) -> sp.Matrix:
        """
        Get the \mathcal{A}-matrix of the GKZ integral obtained when manipulating the Schwinger representation into the form of an A-hypergeometric function. See https://arxiv.org/abs/2609.16107v1 for a full treatment.

        Returns:
        ---
        A-matrix
            The \mathcal{A}-matrix of the GKZ integral. Because the integral consists of a product of two functions, the dehomogenised Symanzik polynomials Ũ and F̃, the \mathcal{A}-matrix will necessarily be of the form::

            1 ... 1 | 0 ... 0
            0 ... 0 | 1 ... 1
            -----------------
              A_1   |   A_1
        """

        [u_tilde_polynomial, f_tilde_polynomial], u = self.dehomogenised_symanzik_polynomials(self)

        A_1 = construct_gkz_matrix(u_tilde_polynomial, u)
        A_2 = construct_gkz_matrix(f_tilde_polynomial, u)

        _, n = A_1.shape
        _, N = A_2.shape

        head = np.zeros((2, (n + N)), dtype=int)

        for i in range(2):
            for j in range((n + N)):

                if ((i == 0) & (j < n)) or ((i == 1) & (j >= n)):

                    head[i][j] = 1

        return sp.Matrix(np.r_[head, np.c_[A_1, A_2]])
