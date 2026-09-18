"""
Lee-Pomeransky parametrisation for Feynman integrals.

The Lee-Pomeransky parametrisation uses the combined polynomial G = U + F,
which is particularly useful for finding critical points and dimensional
recurrence relations.
"""

import sympy as sp

from ..core.constants import DEFAULT_EPSILON, DEFAULT_GAMMA_E, LEE_POMERANSKY_PARAM_PREFIX
from ..core.exceptions import ComputationError
from ..core.graph import Graph
from .base import Parametrisation, ParametrisationResult


class LeePomeranskyParametrisation(Parametrisation):
    """
    Lee-Pomeransky parametrisation of Feynman integrals.

    Lee-Pomeransky uses parameters u with a combined polynomial G = U + F.
    This form is particularly useful for:
    - Finding critical points of the integrand
    - Deriving dimensional recurrence relations
    - Studying singularity structure

    Attributes
    ----------
    graph : Graph
        The Feynman graph to parametrise.
    dimension : sp.Expr
        Spacetime dimension D.
    loop_count : int
        Number of independent loops L.
    propagator_exponents : Dict[int, sp.Expr]
        Propagator exponents {edge_idx: nu}.
    u_polynomial : sp.Expr
        First Symanzik polynomial U.
    f_polynomial : sp.Expr
        Second Symanzik polynomial F.

    References
    ----------
    .. [1] Lee, R.N. and Pomeransky, A.A. (2013). "Critical points and number of
            master integrals." JHEP 11, 165.
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
        return "Lee-Pomeransky"

    def _build_lp_substitution(
        self,
    ) -> tuple[list[sp.Symbol], dict[sp.Symbol, sp.Expr], sp.Expr, sp.Expr]:
        """Build LP parameters, substitution map, and substituted U/F polynomials."""
        internal_edges = self.graph.get_internal_edges()
        lp_params = [
            sp.Symbol(f"{LEE_POMERANSKY_PARAM_PREFIX}_{e.idx}", nonnegative=True, real=True)
            for e in internal_edges
        ]
        param_substitution = {
            self.graph.schwinger_parameters[e.idx]: lp_params[i]
            for i, e in enumerate(internal_edges)
        }
        u_lp = self.u_polynomial.subs(param_substitution)
        f_lp = self.f_polynomial.subs(param_substitution)
        return lp_params, param_substitution, u_lp, f_lp

    def compute(self) -> ParametrisationResult:
        """
        Compute the Lee-Pomeransky parametrisation.

        Returns
        -------
        ParametrisationResult
            Complete Lee-Pomeransky parametrisation, including the G polynomial.

        Notes
        -----
        The Lee-Pomeransky representation is:
            I = prefactor × int_0^infty [prod_i du_i u_i^(nu_i-1)] G(u)^(-D/2)

        where:
        - G(u) = U(u) + F(u) is the combined polynomial
        - prefactor = exp(L·epsilon·gamma_E) · Gamma(D/2) / [Gamma((L+1)D/2 - sum_i nu_i) · prod_i Gamma(nu_i)]
        - measure = prod_i[du_i u_i^(nu_i-1)]
        - integrand = G(u)^(-D/2)
        - Each u_i in [0, infty)

        The G polynomial can be stored in the result for further analysis:
        >>> result = lee_pom.compute()
        >>> # Access G polynomial from integrand or compute separately

        Examples
        --------
        >>> lee_pom = LeePomeranskyParametrisation(graph, D, L, nus, U, F)
        >>> result = lee_pom.compute()
        >>> print(result.integrand)
        """
        try:
            lp_params, _, u_lp, f_lp = self._build_lp_substitution()
            internal_edges = self.graph.get_internal_edges()
            num_internal = len(internal_edges)

            # Combined polynomial G(u) = U(u) + F(u)
            g_polynomial = sp.expand(u_lp + f_lp)

            # Sum of all propagator exponents: sum_i nu_i
            nu_sum = sum([self.propagator_exponents[e.idx] for e in internal_edges])

            # Define regularisation symbols
            epsilon = sp.Symbol(DEFAULT_EPSILON, real=True)
            gamma_e = sp.Symbol(DEFAULT_GAMMA_E, real=True)
            loop_order = sp.Integer(self.loop_count)

            # === Compute prefactor ===
            # prefactor = exp(L·epsilon·gamma_E) · Gamma(D/2) / [Gamma((L+1)D/2 - sum_i nu_i) · prod_i Gamma(nu_i)]
            gamma_numerator = sp.gamma(self.dimension / 2)
            gamma_denominator_1 = sp.gamma((loop_order + 1) * self.dimension / 2 - nu_sum)
            gamma_denominator_2 = sp.prod(
                [sp.gamma(self.propagator_exponents[e.idx]) for e in internal_edges]
            )

            prefactor = (
                sp.exp(loop_order * epsilon * gamma_e)
                * gamma_numerator
                / (gamma_denominator_1 * gamma_denominator_2)
            )

            # === Compute measure ===
            # measure = prod_i[du_i u_i^(nu_i-1)]
            measure = sp.prod(
                [
                    lp_params[i] ** (self.propagator_exponents[internal_edges[i].idx] - 1)
                    for i in range(num_internal)
                ]
            )

            # === Compute integrand ===
            # integrand = G(u)^(-D/2)
            integrand = sp.simplify(g_polynomial ** (-self.dimension / 2))

            # === Constraints ===
            # All u_i >= 0 (implicit in symbol definition)
            constraints: list[sp.Expr] = []

            description = (
                f"Lee-Pomeransky parametrisation with {num_internal} $u$ parameters. "
                f"Each $u \\in [0, \\infty)$. Uses combined polynomial $G = U + F$. "
                f"Integration domain is $\\mathbb{{R}}_+^{{{num_internal}}}$."
            )

            return ParametrisationResult(
                name=self.name,
                prefactor=prefactor,
                measure=measure,
                integrand=integrand,
                parameters=lp_params,
                constraints=constraints,
                description=description,
            )

        except Exception as e:
            raise ComputationError(f"Failed to compute Lee-Pomeransky parametrisation: {e}") from e

    def get_g_polynomial(self) -> sp.Expr:
        """
        Get the combined G polynomial: G = U + F.

        Returns
        -------
        sp.Expr
            The G polynomial in Lee-Pomeransky parameters.

        Examples
        --------
        >>> lee_pom = LeePomeranskyParametrisation(graph, D, L, nus, U, F)
        >>> G = lee_pom.get_g_polynomial()
        >>> print(G)
        """
        _, _, u_lp, f_lp = self._build_lp_substitution()
        return sp.expand(u_lp + f_lp)
