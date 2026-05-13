"""
Feynman parametrisation for Feynman integrals.

The Feynman parametrisation uses parameters constrained to a simplex: sum_i a_i = 1.
This eliminates one integration and is often more efficient than Schwinger parametrisation.
"""

import sympy as sp

from ..core.constants import DEFAULT_EPSILON, DEFAULT_GAMMA_E
from ..core.exceptions import ComputationError
from ..core.graph import Graph
from .base import Parametrisation, ParametrisationResult


class FeynmanParametrisation(Parametrisation):
    """
    Feynman parametrisation of Feynman integrals.

    Feynman parametrisation uses parameters on the simplex Sigma: {a >= 0, sum_i a_i = 1}.
    This is related to Schwinger parametrisation by rescaling to eliminate one variable.

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
        First Symanzik polynomial U(a).
    f_polynomial : sp.Expr
        Second Symanzik polynomial F(a).

    References
    ----------
    .. [1] Feynman, R.P. (1949). "Space-time approach to quantum electrodynamics."
            Phys. Rev. 76, 769.
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
        super().__init__(graph, dimension, loop_count, propagator_exponents, u_polynomial, f_polynomial)

    @property
    def name(self) -> str:
        """Return the name of this parametrisation."""
        return "Feynman"

    def compute(self) -> ParametrisationResult:
        """
        Compute the Feynman parametrisation.

        Returns
        -------
        ParametrisationResult
            Complete Feynman parametrisation.

        Notes
        -----
        The Feynman representation is:
            I = prefactor × int_Sigma [prod_i da_i a_i^(nu_i-1)] U^(sum_i nu_i - (L+1)D/2) / F^(sum_i nu_i - LD/2)

        where:
        - Sigma = {a >= 0, sum_i a_i = 1} is the unit simplex
        - prefactor = exp(L·epsilon·gamma_E) · Gamma(sum_i nu_i - LD/2) / prod_i[Gamma(nu_i)]
        - measure = prod_i[da_i a_i^(nu_i-1)] with delta(sum_i a_i - 1)
        - integrand = U^(sum_i nu_i - (L+1)D/2) / F^(sum_i nu_i - LD/2)

        Examples
        --------
        >>> feynman = FeynmanParametrisation(graph, D, L, nus, U, F)
        >>> result = feynman.compute()
        >>> print(result.integrand)
        """
        try:
            internal_edges = self.graph.get_internal_edges()
            num_internal = len(internal_edges)

            # Use the graph's Schwinger parameters (which serve as Feynman parameters)
            feynman_params = [self.graph.schwinger_parameters[e.idx] for e in internal_edges]

            # Sum of all propagator exponents: sum_i nu_i
            nu_sum = sum([self.propagator_exponents[e.idx] for e in internal_edges])

            # Define regularisation symbols
            epsilon = sp.Symbol(DEFAULT_EPSILON, real=True)
            gamma_e = sp.Symbol(DEFAULT_GAMMA_E, real=True)
            loop_order = sp.Integer(self.loop_count)

            # === Compute prefactor ===
            # prefactor = exp(L·epsilon·gamma_E) · Γ(sum_i nu_i - LD/2) / prod_i[Gamma(nu_i)]
            gamma_numerator = sp.gamma(nu_sum - loop_order * self.dimension / 2)
            gamma_denominator = sp.prod(
                [sp.gamma(self.propagator_exponents[e.idx]) for e in internal_edges]
            )
            prefactor = sp.exp(loop_order * epsilon * gamma_e) * gamma_numerator / gamma_denominator

            # === Compute measure ===
            # measure = prod_i[da_i a_i^(nu_i-1)] (with implicit delta(sum_i a_i - 1))
            measure = sp.prod(
                [
                    feynman_params[i] ** (self.propagator_exponents[internal_edges[i].idx] - 1)
                    for i in range(num_internal)
                ]
            )

            # === Compute integrand ===
            # integrand = U^(sum_i nu_i - (L+1)D/2) / F^(sum_i nu_i - LD/2)
            u_exponent = nu_sum - (loop_order + 1) * self.dimension / 2
            f_exponent = nu_sum - loop_order * self.dimension / 2

            integrand = sp.simplify(
                self.u_polynomial**u_exponent / self.f_polynomial**f_exponent
            )

            # === Constraints ===
            # Simplex constraint: sum_i a_i = 1, all a_i >= 0
            simplex_constraint = sp.Eq(sum(feynman_params), 1)
            constraints = [simplex_constraint]

            description = (
                f"Feynman parametrisation with {num_internal} parameters on the unit simplex. "
                f"Constraint: $\\sum_i a_i = 1$, all $a_i \\geq 0$."
            )

            return ParametrisationResult(
                name=self.name,
                prefactor=prefactor,
                measure=measure,
                integrand=integrand,
                parameters=feynman_params,
                constraints=constraints,
                description=description,
            )

        except Exception as e:
            raise ComputationError(f"Failed to compute Feynman parametrisation: {e}") from e
