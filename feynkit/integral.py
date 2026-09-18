"""
Unified Feynman integral object.

This module exposes :class:`FeynmanIntegral`, a single immutable façade that
gathers every representation of a Feynman integral — graph topology, Symanzik
and Lee-Pomeransky polynomials, parametric representations, the GKZ
hypergeometric system, the Newton polytope, and the toric ideal — as
``cached_property`` attributes derived lazily from one underlying
:class:`Graph` and its kinematic data.

The class is intentionally a thin layer over the existing free functions; the
heavy mathematical machinery lives in the dedicated modules
(``feynkit.polynomials``, ``feynkit.parametrisations``, ``feynkit.systems``,
``feynkit.algebra``, ``feynkit.normal_forms``).

Examples
--------
>>> import sympy as sp
>>> from feynkit import Edge, Graph, FeynmanIntegral
>>> e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
>>> e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
>>> ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
>>> ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)
>>> graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
>>> integral = FeynmanIntegral(graph)
>>> integral.symanzik.u            # first Symanzik polynomial
>>> integral.gkz.a_matrix          # GKZ A-matrix
>>> integral.toric_ideal.generators
>>> integral.with_(dimension=4)    # derive a related integral, fresh cache
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from .database import FeynkitDatabase

from .algebra.toric import compute_toric_ideal_generators
from .core.constants import LEE_POMERANSKY_PARAM_PREFIX
from .core.exceptions import ValidationError
from .core.graph import Graph
from .core.validation import (
    validate_dimension_parameter,
    validate_propagator_exponents,
)
from .kinematics.momentum import create_momentum_products
from .normal_forms.pairing_matrix import PairingMatrixResult, maximal_pairing_matrix
from .parametrisations.base import ParametrisationResult
from .parametrisations.factory import AllParametrisations, _create_parametrisations
from .polynomials.spanning_trees import gkz_exponent_vectors
from .systems.complete import GKZSystem, _create_gkz_system_direct
from .systems.monomial import extract_monomial_support
from .types import (
    NewtonPolytope,
    PolytopeAutomorphisms,
    PolytopeEquivalence,
    SymanzikPolynomials,
    ToricIdeal,
)

__all__ = [
    "FeynmanIntegral",
    "NewtonPolytope",
    "PolytopeAutomorphisms",
    "PolytopeEquivalence",
    "SymanzikPolynomials",
    "ToricIdeal",
]


# ──────────────────────────────────────────────────────────────────────────────
# The façade.
# ──────────────────────────────────────────────────────────────────────────────


_NU_PREFIX = "nu"
_DEFAULT_DIMENSION_NAME = "D"


class FeynmanIntegral:
    """
    Unified, immutable representation of a Feynman integral.

    The constructor only validates inputs and stores symbols; every
    representation is computed lazily on first access through a
    ``cached_property``. Two attributes accessed in succession therefore
    share intermediate results — for instance, ``integral.toric_ideal``
    re-uses the GKZ A-matrix already computed by ``integral.gkz``.

    Parameters
    ----------
    graph
        The :class:`Graph` whose integral is being studied.
    dimension
        The spacetime dimension. Defaults to a positive symbol ``D``.
    propagator_exponents
        Mapping ``edge_idx → ν_i``. Defaults to symbols ``nu_<i>`` for each
        internal edge.
    loop_count
        Number of independent loops. Defaults to ``graph.get_loop_count()``.
    momentum_products
        Mapping ``(i, j) → p_i · p_j`` for external legs. Defaults to symbolic
        products built by :func:`create_momentum_products`.
    use_mandelstam
        Forwarded to :func:`create_momentum_products` when generating default
        kinematics.
    kinematic_constraints
        Optional list of symbolic constraints on the kinematic variables. Not
        used by the current pipeline; preserved on derived instances by
        :meth:`with_`.

    Notes
    -----
    The object is treated as immutable. To explore variations, use
    :meth:`with_` to obtain a new instance with a fresh cache; do not mutate
    attributes in place.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        dimension: sp.Expr | None = None,
        propagator_exponents: dict[int, sp.Expr] | None = None,
        loop_count: int | None = None,
        momentum_products: dict[tuple[int, int], sp.Expr] | None = None,
        use_mandelstam: bool = True,
        kinematic_constraints: list[sp.Expr] | None = None,
        database: FeynkitDatabase | None = None,
    ) -> None:
        if dimension is None:
            dimension = sp.Symbol(_DEFAULT_DIMENSION_NAME, positive=True)
        if propagator_exponents is None:
            propagator_exponents = {
                e.idx: sp.Symbol(f"{_NU_PREFIX}_{e.idx}", positive=True)
                for e in graph.get_internal_edges()
            }
        if loop_count is None:
            loop_count = graph.get_loop_count()
        if momentum_products is None:
            n_external = max(graph.external_legs, 1)
            momentum_products = create_momentum_products(
                n_external=n_external, use_mandelstam=use_mandelstam
            )

        validate_dimension_parameter(dimension)
        validate_propagator_exponents(
            propagator_exponents,
            [e.idx for e in graph.get_internal_edges()],
        )
        expected_loops = graph.get_loop_count()
        if loop_count != expected_loops:
            raise ValidationError(
                f"loop_count={loop_count} does not match graph topology "
                f"(expected {expected_loops} loops from L = E - V + 1)"
            )

        self._graph = graph
        self._dimension = dimension
        self._loop_count = int(loop_count)
        self._propagator_exponents = dict(propagator_exponents)
        self._momentum_products = dict(momentum_products)
        self._use_mandelstam = bool(use_mandelstam)
        self._kinematic_constraints = list(kinematic_constraints or [])
        self._database = database

    # ──────── Read-only views of the input data ────────

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def dimension(self) -> sp.Expr:
        return self._dimension

    @property
    def loop_count(self) -> int:
        return self._loop_count

    @property
    def propagator_exponents(self) -> dict[int, sp.Expr]:
        return dict(self._propagator_exponents)

    @property
    def momentum_products(self) -> dict[tuple[int, int], sp.Expr]:
        return dict(self._momentum_products)

    @property
    def use_mandelstam(self) -> bool:
        return self._use_mandelstam

    @property
    def kinematic_constraints(self) -> list[sp.Expr]:
        return list(self._kinematic_constraints)

    # ──────── Constructors from CNickel ────────

    @classmethod
    def from_cnickel(cls, cnickel: str, **kwargs: Any) -> FeynmanIntegral:
        """
        Construct a :class:`FeynmanIntegral` from a CNickel string.

        Delegates graph construction to :meth:`Graph.from_cnickel`, then
        wraps the result in a :class:`FeynmanIntegral` with default symbolic
        propagator exponents (``nu_<idx>``) and spacetime dimension (``D``).
        Any keyword argument accepted by :class:`FeynmanIntegral` can be
        passed to override the defaults.

        Parameters
        ----------
        cnickel
            CNickel string, e.g. ``"12e|2e|e|:nzz"`` or ``"12e|2e|e|"``.
        **kwargs
            Forwarded to :class:`FeynmanIntegral` (``dimension``,
            ``propagator_exponents``, ``momentum_products``, ``database``, …).

        Examples
        --------
        >>> fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        >>> fi.cnickel
        '12e|2e|e|:zzz'
        >>> fi.toric_ideal.generators
        """
        return cls(Graph.from_cnickel(cnickel), **kwargs)

    @classmethod
    def from_nickel(cls, nickel: str, **kwargs: Any) -> FeynmanIntegral:
        """
        Construct a massless :class:`FeynmanIntegral` from a bare Nickel string.

        Equivalent to ``FeynmanIntegral.from_cnickel(nickel)``.

        Examples
        --------
        >>> fi = FeynmanIntegral.from_nickel("12e|2e|e|")
        >>> fi.cnickel
        '12e|2e|e|:zzz'
        """
        return cls.from_cnickel(nickel, **kwargs)

    # ──────── Topology ────────

    @property
    def nickel_index(self) -> str:
        """Canonical Nickel topology string (delegates to :meth:`Graph.nickel_index`)."""
        return self._graph.nickel_index()

    @property
    def cnickel(self) -> str:
        """Colored Nickel index: topology + mass coloring (``"topology:colors"``)."""
        return self._graph.cnickel()

    # ──────── Internal: shared parametrisation factory ────────

    @cached_property
    def _all_parametrisations(self) -> AllParametrisations:
        return _create_parametrisations(
            graph=self._graph,
            dimension=self._dimension,
            loop_count=self._loop_count,
            propagator_exponents=self._propagator_exponents,
            momentum_products=self._momentum_products,
        )

    # ──────── Polynomials ────────

    @cached_property
    def symanzik(self) -> SymanzikPolynomials:
        """All graph polynomials together: U, F (Schwinger and LP forms) and G."""
        ap = self._all_parametrisations
        # Reuse the Lee–Pomeransky helper that builds the canonical a → u
        # substitution; this guarantees the LP parameters here match those
        # used by the GKZ system below.
        lp_params, _, u_lp, f_lp = ap.lee_pomeransky._build_lp_substitution()
        edges = self._graph.get_internal_edges()
        return SymanzikPolynomials(
            u=ap.u_polynomial,
            f=ap.f_polynomial,
            u_lp=u_lp,
            f_lp=f_lp,
            g=sp.expand(u_lp + f_lp),
            schwinger_parameters=[self._graph.schwinger_parameters[e.idx] for e in edges],
            lp_parameters=list(lp_params),
        )

    # ──────── Parametric representations ────────

    @cached_property
    def schwinger(self) -> ParametrisationResult:
        return self._all_parametrisations.schwinger.compute()

    @cached_property
    def feynman(self) -> ParametrisationResult:
        return self._all_parametrisations.feynman.compute()

    @cached_property
    def lee_pomeransky(self) -> ParametrisationResult:
        return self._all_parametrisations.lee_pomeransky.compute()

    # ──────── Algebraic-geometry representations ────────

    @cached_property
    def gkz(self) -> GKZSystem:
        graph = self._graph
        edges = graph.get_internal_edges()
        n_int = graph.internal_vertices
        edge_pairs = [(e.v1 - 1, e.v2 - 1) for e in edges]
        lp_params = [
            sp.Symbol(f"{LEE_POMERANSKY_PARAM_PREFIX}_{e.idx}", nonnegative=True, real=True)
            for e in edges
        ]
        leg_to_vertex: dict[int, int] = {}
        for ext_edge in graph.get_external_edges():
            leg_to_vertex[ext_edge.v2 - n_int] = ext_edge.v1 - 1
        masses = [e.get_mass() for e in edges]
        exponent_vecs = gkz_exponent_vectors(
            n_int, edge_pairs, leg_to_vertex, self._momentum_products, masses
        )
        propagator_list = [self._propagator_exponents[e.idx] for e in edges]
        return _create_gkz_system_direct(exponent_vecs, lp_params, self._dimension, propagator_list)

    @cached_property
    def newton_polytope(self) -> NewtonPolytope:
        from .systems.gkz import construct_gkz_matrix_from_exponents

        s = self.symanzik
        support = extract_monomial_support(s.g, s.lp_parameters)
        exponent_vecs = [alpha for alpha, _ in support]
        a_matrix = construct_gkz_matrix_from_exponents(exponent_vecs, len(s.lp_parameters))
        return NewtonPolytope(
            support=support,
            a_matrix=a_matrix,
            parameters=list(s.lp_parameters),
        )

    @cached_property
    def polytope_automorphisms(self) -> PolytopeAutomorphisms:
        """
        Unimodular automorphism group of the Newton polytope of G.

        Computes Aut(P) = {(U, t) : U ∈ GL_n(ℤ), |det U|=1, t ∈ ℤⁿ,
        {Up + t : p ∈ P} = P}.  The identity is always included; for generic
        polytopes it is the only element.

        For highly symmetric diagrams (massless triangle → S₃, massless box
        → D₄, bananas → Sₙ) the group is non-trivial and reflects the
        symmetry of the GKZ system.

        See Also
        --------
        graph_automorphisms : vertex-permutation subgroup.
        feynkit.normal_forms.coefficient_preserving_indices : symmetries
            relevant for functional equations.
        """
        from .normal_forms.polytope_automorphisms import compute_polytope_automorphisms

        return compute_polytope_automorphisms(self.newton_polytope.points)

    @cached_property
    def symmetry_pairs(self) -> list:
        """
        All integer affine self-maps of the GKZ A-configuration.

        Returns every :class:`SymmetryPair` (M, t, P) satisfying T·A = A·Π_P,
        where T = [[1, 0^T], [t, M]] is an invertible integer matrix and P is
        a column permutation.  Each pair gives the Feynman-integral identity:

            I_A(β, z) = I_A(T β, z_P)

        where z_P = (z_{P(0)}, …, z_{P(N-1)}) (de la Cruz 2024).

        The unimodular subset (``pair.is_unimodular``) coincides with the
        polytope automorphism group from :attr:`polytope_automorphisms`.
        Non-unimodular pairs (``|det M| > 1``) give transformation identities
        between different parameter regimes of the same integral.

        Notes
        -----
        Operates on all N monomials of G (not only hull vertices) so that
        interior monomials are correctly accounted for.
        """
        from .a_configuration import AConfiguration
        from .a_configuration import symmetry_pairs as _symmetry_pairs

        return _symmetry_pairs(AConfiguration(self.gkz.a_matrix))

    @property
    def graph_automorphisms(self) -> list[list[int]]:
        """
        Vertex permutations of the Feynman graph preserving topology and masses.

        Each element is a list of length V (internal vertices) giving the new
        1-indexed label for each original vertex.  The identity ``[1, 2, …, V]``
        is always included.

        These are the graph-theoretic automorphisms; the polytope automorphism
        group can be larger (e.g. the 3-propagator banana has vertex
        automorphism ℤ/2 but polytope automorphism S₃).
        """
        from .normal_forms.polytope_automorphisms import compute_graph_automorphisms

        return compute_graph_automorphisms(self._graph)

    @cached_property
    def toric_ideal(self) -> ToricIdeal:
        gkz = self.gkz

        if self._database is not None:
            cached = self._database._lookup_toric(self.newton_polytope.points)
            if cached is not None:
                return ToricIdeal(
                    generators=cached,
                    a_matrix=gkz.a_matrix,
                    z_variables=list(gkz.z_variables),
                )

        generators = compute_toric_ideal_generators(gkz.a_matrix)

        if self._database is not None:
            self._database._store_toric(self, generators)

        return ToricIdeal(
            generators=generators,
            a_matrix=gkz.a_matrix,
            z_variables=list(gkz.z_variables),
        )

    # ──────── Actions ────────

    def with_(self, **overrides: Any) -> FeynmanIntegral:
        """
        Return a new :class:`FeynmanIntegral` with the given fields replaced.

        The cache is fresh; nothing is shared with ``self`` beyond the
        unchanged constructor arguments. If ``graph`` is overridden but
        ``loop_count`` is not, the loop count is re-derived from the new
        graph topology.
        """
        new_graph = overrides.pop("graph", self._graph)
        loop_count = overrides.pop(
            "loop_count",
            new_graph.get_loop_count() if new_graph is not self._graph else self._loop_count,
        )
        return FeynmanIntegral(
            graph=new_graph,
            dimension=overrides.pop("dimension", self._dimension),
            propagator_exponents=overrides.pop(
                "propagator_exponents", dict(self._propagator_exponents)
            ),
            loop_count=loop_count,
            momentum_products=overrides.pop("momentum_products", dict(self._momentum_products)),
            use_mandelstam=overrides.pop("use_mandelstam", self._use_mandelstam),
            kinematic_constraints=overrides.pop(
                "kinematic_constraints", list(self._kinematic_constraints)
            ),
            database=overrides.pop("database", self._database),
            **overrides,
        )

    def canonicalise(self, matrix: sp.Matrix) -> PairingMatrixResult:
        """
        Lex-maximal canonical form of a pairing matrix under independent
        row/column permutations (Grinis–Kasprzyk).

        Exposed as a method rather than a property because the relevant
        pairing matrix depends on which downstream object the user wishes
        to canonicalise (e.g. the GKZ A-matrix, or a vertex–facet pairing
        matrix once the face lattice is wired up in Part B).
        """
        return maximal_pairing_matrix(matrix)

    def tikz(self) -> str:
        """TikZ source for the Feynman graph."""
        from .visualisation.tikz import graph_to_tikz

        return graph_to_tikz(self._graph)

    def visualise_polytope(self, **kwargs: Any) -> str:
        """TikZ source for the Newton polytope of G."""
        from .visualisation.polytope import visualise_newton_polytope

        return visualise_newton_polytope(self.newton_polytope.support, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Self-contained LaTeX analysis document for the integral."""
        from .io.latex import _create_analysis_document

        s = self.symanzik
        return _create_analysis_document(
            graph=self._graph,
            u_polynomial=s.u,
            f_polynomial=s.f,
            gkz_system=self.gkz,
            parametrisation_results={
                "Schwinger": self.schwinger,
                "Feynman": self.feynman,
                "Lee-Pomeransky": self.lee_pomeransky,
            },
            toric_generators=self.toric_ideal.generators,
            **kwargs,
        )

    def to_text(self, **kwargs: Any) -> str:
        """Plain-text analysis report for the integral."""
        from .io.text import _create_analysis_report

        s = self.symanzik
        return _create_analysis_report(
            graph=self._graph,
            u_polynomial=s.u,
            f_polynomial=s.f,
            gkz_system=self.gkz,
            parametrisation_results={
                "Schwinger": self.schwinger,
                "Feynman": self.feynman,
                "Lee-Pomeransky": self.lee_pomeransky,
            },
            toric_generators=self.toric_ideal.generators,
            **kwargs,
        )

    # ──────── Comparison ────────

    def is_unimodular_equivalent_to(self, other: FeynmanIntegral) -> PolytopeEquivalence:
        """
        Test whether two integrals' Newton polytopes are unimodularly
        equivalent (Liu–Cai, arXiv:2506.23846).

        Compares the lattice point configurations spanned by the monomial
        support of each integral's Lee–Pomeransky G polynomial. On success
        the result carries an integer witness map ``U ∈ GL_n(ℤ)`` and a
        vertex correspondence.
        """
        from .normal_forms.affine_equivalence import is_unimodular_equivalent

        return is_unimodular_equivalent(
            self.newton_polytope.points,
            other.newton_polytope.points,
        )

    def is_affinely_equivalent_to(self, other: FeynmanIntegral) -> PolytopeEquivalence:
        """
        Test whether two integrals' Newton polytopes are equivalent under an
        affine map over the rationals (broader than unimodular). The search
        is exact but slow on larger polytopes.
        """
        from .normal_forms.affine_equivalence import is_affinely_equivalent

        return is_affinely_equivalent(
            self.newton_polytope.points,
            other.newton_polytope.points,
        )

    def __repr__(self) -> str:
        return (
            "FeynmanIntegral("
            f"graph=Graph(internal_vertices={self._graph.internal_vertices}, "
            f"external_legs={self._graph.external_legs}, "
            f"edges={len(self._graph.edges)}), "
            f"dimension={self._dimension}, "
            f"loop_count={self._loop_count})"
        )
