"""
Unified Feynman integral object.

This module exposes :class:`FeynmanIntegral`, a single immutable facade that
gathers every representation of a Feynman integral, graph topology, Symanzik
and Lee-Pomeransky polynomials, parametric representations, the GKZ
hypergeometric system, the Newton polytope, and the toric ideal, as
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

import dataclasses
from collections.abc import Collection, Mapping
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Any

import sympy as sp

if TYPE_CHECKING:
    from .database import FeynkitDatabase
    from .kinematics.classes import ExternalAxis, InternalAxis, KinematicClass
    from .landau import LandauAnalysis
    from .point_count import TorusCount

from . import _exact
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
from .systems.cayley import CayleyGKZSystem, create_cayley_system
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


# ------------------------------------------------------------------------------
# The facade.
# ------------------------------------------------------------------------------


_NU_PREFIX = "nu"
_DEFAULT_DIMENSION_NAME = "D"


class FeynmanIntegral:
    """
    Unified, immutable representation of a Feynman integral.

    The constructor only validates inputs and stores symbols; every
    representation is computed lazily on first access through a
    ``cached_property``. Two attributes accessed in succession therefore
    share intermediate results, for instance, ``integral.toric_ideal``
    re-uses the GKZ A-matrix already computed by ``integral.gkz``.

    Parameters
    ----------
    graph
        The :class:`Graph` whose integral is being studied.
    dimension
        The spacetime dimension. Defaults to a positive symbol ``D``.
    propagator_exponents
        Mapping ``edge_idx -> nu_i``. Defaults to symbols ``nu_<i>`` for each
        internal edge.
    loop_count
        Number of independent loops. Defaults to ``graph.get_loop_count()``.
    momentum_products
        Mapping ``(i, j) -> p_i * p_j`` for external legs. Defaults to symbolic
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

    # -------- Read-only views of the input data --------

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

    # -------- Constructors from CNickel --------

    @classmethod
    def from_cnickel(
        cls, cnickel: str, *, kinematics: KinematicClass | None = None, **kwargs: Any
    ) -> FeynmanIntegral:
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
        kinematics
            A kinematic class to impose with :meth:`with_kinematics`, one of
            ``generic``, ``massless_off_shell``, ``massless_on_shell`` and
            ``equal_masses``; by default the kinematics of the string.
        **kwargs
            Forwarded to :class:`FeynmanIntegral` (``dimension``,
            ``propagator_exponents``, ``momentum_products``, ``database``, ...).

        Raises
        ------
        ValidationError
            As :meth:`with_kinematics` raises, when ``kinematics`` is given.

        Examples
        --------
        >>> fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        >>> fi.cnickel
        '12e|2e|e|:zzz'
        >>> fi.toric_ideal.generators
        >>> FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz", kinematics="massless_on_shell")
        """
        integral = cls(Graph.from_cnickel(cnickel), **kwargs)
        return integral if kinematics is None else integral.with_kinematics(kinematics)

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

    # -------- Topology --------

    @property
    def nickel_index(self) -> str:
        """Canonical Nickel topology string (delegates to :meth:`Graph.nickel_index`)."""
        return self._graph.nickel_index()

    @property
    def cnickel(self) -> str:
        """Colored Nickel index: topology + mass colouring (``"topology:colours"``)."""
        return self._graph.cnickel()

    # -------- Kinematic class --------

    @cached_property
    def kinematic_axes(self) -> tuple[InternalAxis, ExternalAxis]:
        """
        The internal and external kinematic axes, derived from the integral as
        it stands.

        The internal axis is ``zero``, ``equal``, ``generic`` or ``other``, read
        from the masses m_e; the external axis is ``off_shell``, ``on_shell``,
        ``equal`` or ``other``, read from the momentum products and any
        kinematic constraints. See :func:`feynkit.kinematics.kinematic_axes`
        for the rules. It never raises.
        """
        from .kinematics.classes import kinematic_axes

        return kinematic_axes(self)

    @cached_property
    def kinematic_class(self) -> KinematicClass:
        """
        The kinematic class of the axes: ``generic``, ``massless_off_shell``,
        ``massless_on_shell``, ``equal_masses`` or ``other``.

        They are the kinematics of the entries generic_generic, zero_generic,
        zero_zero and equal_generic of the principal Landau determinant
        database of Fevola, Mizera and Telen (arXiv:2311.16219). Every other
        pair of axes, such as massive propagators with on-shell legs, is
        ``other``.
        """
        from .kinematics.classes import CLASS_OF_AXES

        return CLASS_OF_AXES.get(self.kinematic_axes, "other")

    def with_kinematics(self, kinematic_class: KinematicClass) -> FeynmanIntegral:
        """
        The integral with a kinematic class imposed by substitution.

        Returns ``self`` when the integral already has the class. Otherwise
        ``massless_on_shell`` sets each p_i^2 to 0 in every momentum product,
        which needs massless propagators, two or more legs and p_i^2 that are
        symbols, as with ``use_mandelstam=True``; ``equal_masses`` gives every
        propagator the mass m_a of the CNickel code a, which needs massive
        propagators. ``generic`` and ``massless_off_shell`` make no
        substitution. A class never changes which propagators are massless.

        Raises
        ------
        ValidationError
            If the class is not one of the four above, if the substitution
            cannot be made, or if the result is not of the class.
        """
        from .kinematics.classes import impose_kinematics

        return impose_kinematics(self, kinematic_class)

    # -------- Internal: shared parametrisation factory --------

    @cached_property
    def _all_parametrisations(self) -> AllParametrisations:
        return _create_parametrisations(
            graph=self._graph,
            dimension=self._dimension,
            loop_count=self._loop_count,
            propagator_exponents=self._propagator_exponents,
            momentum_products=self._momentum_products,
        )

    # -------- Polynomials --------

    @cached_property
    def symanzik(self) -> SymanzikPolynomials:
        """All graph polynomials together: U, F (Schwinger and LP forms) and G."""
        ap = self._all_parametrisations
        # Reuse the Lee-Pomeransky helper that builds the canonical a -> u
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

    # -------- Parametric representations --------

    @cached_property
    def schwinger(self) -> ParametrisationResult:
        return self._all_parametrisations.schwinger.compute()

    @cached_property
    def feynman(self) -> ParametrisationResult:
        return self._all_parametrisations.feynman.compute()

    @cached_property
    def lee_pomeransky(self) -> ParametrisationResult:
        return self._all_parametrisations.lee_pomeransky.compute()

    # -------- Algebraic-geometry representations --------

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
    def is_scaleless(self) -> bool:
        """
        Whether the integral is scaleless by Lee's criterion of a zero sector.

        Lee (R. N. Lee, arXiv:1310.1145, section 3) calls a sector zero when
        sum_e k_e u_e dG/du_e = G has a solution k independent of u. That is
        k . alpha = 1 for every exponent alpha of G, so the criterion holds
        exactly when some equation h_0 + h . x = 0 of the affine hull of the
        Newton polytope has h_0 != 0, that is when the origin does not lie in
        that affine hull. Substituting lambda^(h_e) u_e for u_e then multiplies
        the Lee-Pomeransky integral by lambda^(h_0 D/2 + h . nu), whose
        exponent involves D, and dimensional regularisation sets the integral
        to zero.

        A scaleless integral has a Newton polytope that is not
        full-dimensional; the converse fails. For 1ee|1|:zn, a massive tadpole
        joined to its external vertex by a massless line that carries no
        momentum, every equation has h_0 = 0: the integral converges for no D
        and nu, but it is not scaleless by Lee's criterion, and dimensional
        regularisation does not regulate it. The flag depends on the monomials
        of G only, not on D or nu.
        """
        points = [tuple(int(x) for x in p) for p in self.newton_polytope.points]
        origin = (0,) * len(self.newton_polytope.parameters)
        return _exact.affine_rank([*points, origin]) > _exact.affine_rank(points)

    @cached_property
    def schwinger_gkz(self) -> CayleyGKZSystem:
        """GKZ system of the Schwinger representation (two-block Cayley form).

        Built from the dehomogenised Symanzik polynomials U~ and F~ with
        alpha_N set to one (arXiv:2609.16107, section 3; Klausen 2023,
        section 3.4). See :mod:`feynkit.systems.cayley` for conventions.
        """
        schwinger = self._all_parametrisations.schwinger
        (u_tilde, f_tilde), u_vars = schwinger.dehomogenised_symanzik_polynomials()
        edges = self._graph.get_internal_edges()
        return create_cayley_system(
            u_tilde,
            f_tilde,
            u_vars,
            dimension=self._dimension,
            propagator_exponents=[self._propagator_exponents[e.idx] for e in edges],
            loop_count=self._loop_count,
            prefactor=self._all_parametrisations.schwinger.prefactor,
        )

    @cached_property
    def polytope_automorphisms(self) -> PolytopeAutomorphisms:
        """
        Automorphism group of the Newton polytope P of G.

        Aut(P) is the group of P as a lattice polytope in the affine lattice
        aff(P) cap Z^n: the affine bijections of the affine hull of P that map
        its integer points onto themselves and P onto itself. When P is
        full-dimensional this is {(U, t) : U in GL_n(Z), |det U| = 1,
        t in Z^n, {Uv + t : v in V} = V} for the vertex set V of P. Below full
        dimension each element is returned as one such (U, t) that extends it,
        and U fixes a complement of the direction space of the affine hull,
        the span of the differences of the vertices. The identity is always
        included; for generic polytopes it is the only element.

        For highly symmetric diagrams (the massless triangle has 48
        automorphisms, the massless box 120, the massless banana with n
        propagators (n + 1)!) the group is non-trivial and reflects the
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

        Returns every :class:`SymmetryPair` (M, t, P) satisfying T*A = A*Pi_P,
        where T = [[1, 0^T], [t, M]] is an invertible integer matrix and P is
        a column permutation.  Each pair gives the Feynman-integral identity:

            I_A(beta, z_P) = I_A(T beta, z)

        where z_P = (z_{P(0)}, ..., z_{P(N-1)}) and I_A is the Lee-Pomeransky
        integral without its prefactor (de la Cruz 2024).

        Every pair is unimodular (``pair.is_unimodular``): P has finite
        order k, so M^k = I and det M = +/-1.  On the affine hull of the
        points the pairs are the elements of :attr:`polytope_automorphisms`
        that permute every column of A, not only the vertices, and so all of
        them when every column is a vertex.  Below full dimension each is one
        extension of such an element to Z^n, and the identities hold only
        trivially: T beta depends on the extension, and for generic D and nu
        the GKZ system has no non-zero solutions.
        Maps with |det M| > 1 relate two different configurations; see
        :func:`feynkit.a_configuration.finite_index_map`.

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
        1-indexed label for each original vertex.  The identity ``[1, 2, ..., V]``
        is always included.

        These are the graph-theoretic automorphisms; the polytope automorphism
        group can be larger (e.g. the 3-propagator banana has vertex
        automorphism Z/2 but polytope automorphism S_3).
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

    # -------- Actions --------

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
        row/column permutations (Grinis-Kasprzyk).

        Exposed as a method rather than a property because the relevant
        pairing matrix depends on which downstream object the user wishes
        to canonicalise (e.g. the GKZ A-matrix, or a vertex-facet pairing
        matrix built from the facets of polytope_data).
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

    def to_latex(
        self,
        sections: Collection[str] | None = None,
        *,
        title: str | None = None,
        max_face_points: int = 12,
    ) -> str:
        """
        The analysis report of the integral as a LaTeX document.

        Parameters
        ----------
        sections
            Names of the report sections to include, from
            :data:`feynkit.io.report.SECTION_NAMES`; by default
            :data:`~feynkit.io.report.DEFAULT_SECTIONS`, every section but
            ``torus``. The graph, the conventions and the Symanzik polynomials
            are always included.
        title
            Document title; by default "Feynman integral" followed by the
            CNickel string.
        max_face_points
            Faces of the Newton polytope with more monomials than this are
            left out of the Landau analysis and listed as skipped.

        Returns
        -------
        str
            A complete ``article`` document, compilable with pdflatex.

        Raises
        ------
        ValidationError
            If ``sections`` names something that is not a section, or, when it
            holds ``torus``, as :meth:`torus_count` raises: for instance when
            the integral has kinematic constraints, or when counting needs more
            than the default budget of 2 * 10^9 evaluations of G.
        """
        from .io.report import AnalysisReport
        from .io.report_latex import render_latex

        report = AnalysisReport.from_integral(self, sections, max_face_points=max_face_points)
        return render_latex(report, title=title)

    def to_text(
        self,
        sections: Collection[str] | None = None,
        *,
        title: str | None = None,
        max_face_points: int = 12,
    ) -> str:
        """
        The analysis report of the integral as plain text.

        It states the facts of :meth:`to_latex` in the same order, in ASCII.
        The parameters are those of :meth:`to_latex`.

        Raises
        ------
        ValidationError
            If ``sections`` names something that is not a section, or, when it
            holds ``torus``, as :meth:`torus_count` raises: for instance when
            the integral has kinematic constraints, or when counting needs more
            than the default budget of 2 * 10^9 evaluations of G.
        """
        from .io.report import AnalysisReport
        from .io.report_text import render_text

        report = AnalysisReport.from_integral(self, sections, max_face_points=max_face_points)
        return render_text(report, title=title)

    def torus_count(
        self,
        *,
        seed: int = 0,
        point: Mapping[sp.Expr, int | Fraction] | None = None,
        allow_singular: bool = False,
        on_shell: Mapping[sp.Symbol, sp.Expr] | None = None,
        landau: LandauAnalysis | None = None,
        verification: int = 4,
        max_face_points: int = 12,
        max_evaluations: int = 2 * 10**9,
        backend: str = "numpy",
    ) -> TorusCount:
        """
        Finite-field point counts of G = 0 in the torus, and the candidates they give.

        Counts the points of {G = 0} in (F_p^*)^N at one rational kinematic
        point for finitely many primes, with the energy scale set to 1, and
        fits a polynomial P(q). When the fit passes its tests, among them a
        check at further primes, chi(X) = -P(1) for the complement X of
        {G = 0} in the torus is a candidate Euler characteristic, and
        C = (-1)^N chi(X) a candidate number of master integrals (Bitoun,
        Bogner, Klausen and Panzer 2019, Corollary 37). A candidate is not a
        proof: Katz's theorem needs the count to be polynomial for every finite
        field of all but finitely many characteristics. See
        :func:`feynkit.point_count.count_torus_points`.

        Parameters
        ----------
        seed, point, allow_singular, verification, max_evaluations, backend
            As for :func:`~feynkit.point_count.count_torus_points`.
        on_shell
            Substitutions made in the momentum products first. With Mandelstam
            invariants (``use_mandelstam=True``) ``{p1^2: 0}`` makes a leg
            massless; with dot products a relation such as ``{p1p2: -p1p3}`` is
            needed. Each key must be a symbol of the momentum products, with
            its assumptions, as ``Symbol("p1^2", real=True)``. Each value is an
            exact expression, an int, a Fraction, a string or a SymPy
            expression without floats, and contains no key. A symbol of a
            value that shares its name with a momentum-product symbol, a mass
            or the energy scale must be that symbol, so a string such as
            ``"s12"``, which sympify parses without assumptions, is refused.
            String values are parsed by SymPy, so ``"p1^2"`` means ``p1**2``;
            pass SymPy symbols for invariants.
        landau
            The Landau analysis of this integral, with ``on_shell`` applied, if
            already computed; it keeps the energy scale symbolic.
        max_face_points
            Passed to :func:`~feynkit.landau.landau_analysis` when ``landau`` is
            not given.

        Raises
        ------
        ValidationError
            If the integral has kinematic constraints, which are not applied;
            if ``on_shell`` names a symbol the momentum products do not
            contain, gives a value that is not an exact expression, gives a
            value containing one of its keys or a symbol that only shares a
            name with one of the integral's; or as
            :func:`~feynkit.point_count.count_torus_points` raises.
        RuntimeError
            If ``backend`` is "flint" and python-flint is not installed.
        """
        from .landau import landau_analysis
        from .point_count import count_torus_points

        if self._kinematic_constraints:
            raise ValidationError(
                "torus_count does not apply kinematic_constraints; substitute them with on_shell"
            )
        substitutions: dict[sp.Symbol, sp.Expr] = {}
        for key, raw in (on_shell or {}).items():
            value: object
            try:
                value = sp.sympify(raw)
            except sp.SympifyError:
                value = None
            # A float would reach the Landau analysis, which cannot parse it back from Singular.
            if not isinstance(value, sp.Expr) or value.has(sp.Float):
                raise ValidationError(
                    f"on_shell gives {key} the value {raw!r}, which is not an exact expression; "
                    "give an int, a Fraction, a string or a SymPy expression without floats"
                )
            substitutions[key] = value
        integral = self
        if substitutions:
            products = self.momentum_products
            known: set[sp.Basic] = set().union(
                *(sp.sympify(value).free_symbols for value in products.values())
            )
            unknown = sorted(str(key) for key in substitutions if key not in known)
            if unknown:
                # A key can share its name with a symbol of the products but not its assumptions.
                clashes = sorted(sp.srepr(x) for x in known if str(x) in unknown)
                hint = f"; keys carry their assumptions, as {', '.join(clashes)}" if clashes else ""
                raise ValidationError(
                    f"on_shell names {', '.join(unknown)}, which the momentum products do not "
                    f"contain; they contain {', '.join(sorted(map(str, known)))}{hint}"
                )
            # Substituted one after the other, a value containing a key would depend on the order.
            chained = sorted(
                str(key) for key in substitutions if any(v.has(key) for v in substitutions.values())
            )
            if chained:
                raise ValidationError(
                    f"on_shell values contain its keys {', '.join(chained)}; substitute them "
                    "in the values first"
                )
            # A value can name a symbol of the integral without being it, as sympify makes of
            # "s12"; that symbol would be drawn separately.
            masses = (edge.get_mass() for edge in self._graph.get_internal_edges())
            own = known.union(
                *(sp.sympify(mass).free_symbols for mass in masses), {self._graph.energy_scale}
            )
            names = {str(x) for x in own}
            strays = sorted(
                {str(x) for v in substitutions.values() for x in v.free_symbols if x not in own}
                & names
            )
            if strays:
                clashes = sorted(sp.srepr(x) for x in own if str(x) in strays)
                raise ValidationError(
                    f"on_shell values contain symbols {', '.join(strays)} that only share a name "
                    f"with the integral's; symbols carry their assumptions, as {', '.join(clashes)}"
                )
            integral = self.with_(
                momentum_products={
                    key: sp.expand(sp.sympify(value).subs(substitutions))
                    for key, value in products.items()
                }
            )
        if landau is None:
            landau = landau_analysis(integral, max_face_points=max_face_points)
        symanzik = integral.symanzik
        count = count_torus_points(
            symanzik.g,
            symanzik.lp_parameters,
            scale=integral.graph.energy_scale,
            seed=seed,
            point=point,
            allow_singular=allow_singular,
            landau=landau,
            verification=verification,
            max_evaluations=max_evaluations,
            backend=backend,
        )
        return dataclasses.replace(
            count, on_shell=tuple(sorted(substitutions.items(), key=lambda item: item[0].name))
        )

    # -------- Comparison --------

    def is_unimodular_equivalent_to(self, other: FeynmanIntegral) -> PolytopeEquivalence:
        """
        Test whether two integrals' Newton polytopes are unimodularly
        equivalent (Liu-Cai, arXiv:2506.23846).

        Compares the lattice point configurations spanned by the monomial
        support of each integral's Lee-Pomeransky G polynomial. On success
        the result carries an integer witness map ``U in GL_n(Z)`` and a
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
