"""
Frozen description of everything feynkit computes for one Feynman integral.

:meth:`AnalysisReport.from_integral` gathers the graph data, the Symanzik
polynomials, the parametric representations, the Newton polytope, the GKZ
system, the symmetries, the Landau surfaces and the Schwinger-representation
system into one immutable value, so that a renderer can walk the result
without reaching back into the integral. The module carries data only: it
makes no formatting decisions and holds no LaTeX beyond the TikZ figures it
stores.

Conventions
-----------
The Newton polytope coordinates are the Lee-Pomeransky parameters
u_1, ..., u_N in internal-edge order, the order of
``symanzik.lp_parameters``, so a facet inequality m . x <= b of the Newton
polytope of G contributes the convergence condition
b D/2 - sum_e m_e nu_e > 0, with nu_e the exponent of edge e (Klausen 2023,
Thm FIconvergence). The parameter vector of the Lee-Pomeransky GKZ system is
beta = (-D/2, -nu_1, ..., -nu_N).

U and F are stored in the Schwinger parameters a_e, the form in which the
graph-theoretic definitions are usually quoted, while G is stored in the
Lee-Pomeransky parameters u_e, the variables of the Newton polytope and of
the GKZ system. The two parameter sets differ only in the name of the
symbols.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from .. import _exact
from ..a_configuration import SymmetryPair
from ..core.exceptions import ValidationError
from ..kinematics.mandelstam import KinematicInvariants, standard_invariants
from ..landau import (
    LandauAnalysis,
    landau_analysis,
    one_loop_bridge_poles,
    one_loop_landau_surfaces_by_type,
)
from ..lattice_invariants import LatticeInvariants
from ..normal_forms.polytope_automorphisms import coefficient_preserving_indices
from ..parametrisations.base import ParametrisationResult
from ..point_count import TorusCount
from ..polytope import PolytopeData, polytope_data
from ..systems.cayley import CayleyGKZSystem
from ..systems.complete import GKZSystem
from ..systems.monomial import extract_monomial_support
from .latex import factor_energy_scale

if TYPE_CHECKING:
    from ..integral import FeynmanIntegral

__all__ = [
    "DEFAULT_SECTIONS",
    "LATTICE_BUDGET",
    "NORMALIZ_TIMEOUT",
    "SECTION_NAMES",
    "AnalysisReport",
    "Conventions",
    "GKZ",
    "Identity",
    "Landau",
    "Polynomials",
    "Polytope",
    "Representations",
    "Schwinger",
    "Symmetries",
    "ZEntry",
]


SECTION_NAMES = (
    "identity",
    "conventions",
    "polynomials",
    "representations",
    "polytope",
    "torus",
    "gkz",
    "symmetries",
    "landau",
    "schwinger",
)

# The sections built when none are named: all but the point counts, which take seconds for
# five propagators; seven exceed the default budget.
DEFAULT_SECTIONS = tuple(name for name in SECTION_NAMES if name != "torus")

# The budget and Normaliz timeout of the lattice invariants in the report and in fk analyse:
# about five seconds of pure-Python work, beyond which Normaliz takes over or the fields that
# need more are not computed. The massless three-loop box would need about 10^10 steps.
LATTICE_BUDGET = 2 * 10**7
NORMALIZ_TIMEOUT = 60.0


# --- section data types ------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    """What the graph is.

    Attributes
    ----------
    cnickel, nickel
        Coloured and bare Nickel indices of the topology.
    loop_count, propagators, external_legs
        L, the number of internal edges and the number of external legs.
    edge_indices, edge_endpoints
        The index e of each internal edge, which names its parameters a_e and
        u_e and its exponent nu_e, and the two vertices it joins, in
        internal-edge order. The indices need not run from 1 to N.
    edge_masses, edge_exponents
        The mass m_e and the exponent nu_e of each internal edge, in
        internal-edge order; a massless edge has mass 0.
    graph_tikz
        TikZ source for the graph figure.
    """

    cnickel: str
    nickel: str
    loop_count: int
    propagators: int
    external_legs: int
    edge_indices: tuple[int, ...]
    edge_endpoints: tuple[tuple[int, int], ...]
    edge_masses: tuple[sp.Expr, ...]
    edge_exponents: tuple[sp.Expr, ...]
    graph_tikz: str


@dataclass(frozen=True)
class Conventions:
    """The dimension, the energy scale and the kinematics.

    Attributes
    ----------
    dimension
        The spacetime dimension D.
    energy_scale
        The scale mu that keeps the Symanzik polynomials dimensionless.
    invariants
        The standard planar invariants, or None when the integral has fewer
        than two external legs or does not use Mandelstam variables.
    momentum_products
        The products p_i . p_j as ((i, j), value) pairs, sorted by key.
    kinematic_axes, kinematic_class
        The internal and external kinematic axes and the kinematic class, as
        FeynmanIntegral.kinematic_axes and kinematic_class give them.
    """

    dimension: sp.Expr
    energy_scale: sp.Symbol
    invariants: KinematicInvariants | None
    momentum_products: tuple[tuple[tuple[int, int], sp.Expr], ...]
    kinematic_axes: tuple[str, str]
    kinematic_class: str


@dataclass(frozen=True)
class ZEntry:
    """One monomial of G and the GKZ variable standing for its coefficient.

    Attributes
    ----------
    index
        The index j of z_j, counting from one, in the column order of the
        GKZ A-matrix.
    exponent
        The exponent vector in the Lee-Pomeransky parameters.
    monomial
        The monomial itself, the product of u_e**a_e.
    coefficient
        The physical value of z_j, the coefficient the monomial carries in
        G.
    """

    index: int
    exponent: tuple[int, ...]
    monomial: sp.Expr
    coefficient: sp.Expr


@dataclass(frozen=True)
class Polynomials:
    """The Symanzik polynomials and the counts derived from them.

    Attributes
    ----------
    u, g
        U in the Schwinger parameters and G = U + F in the Lee-Pomeransky
        parameters.
    f_numerator, f_scale_power
        F written as ``f_numerator / mu**f_scale_power`` with the numerator
        free of mu; the power is 2 for an integral with kinematics and 0
        when mu does not occur.
    degree_u, degree_f
        The total degrees L and L + 1.
    monomials_f, monomials_g
        The number of monomials of F and of G.
    z_table
        One entry per monomial of G, in GKZ column order.
    codimension
        ``monomials_g - rank A``, the codimension of the A-configuration
        (Klausen 2023, Table A.4); it is ``monomials_g - N - 1`` for N internal
        edges when the Newton polytope is full-dimensional.
    independent_invariants
        The number of distinct kinematic symbols, masses and invariants, in
        the coefficients of G; mu is not counted.
    scaleless
        Whether the integral is scaleless by Lee's criterion, as
        FeynmanIntegral.is_scaleless says.
    """

    u: sp.Expr
    f_numerator: sp.Expr
    f_scale_power: int
    g: sp.Expr
    degree_u: int
    degree_f: int
    monomials_f: int
    monomials_g: int
    z_table: tuple[ZEntry, ...]
    codimension: int
    independent_invariants: int
    scaleless: bool


@dataclass(frozen=True)
class Representations:
    """The parametric representations and where the last one converges.

    Attributes
    ----------
    schwinger, feynman, lee_pomeransky
        The three parametric representations.
    convergence
        Expressions that must all be positive for the Lee-Pomeransky
        integral to converge, one per facet of the Newton polytope; None
        when the polytope is not full-dimensional, in which case the
        integral converges nowhere.
    """

    schwinger: ParametrisationResult
    feynman: ParametrisationResult
    lee_pomeransky: ParametrisationResult
    convergence: tuple[sp.Expr, ...] | None


@dataclass(frozen=True)
class Polytope:
    """The Newton polytope of G, its figure and its lattice invariants.

    Attributes
    ----------
    data
        Vertices, faces, facet inequalities and normalised volume.
    figure
        TikZ source, or None when the polytope has too many vertices to
        draw legibly.
    invariants
        The lattice invariants of FeynmanIntegral.lattice_invariants, in the
        lattice the points generate, within LATTICE_BUDGET and
        NORMALIZ_TIMEOUT; None only for a Polytope built by hand.

    Point indices in ``data`` (``vertex_indices``, ``faces``, each facet's
    ``point_indices``) follow the order of ``newton_polytope.points``, which
    is not the z-index order of ``gkz.support`` that :class:`ZEntry` uses: the
    two enumerate the same monomials independently and need not agree.
    """

    data: PolytopeData
    figure: str | None
    invariants: LatticeInvariants | None = None


@dataclass(frozen=True)
class GKZ:
    """The Lee-Pomeransky GKZ system.

    Attributes
    ----------
    a_matrix, beta, z_variables
        The A-matrix, the parameter vector and the coefficient variables.
    euler_rows
        One (row of A, beta_r) pair per row, so that a renderer can write
        the Euler operator sum_j A_rj theta_j - beta_r.
    toric_generators
        Generators of the toric ideal of A.
    """

    a_matrix: sp.Matrix
    beta: tuple[sp.Expr, ...]
    z_variables: tuple[sp.Symbol, ...]
    euler_rows: tuple[tuple[tuple[int, ...], sp.Expr], ...]
    toric_generators: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class Symmetries:
    """Symmetries of the polytope, of the graph and of the A-configuration.

    Attributes
    ----------
    automorphism_order, vertex_orbits
        The order of Aut(P) and its orbits on the vertices.
    graph_automorphisms
        Vertex permutations of the graph preserving topology and masses.
    coefficient_preserving
        How many polytope automorphisms also preserve the coefficients of
        G, the subgroup relevant for functional equations.
    symmetry_pairs
        Every integer affine self-map of the A-configuration.
    full_dimensional
        Whether P is full-dimensional. Below full dimension Aut(P) is the
        group of P in its affine hull, and the identities the symmetry pairs
        give hold only trivially.

    ``vertex_orbits`` indexes ``PolytopeData.vertices``, the list the Newton
    polytope section numbers v_1, v_2, ..., not the z-index order of
    ``gkz.support``.
    """

    automorphism_order: int
    vertex_orbits: tuple[tuple[int, ...], ...]
    graph_automorphisms: tuple[tuple[int, ...], ...]
    coefficient_preserving: int
    symmetry_pairs: tuple[SymmetryPair, ...]
    full_dimensional: bool


@dataclass(frozen=True)
class Landau:
    """The reduced principal A-determinant and the surfaces it factors into.

    Attributes
    ----------
    analysis
        The face-by-face computation.
    by_dimension
        The non-trivial face discriminants grouped by face dimension,
        ascending, each group free of repeats.
    first_type, second_type
        The Cayley and Gram factors of the one-loop closed form; both empty
        for more than one loop, where no closed form is available. For a
        graph with bridges they are those of its cycle.
    skipped
        One (dimension, number of points, whether it is P itself) triple per
        face left out as too large to eliminate, in the order of
        ``analysis.skipped_faces``.
    bridge_poles
        The factors of the poles m_b^2 = q_b^2 of the bridges of a one-loop
        graph, from :func:`~feynkit.landau.one_loop_bridge_poles`; empty
        without bridges or for more than one loop.
    """

    analysis: LandauAnalysis
    by_dimension: tuple[tuple[int, tuple[sp.Expr, ...]], ...]
    first_type: tuple[sp.Expr, ...]
    second_type: tuple[sp.Expr, ...]
    skipped: tuple[tuple[int, int, bool], ...] = ()
    bridge_poles: tuple[sp.Expr, ...] = ()


@dataclass(frozen=True)
class Schwinger:
    """The GKZ system of the Schwinger representation.

    Attributes
    ----------
    system
        The two-block Cayley system of U~ and F~.
    f_block
        Its face subsystem on the F~ block.
    lp_to_cayley
        The unimodular T taking the Lee-Pomeransky A-matrix to the Cayley
        one up to column order, and beta_LP to beta_Cayley.
    columns_match
        Whether the columns of ``T A_LP`` are a permutation of those of the
        Cayley matrix.
    """

    system: CayleyGKZSystem
    f_block: GKZSystem
    lp_to_cayley: sp.Matrix
    columns_match: bool


# --- section builders --------------------------------------------------------


def _identity(fi: FeynmanIntegral) -> Identity:
    edges = fi.graph.get_internal_edges()
    exponents = fi.propagator_exponents
    return Identity(
        cnickel=fi.cnickel,
        nickel=fi.nickel_index,
        loop_count=fi.loop_count,
        propagators=len(edges),
        external_legs=fi.graph.external_legs,
        edge_indices=tuple(e.idx for e in edges),
        edge_endpoints=tuple((e.v1, e.v2) for e in edges),
        edge_masses=tuple(sp.sympify(e.get_mass()) for e in edges),
        edge_exponents=tuple(exponents[e.idx] for e in edges),
        graph_tikz=fi.tikz(),
    )


def _conventions(fi: FeynmanIntegral) -> Conventions:
    n_legs = fi.graph.external_legs
    invariants = standard_invariants(n_legs) if fi.use_mandelstam and n_legs >= 2 else None
    return Conventions(
        dimension=fi.dimension,
        energy_scale=fi.graph.energy_scale,
        invariants=invariants,
        momentum_products=tuple(sorted(fi.momentum_products.items())),
        kinematic_axes=fi.kinematic_axes,
        kinematic_class=fi.kinematic_class,
    )


def _polynomials(fi: FeynmanIntegral) -> Polynomials:
    symanzik = fi.symanzik
    parameters = list(symanzik.lp_parameters)
    scale = fi.graph.energy_scale
    numerator, power = factor_energy_scale(symanzik.f, scale)

    physical = {tuple(alpha): coefficient for alpha, coefficient in fi.newton_polytope.support}
    z_table = tuple(
        ZEntry(
            index=j,
            exponent=tuple(int(k) for k in alpha),
            monomial=sp.Mul(*[v ** int(k) for v, k in zip(parameters, alpha, strict=True)]),
            coefficient=physical.get(tuple(alpha), sp.Integer(0)),
        )
        for j, (alpha, _) in enumerate(fi.gkz.support, start=1)
    )
    invariants = {s for entry in z_table for s in entry.coefficient.free_symbols} - {scale}

    f_expanded = sp.expand(symanzik.f)
    monomials_f = (
        0
        if f_expanded == 0
        else len(extract_monomial_support(f_expanded, list(symanzik.schwinger_parameters)))
    )
    return Polynomials(
        u=symanzik.u,
        f_numerator=numerator,
        f_scale_power=power,
        g=symanzik.g,
        degree_u=int(sp.Poly(symanzik.u_lp, *parameters).total_degree()),
        degree_f=int(sp.Poly(symanzik.f_lp, *parameters).total_degree()),
        monomials_f=monomials_f,
        monomials_g=len(z_table),
        z_table=z_table,
        codimension=len(z_table) - int(fi.gkz.a_matrix.rank()),
        independent_invariants=len(invariants),
        scaleless=fi.is_scaleless,
    )


def _representations(fi: FeynmanIntegral, data: PolytopeData) -> Representations:
    convergence: tuple[sp.Expr, ...] | None = None
    if data.is_full_dimensional:
        edges = fi.graph.get_internal_edges()
        exponents = [fi.propagator_exponents[e.idx] for e in edges]
        half_dimension = fi.dimension / 2
        inequalities = []
        for facet in data.facets:
            if len(facet.normal) != len(exponents):
                raise ValidationError(
                    f"Facet normal of length {len(facet.normal)} does not match "
                    f"{len(exponents)} propagator exponents"
                )
            inequalities.append(
                facet.offset * half_dimension
                - sum(m * nu for m, nu in zip(facet.normal, exponents, strict=True))
            )
        convergence = tuple(inequalities)
    return Representations(
        schwinger=fi.schwinger,
        feynman=fi.feynman,
        lee_pomeransky=fi.lee_pomeransky,
        convergence=convergence,
    )


def _polytope(fi: FeynmanIntegral, data: PolytopeData, figure_max_vertices: int) -> Polytope:
    draw = len(data.vertex_indices) <= figure_max_vertices
    return Polytope(
        data=data,
        figure=fi.visualise_polytope() if draw else None,
        invariants=fi.lattice_invariants(budget=LATTICE_BUDGET, timeout=NORMALIZ_TIMEOUT),
    )


def _gkz(fi: FeynmanIntegral) -> GKZ:
    system = fi.gkz
    a_matrix = system.a_matrix
    beta = tuple(system.beta_parameters)
    euler_rows = tuple(
        (tuple(int(x) for x in a_matrix.row(r)), beta[r]) for r in range(a_matrix.rows)
    )
    return GKZ(
        a_matrix=a_matrix,
        beta=beta,
        z_variables=tuple(system.z_variables),
        euler_rows=euler_rows,
        toric_generators=tuple(fi.toric_ideal.generators),
    )


def _symmetries(fi: FeynmanIntegral, data: PolytopeData) -> Symmetries:
    automorphisms = fi.polytope_automorphisms
    return Symmetries(
        automorphism_order=automorphisms.order,
        vertex_orbits=tuple(tuple(orbit) for orbit in automorphisms.vertex_orbits),
        graph_automorphisms=tuple(tuple(p) for p in fi.graph_automorphisms),
        coefficient_preserving=len(coefficient_preserving_indices(fi, automorphisms)),
        symmetry_pairs=tuple(fi.symmetry_pairs),
        full_dimensional=data.is_full_dimensional,
    )


def _landau(fi: FeynmanIntegral, analysis: LandauAnalysis) -> Landau:
    grouped: dict[int, list[sp.Expr]] = {}
    for face in analysis.face_discriminants:
        if face.discriminant == 1:
            continue
        distinct = grouped.setdefault(face.dimension, [])
        if face.discriminant not in distinct:
            distinct.append(face.discriminant)
    first: tuple[sp.Expr, ...] = ()
    second: tuple[sp.Expr, ...] = ()
    poles: tuple[sp.Expr, ...] = ()
    if fi.loop_count == 1:
        first, second = one_loop_landau_surfaces_by_type(fi)
        poles = one_loop_bridge_poles(fi)
    dimensions = [_affine_dimension(face) for face in analysis.skipped_faces]
    # Only P itself has the top dimension among the faces.
    top = max([face.dimension for face in analysis.face_discriminants] + dimensions, default=0)
    return Landau(
        analysis=analysis,
        by_dimension=tuple((d, tuple(faces)) for d, faces in sorted(grouped.items())),
        first_type=first,
        second_type=second,
        skipped=tuple(
            (d, len(face), d == top)
            for d, face in zip(dimensions, analysis.skipped_faces, strict=True)
        ),
        bridge_poles=poles,
    )


def _affine_dimension(points: tuple[tuple[int, ...], ...]) -> int:
    """The dimension of the affine hull of the points."""
    return _exact.affine_rank(points)


def _lp_to_cayley(n_edges: int, loop_count: int) -> sp.Matrix:
    """The unimodular map from the Lee-Pomeransky A-matrix to the Cayley one.

    The rows of the Lee-Pomeransky matrix are (1, ..., 1) and the exponents
    alpha_1, ..., alpha_N. The Cayley matrix of the Schwinger representation
    has the two grading rows (L + 1) 1 - sum_i alpha_i and -L 1 + sum_i
    alpha_i, which count the U~ and F~ blocks, followed by alpha_1, ...,
    alpha_{N-1}; the last parameter is set to one (Klausen 2023, section
    3.4).
    """
    t = sp.zeros(n_edges + 1, n_edges + 1)
    t[0, 0] = loop_count + 1
    t[1, 0] = -loop_count
    for j in range(1, n_edges + 1):
        t[0, j] = -1
        t[1, j] = 1
    for i in range(2, n_edges + 1):
        t[i, i - 1] = 1
    return t


def _columns(matrix: sp.Matrix) -> list[tuple[int, ...]]:
    return sorted(tuple(int(x) for x in matrix.col(j)) for j in range(matrix.cols))


def _schwinger(fi: FeynmanIntegral) -> Schwinger:
    system = fi.schwinger_gkz
    t = _lp_to_cayley(len(fi.symanzik.lp_parameters), fi.loop_count)
    return Schwinger(
        system=system,
        f_block=system.restrict_to_f_block(),
        lp_to_cayley=t,
        columns_match=_columns(t * fi.gkz.a_matrix) == _columns(system.a_matrix),
    )


# --- the report --------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisReport:
    """Everything feynkit computes for one integral, section by section.

    The first three sections are always present; the rest are None when the
    caller did not ask for them. ``torus`` holds the finite-field point counts
    of :meth:`FeynmanIntegral.torus_count`, whose candidates are not proven.
    """

    identity: Identity
    conventions: Conventions
    polynomials: Polynomials
    representations: Representations | None
    polytope: Polytope | None
    gkz: GKZ | None
    symmetries: Symmetries | None
    landau: Landau | None
    schwinger: Schwinger | None
    torus: TorusCount | None = None

    @classmethod
    def from_integral(
        cls,
        integral: FeynmanIntegral,
        sections: Collection[str] | None = None,
        *,
        max_face_points: int = 12,
        figure_max_vertices: int = 12,
        torus_seed: int = 0,
        torus_budget: int = 2 * 10**9,
    ) -> AnalysisReport:
        """Build the report for an integral.

        Parameters
        ----------
        integral
            The integral to describe.
        sections
            Names from :data:`SECTION_NAMES` to build; :data:`DEFAULT_SECTIONS`,
            every section but ``torus``, by default. ``identity``,
            ``conventions`` and ``polynomials`` are built whatever is asked for,
            since the rest of the report reads as a fragment without them.
        max_face_points
            Faces of the Newton polytope with more monomials than this are
            left out of the Landau analysis and listed as skipped.
        figure_max_vertices
            Polytopes with more vertices than this get no figure.
        torus_seed, torus_budget
            The ``seed`` and ``max_evaluations`` of
            :meth:`FeynmanIntegral.torus_count` for the ``torus`` section, which
            shares the Landau analysis with the ``landau`` section.

        Raises
        ------
        ValidationError
            If ``sections`` names something that is not a section, or as
            :meth:`FeynmanIntegral.torus_count` raises for the ``torus`` section,
            for instance when the integral has kinematic constraints, which is
            checked before any section is built, or when counting needs more
            than ``torus_budget`` evaluations of G.
        """
        wanted = set(DEFAULT_SECTIONS) if sections is None else set(sections)
        unknown = sorted(wanted - set(SECTION_NAMES))
        if unknown:
            raise ValidationError(
                f"Unknown report section(s): {', '.join(unknown)}; "
                f"expected any of {', '.join(SECTION_NAMES)}"
            )

        # torus_count rejects kinematic constraints; say so before the Landau analysis runs.
        if "torus" in wanted and integral.kinematic_constraints:
            raise ValidationError(
                "the torus section does not apply kinematic_constraints; leave it out, or count "
                "with FeynmanIntegral.torus_count and substitute them with on_shell"
            )

        representations: Representations | None = None
        polytope: Polytope | None = None
        symmetries: Symmetries | None = None
        if wanted & {"representations", "polytope", "symmetries"}:
            data = polytope_data(integral.newton_polytope.points)
            if "representations" in wanted:
                representations = _representations(integral, data)
            if "polytope" in wanted:
                polytope = _polytope(integral, data, figure_max_vertices)
            if "symmetries" in wanted:
                symmetries = _symmetries(integral, data)

        analysis: LandauAnalysis | None = None
        if wanted & {"landau", "torus"}:
            analysis = landau_analysis(integral, max_face_points=max_face_points)
        torus: TorusCount | None = None
        if "torus" in wanted:
            torus = integral.torus_count(
                seed=torus_seed, max_evaluations=torus_budget, landau=analysis
            )

        return cls(
            identity=_identity(integral),
            conventions=_conventions(integral),
            polynomials=_polynomials(integral),
            representations=representations,
            polytope=polytope,
            gkz=_gkz(integral) if "gkz" in wanted else None,
            symmetries=symmetries,
            landau=(
                _landau(integral, analysis) if analysis is not None and "landau" in wanted else None
            ),
            schwinger=_schwinger(integral) if "schwinger" in wanted else None,
            torus=torus,
        )

    def summary(self) -> tuple[tuple[str, str], ...]:
        """The report's numbers as (label, value) rows, sections absent omitted."""
        rows: list[tuple[str, str]] = [
            ("Loops", str(self.identity.loop_count)),
            ("Propagators", str(self.identity.propagators)),
            ("External legs", str(self.identity.external_legs)),
            ("Kinematic class", self.conventions.kinematic_class),
            ("Monomials of F", str(self.polynomials.monomials_f)),
            ("Monomials of G", str(self.polynomials.monomials_g)),
            ("Independent invariants", str(self.polynomials.independent_invariants)),
            ("Codimension", str(self.polynomials.codimension)),
            ("Scaleless", "yes" if self.polynomials.scaleless else "no"),
        ]
        if self.polytope is not None:
            rows.append(("Polytope vertices", str(len(self.polytope.data.vertex_indices))))
            rows.append(("Normalised volume", str(self.polytope.data.normalized_volume)))
        if self.torus is not None:
            master = self.torus.candidate_master_count
            rows.append(("Candidate master count", "none" if master is None else str(master)))
        if self.polytope is not None and self.polytope.invariants is not None:
            found = self.polytope.invariants
            index = found.gorenstein_index
            rows.append(("Lattice points", str(found.lattice_points)))
            rows.append(("Interior lattice points", str(found.interior_points)))
            rows.append(("Gorenstein index", "none" if index is None else str(index)))
            normal = "not computed" if found.normal is None else "yes" if found.normal else "no"
            rows.append(("Normal configuration", normal))
        if self.symmetries is not None:
            rows.append(("Polytope automorphisms", str(self.symmetries.automorphism_order)))
        if self.gkz is not None:
            rows.append(("Toric generators", str(len(self.gkz.toric_generators))))
        if self.landau is not None:
            rows.append(("Landau surfaces", str(len(self.landau.analysis.landau_surfaces))))
        return tuple(rows)
