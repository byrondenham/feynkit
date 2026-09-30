"""
Frozen description of everything feynkit computes for one Feynman integral.

:meth:`AnalysisReport.from_integral` gathers the graph data, the Symanzik
polynomials, the parametric representations, the Newton polytope and the
graphs of its faces, the GKZ system, the symmetries, the Landau surfaces and the Schwinger-representation
system into one immutable value, so that a renderer can walk the result
without reaching back into the integral. The module carries data only: it
makes no formatting decisions and holds no LaTeX beyond the TikZ figures it
stores. The data type of each section, the code that builds it and the
renderers of the section live in their own module of
:mod:`feynkit.io.sections`; this module keeps the report that holds them and
the context they build from.

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
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from ..core.exceptions import ValidationError
from ..landau import LandauAnalysis, landau_analysis
from ..point_count import TorusCount
from ..polytope import PolytopeData, polytope_data
from ..resonance import D0Source, choose_d0
from .sections import SECTIONS, summary_rows
from .sections.conventions import Conventions
from .sections.faces import FACE_CODIMENSION, Faces
from .sections.gkz import GKZ
from .sections.identity import Identity
from .sections.landau import Landau
from .sections.polynomials import Polynomials, ZEntry
from .sections.polytope import LATTICE_BUDGET, NORMALIZ_TIMEOUT, Polytope
from .sections.representations import Representations
from .sections.resonance import RESONANCE_WINDOW, Resonance
from .sections.schwinger import Schwinger
from .sections.symmetries import Symmetries

if TYPE_CHECKING:
    from ..integral import FeynmanIntegral

__all__ = [
    "DEFAULT_SECTIONS",
    "FACE_CODIMENSION",
    "LATTICE_BUDGET",
    "NORMALIZ_TIMEOUT",
    "RESONANCE_WINDOW",
    "SECTION_NAMES",
    "AnalysisReport",
    "BuildContext",
    "Conventions",
    "Faces",
    "GKZ",
    "Identity",
    "Landau",
    "Polynomials",
    "Polytope",
    "Representations",
    "Resonance",
    "Schwinger",
    "Symmetries",
    "ZEntry",
]


SECTION_NAMES = tuple(section.name for section in SECTIONS)

# The sections built when none are named: all but the point counts, which take seconds for
# five propagators; seven exceed the default budget.
DEFAULT_SECTIONS = tuple(name for name in SECTION_NAMES if name != "torus")


# --- the report --------------------------------------------------------------


@dataclass
class BuildContext:
    """What the sections build from: the integral, the options and the shared work.

    The polytope data and the Landau analysis are computed on first use and
    shared by every section that reads them.

    Attributes
    ----------
    integral
        The integral to describe.
    wanted
        The names of the sections asked for.
    max_face_points, figure_max_vertices, torus_seed, torus_budget, limits
        The arguments of :meth:`AnalysisReport.from_integral` of these names.
    d0, d0_source
        D_0 of the resonance section and where it came from.
    """

    integral: FeynmanIntegral
    wanted: frozenset[str]
    max_face_points: int
    figure_max_vertices: int
    torus_seed: int
    torus_budget: int
    limits: bool | str | None
    d0: Fraction
    d0_source: D0Source
    _data: PolytopeData | None = field(default=None, init=False, repr=False)
    _analysis: LandauAnalysis | None = field(default=None, init=False, repr=False)

    def polytope_data(self) -> PolytopeData:
        """The data of the Newton polytope, computed once."""
        if self._data is None:
            self._data = polytope_data(self.integral.newton_polytope.points)
        return self._data

    def analysis(self) -> LandauAnalysis:
        """The Landau analysis, computed once and shared by ``landau`` and ``torus``.

        The point counts use the principal Landau determinant, not the limit
        surfaces, so the analysis looks for limits only when ``landau`` is
        wanted.
        """
        if self._analysis is None:
            self._analysis = landau_analysis(
                self.integral,
                max_face_points=self.max_face_points,
                limits=self.limits if "landau" in self.wanted else False,
            )
        return self._analysis


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
    resonance: Resonance | None = None
    faces: Faces | None = None

    @classmethod
    def from_integral(
        cls,
        integral: FeynmanIntegral,
        sections: Collection[str] | None = None,
        *,
        max_face_points: int = 14,
        figure_max_vertices: int = 12,
        torus_seed: int = 0,
        torus_budget: int = 2 * 10**9,
        limits: bool | str | None = None,
        d0: int | Fraction | None = None,
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
        limits
            Whether the ``landau`` section looks for limit surfaces, as the
            ``limits`` of :func:`~feynkit.landau.landau_analysis`: True, False or
            ``"one-loop"``; None, the default, takes its default. The ``torus``
            section never does.
        d0
            D_0 of the ``resonance`` section, which takes D = D_0 - 2 eps: an
            integer or a Fraction. None, the default, reads it from the
            dimension of the integral when that is D_0 - 2 eps with D_0 a
            number, and takes 4 otherwise; see
            :func:`feynkit.resonance.choose_d0`.

        Raises
        ------
        ValidationError
            If ``sections`` names something that is not a section, or as
            :meth:`FeynmanIntegral.torus_count` raises for the ``torus`` section,
            for instance when the integral has kinematic constraints, which is
            checked before any section is built, or when counting needs more
            than ``torus_budget`` evaluations of G; or if ``d0`` is not an
            integer or a Fraction, also checked first.
        """
        wanted = set(DEFAULT_SECTIONS) if sections is None else set(sections)
        unknown = sorted(wanted - set(SECTION_NAMES))
        if unknown:
            raise ValidationError(
                f"Unknown report section(s): {', '.join(unknown)}; "
                f"expected any of {', '.join(SECTION_NAMES)}"
            )

        d0_value, d0_source = choose_d0(d0, integral.dimension)
        # torus_count rejects kinematic constraints; say so before the Landau analysis runs.
        if "torus" in wanted and integral.kinematic_constraints:
            raise ValidationError(
                "the torus section does not apply kinematic_constraints; leave it out, or count "
                "with FeynmanIntegral.torus_count and substitute them with on_shell"
            )

        context = BuildContext(
            integral=integral,
            wanted=frozenset(wanted),
            max_face_points=max_face_points,
            figure_max_vertices=figure_max_vertices,
            torus_seed=torus_seed,
            torus_budget=torus_budget,
            limits=limits,
            d0=d0_value,
            d0_source=d0_source,
        )
        built: dict[str, Any] = {
            section.name: (
                section.build(context) if section.always or section.name in wanted else None
            )
            for section in SECTIONS
        }
        return cls(**built)

    def summary(self) -> tuple[tuple[str, str], ...]:
        """The report's numbers as (label, value) rows, sections absent omitted."""
        return summary_rows(self)
