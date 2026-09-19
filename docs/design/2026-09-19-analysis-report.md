# Analysis report

Design note for replacing the LaTeX and plain-text documents produced by
`FeynmanIntegral.to_latex` and `to_text`.

## Purpose

The current document is a concatenation of strings written when the package
had fewer features. It compiles, but it states no conventions, never defines
the variables it uses, calls toric generators IBP relations, omits the Newton
polytope, the convergence region, the Landau surfaces and the symmetries, and
wraps the results in boilerplate. This note specifies a document that says
what a reader of a GKZ analysis of a Feynman integral expects to find, with
each claim tied to its source, built from one description of the results and
rendered to LaTeX or text.

## What the literature presents

The sections below follow what the papers and packages that analyse specific
integrals actually report.

- FeynGKZ (Ananthanarayan, Banik, Bera, Datta, arXiv:2211.01285, sections 5
  and 6): $U$, $F$, $G$, the A-matrix with its codimension $N - n - 1$, the
  normalised volume, and the explicit substitution from generic coefficients
  to physical kinematics.
- Klausen's thesis (arXiv:2302.13184, Table A.4): per graph the monomial
  counts of $\mathcal{F}$ and $\mathcal{G}$, the number of physically
  independent variables, the corank, the volume and the face counts. Section
  3.3 gives the convergence region and the Gamma factor per facet; section
  3.4 the Cohen-Macaulay condition under which rank equals volume for all
  parameters.
- Bitoun, Bogner, Klausen, Panzer (arXiv:1712.09215, section 3): the number
  of master integrals is the Euler characteristic of the complement of
  $\{\mathcal{G} = 0\}$ in the torus, bounded above by the normalised volume,
  with equality only for generic coefficients.
- Chestnov et al. (arXiv:2204.12983, Thm 2.1, Rmk 6.2) and Henn, Pratt,
  Sattelberger, Zoia (arXiv:2303.11105, Table 1): rank equals volume for
  non-resonant $\beta$ and $z$ off the principal A-determinant; the physical
  restriction can only lower the rank; toric relations are an analogue of
  IBP relations, not IBP relations.
- Dlapa, Helmer, Papathanasiou, Tellander (arXiv:2304.02629) and Fevola,
  Mizera, Telen (arXiv:2311.14669): the principal A-determinant over all
  faces, the first-type and second-type labelling of its factors, and the
  caveat that its factors are candidate loci on all sheets.

## Structure

`feynkit/io/report.py` defines `AnalysisReport`, a frozen dataclass holding
every fact the document states, built once by `AnalysisReport.from_integral`.
Two renderers, `render_latex` and `render_text`, consume it. `to_latex`,
`to_text` and the `fk` command keep their signatures and call these. Each
section is a field that may be `None`, and `from_integral` takes a set of
section names so a survey can build a short report and the dissertation a
full one without recomputing.

Fields, grouped by section:

| Section | Fields |
|---|---|
| identity | CNickel, loops, propagators, legs, masses, graph figure (TikZ) |
| conventions | dimension symbol, external-mass and planar-invariant symbols, propagator form, sign of $F$ |
| polynomials | $U$, $F$, $G$; degrees; monomial counts; the $z_j$ table |
| representations | Schwinger, Feynman, Lee-Pomeransky prefactors and integrands in terms of $U$, $F$, $G$; convergence region as facet inequalities |
| polytope | vertices, dimension, full-dimensionality, normalised volume, face counts, figure |
| gkz | $A$, $\beta$, Euler operators in $\theta$ notation, toric generators |
| symmetries | polytope automorphism order and orbits, graph automorphisms, coefficient-preserving subgroup, symmetry pairs |
| landau | face discriminants by dimension, surfaces, skipped faces |
| schwinger | the Cayley system and its facet reduction, once built |

## The document

1. Title with the CNickel string. No abstract, author or date unless given.
2. Summary table: loops, propagators, legs, monomials of $F$ and $G$,
   independent invariants, corank, polytope vertices, normalised volume,
   $|\mathrm{Aut}(P)|$, toric generator count, Landau surface count.
3. Graph figure.
4. Conventions. $D = D_0 - 2\epsilon$ with $D_0$ stated; mostly-minus
   metric; propagators $1/(-q^2 + m^2)^{\nu}$; the invariants as defined in
   `standard_invariants`; $F = -\sum_{2\text{-forests}} s_F \prod a_e +
   U \sum m_e^2 a_e$ divided by $\mu^2$ (Weinzierl 2022, section 2).
5. Symanzik polynomials. $U$, $F$ with the $1/\mu^2$ factored out, $G$.
   Degrees $L$ and $L+1$. The $z_j$ table: index, monomial, coefficient,
   physical value. Codimension $N - n - 1$ against the number of independent
   invariants (Klausen 2023, Table A.4).
6. Representations. Each written symbolically in $U$, $F$ or $G$ with its
   prefactor, citing Weinzierl for Schwinger and Feynman and Lee and
   Pomeransky (2013) for the third. Convergence region of the Lee-Pomeransky
   integral as the facet inequalities $b_j \,\mathrm{Re}\,\nu_0 -
   m_j^\top \mathrm{Re}\,\nu > 0$ and the statement that a non-full-dimensional
   polytope means the integral converges nowhere (Klausen 2023, Thm
   FIconvergence).
7. Newton polytope. Vertices, dimension, volume, face counts, figure when
   the vertex count is at most twelve. The rank statement with its
   conditions: rank equals volume for non-resonant $\beta$ and generic $z$
   (Adolphson, via Chestnov et al. Thm 2.1); for all $\beta$ when the toric
   ideal is Cohen-Macaulay, which holds for all-massive and all-massless
   graphs (Klausen 2023, Thm CohenMacaulayFeynman); the master-integral count
   is the Euler characteristic and is at most the volume (BBKP, section 3),
   with the remark that graph-polynomial coefficients are rarely generic.
   feynkit does not compute the rank or the Euler characteristic; the
   document says so.
8. GKZ system. $A$; $\beta = (-D/2, -\nu)$ with the operator form and the
   citations (de la Cruz 2019; Klausen 2020, Thm 3.1); Euler operators as
   $\theta_j = z_j \partial_j$, one line each, interpreted as index and
   dimension shifts; toric generators described as annihilators of the
   generalised integral and "an analogue of IBP relations" (Chestnov et al.
   Thm 6.1), with the note that the physical specialisation is a separate
   step.
9. Symmetries. $\mathrm{Aut}(P)$ order and vertex orbits, graph
   automorphisms, coefficient-preserving subgroup, and the identities
   $I(\beta, z) = I(T\beta, z_P)$ from symmetry pairs (de la Cruz 2024).
10. Landau surfaces. The reduced principal A-determinant by face dimension,
    labelled first-type or second-type for one-loop graphs by comparison with
    the closed form, with the caveats from the Landau module docstring.
    Skipped faces listed.
11. Schwinger-representation system, when present: the Cayley matrix,
    $\beta$, the equivalence to the Lee-Pomeransky configuration, and the
    facet reduction with the caveat from its design note.
12. References, as a `thebibliography` list of the works cited above.

## Typography

- Products by juxtaposition; SymPy's `mul_symbol` unset.
- Long polynomials through `to_latex_split`; matrices in `bmatrix`.
- No forced page breaks; sections flow.
- Symbols: $\nu_i$, $D$, $\epsilon$, $\mu$, $a_e$ for Schwinger parameters,
  $u_e$ for Lee-Pomeransky parameters, $z_j$ for coefficients, $\theta_j$ for
  Euler operators, consistent with the mathematics reference.

## Testing

- The document for the massive bubble, the massless triangle and the box
  compiles under pdflatex with no errors and no overfull boxes, skipped when
  pdflatex is absent.
- Content tests on `AnalysisReport`: the $z_j$ table matches `gkz.support`,
  the summary numbers match the objects they summarise, the convergence
  inequalities are one per facet, the Landau section matches
  `landau_analysis`.
- Golden tests on short rendered fragments (the $\beta$ paragraph, one Euler
  operator line, the conventions paragraph) so wording changes are
  deliberate.
- The text renderer reports the same facts as the LaTeX one, checked by
  rendering both from the same `AnalysisReport` and comparing the numbers.

## Out of scope

Series solutions and triangulations, the holonomic rank and Euler
characteristic, Pfaffian systems, and any restriction of the GKZ system to
physical kinematics. The document names each of these as not computed rather
than leaving the reader to guess.
