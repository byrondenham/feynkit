# Changelog

## Unreleased

## 0.3.0 (2026-09-25)

### Breaking changes

- Kinematic invariants. With `use_mandelstam=True` the external dot products
  are now written in the standard invariants: the external masses p_i^2 and
  the planar invariants s_{i...j-1} = (p_i + ... + p_{j-1})^2, so that for four
  legs the variables are s, t and p_1^2 ... p_4^2. For two legs the single
  invariant is s = p^2 with p_1 . p_2 = -s, so the massive bubble's threshold
  sits at s = (m_1 + m_2)^2 as in the literature. Previously s_ij meant
  2 p_i . p_j and the bubble threshold sat at s = -2 (m_1 + m_2)^2. A graph
  with fewer than two external legs, such as the tadpole `0|:n`, now raises
  `ValueError` with `use_mandelstam=True`, the default; build it with
  `use_mandelstam=False`, as `fk` does.
- `feynkit.landau` now computes the reduced principal A-determinant over all
  faces of the Newton polytope, not only its edges, and the edge computation
  used a wrong exponent (the dot product with the direction instead of the
  lattice coordinate), which squared the Källén factor of the bubble and
  produced spurious surfaces for the massless triangle. `EdgeDiscriminant`
  is replaced by `FaceDiscriminant`, `landau_polynomial` by
  `principal_a_determinant`, and `skipped_faces` is added.
- The GKZ parameter vector `FeynmanIntegral.gkz.beta_parameters` is now
  beta = (-D/2, -nu_1, ..., -nu_N), the value the Euler equations
  sum_j A_rj z_j d/dz_j Phi = beta_r Phi require for the Lee-Pomeransky
  integral (de la Cruz 2019; Klausen 2020). Earlier versions reported
  (sum(nu) - D/2, nu_1, ..., nu_N), which does not satisfy those equations. A
  numerical homogeneity test now pins the correct values. The A-matrix, toric
  ideal and polytope data are unchanged.
- `FeynmanIntegral.to_latex` and `to_text` now produce the analysis report
  (see Added) and take `sections=None, *, title=None, max_face_points=12`.
  They no longer accept `author`, `polytope_tikz` or any other keyword
  argument, and the default title is "Feynman integral" followed by the
  CNickel string.
- The exporters behind the old documents are removed: `parametrisation_to_latex`,
  `gkz_system_to_latex`, `toric_ideal_to_latex` and `_create_analysis_document`
  from `feynkit.io.latex`, and `parametrisation_to_text`, `gkz_system_to_text`,
  `toric_ideal_to_text` and `_create_analysis_report` from `feynkit.io.text`.
  Build an `AnalysisReport` and pass it to `render_latex` or `render_text`
  instead.
- `landau_analysis` no longer counts the energy scale mu as a Landau surface or
  as a factor of `principal_a_determinant`: mu only normalises the
  coefficients z_j. Where it appeared the surface count drops by one, from 6
  to 5 for the massive bubble and from 9 to 8 for the one-mass triangle and
  the massive sunrise. `FaceDiscriminant.discriminant` still carries mu.
- `fk` no longer accepts abbreviated long options, such as `--sym` for `--symanzik`, and section
  flags given with two diagrams are a usage error (exit status 2) rather than ignored.

### Added

- `FeynmanIntegral.schwinger_gkz`: the two-block Cayley GKZ system of the
  Schwinger representation (Jimenez-Santacruz, Lopez-Arcos, Quintero Velez
  2026; Klausen 2023, section 3.4), with its toric ideal and the face
  subsystem on the F block. `SchwingerParametrisation.get_A_matrix` now
  delegates to it.
- `one_loop_landau_surfaces` and `one_loop_principal_a_determinant`: the
  one-loop closed form of Dlapa, Helmer, Papathanasiou and Tellander (2023)
  from the principal minors of the modified Cayley matrix, used to check the
  face computation.
- Singular is used for the elimination ideals of non-simplex faces when the
  `Singular` binary is installed; SymPy is the fallback.
- `feynkit.kinematics` now exports `KinematicInvariants` and
  `standard_invariants`.
- An opt-in smoke test that runs every script in `examples/` and checks that it
  exits 0 without writing into the working directory. It is skipped by default;
  run it with `pytest -m examples`.
- `hull_vertex_indices` is now exported from `feynkit.normal_forms`, so callers
  no longer have to reach into the private `_invariants` module.
- `feynkit.polytope`: `polytope_data(points)` returns a `PolytopeData` with the
  vertices, the face lattice, the normalised volume and, for a full-dimensional
  polytope, the facet inequalities m . x <= b as `Facet` objects with primitive
  integer outward normals. All three names are exported from `feynkit`.
- `feynkit.io.AnalysisReport`, a frozen record of everything feynkit computes
  for one integral, built by `AnalysisReport.from_integral(fi, sections)`, and
  the renderers `render_latex` and `render_text`, which write it as a LaTeX
  article or as plain text. The report states its conventions and cites a
  source for each claim. It gives the Symanzik polynomials with the
  coefficients z_j at their physical values, the three parametric
  representations with the convergence region of the Lee-Pomeransky integral,
  the Newton polytope and the conditions under which the holonomic rank equals
  its volume, the GKZ system, the symmetries and the identity each symmetry
  pair gives, the Landau surfaces, and the Schwinger-representation system. It
  also names what feynkit does not compute. The monomials of F are counted by
  exponent vector, so terms that share a monomial count once.
- The report leaves the symmetries out, and says why, when the Newton polytope
  has dimension below 2, as for the massive tadpole `0|:n`, whose polytope is
  a segment, or is not full-dimensional, as for `012e|2e|e|:znnn`: the
  automorphism computation is built for full-dimensional polytopes of
  dimension 2 and above.
- `feynkit.io.latex.factor_energy_scale(expr, scale)`, which writes an
  expression as numerator / scale^k with the numerator free of the scale, and
  `to_latex_lines`, which breaks the LaTeX of a long sum into lines.
- `one_loop_landau_surfaces_by_type`, which splits the one-loop closed form into
  first-type (Cayley) and second-type (Gram) factors, and a keyword-only `scale`
  argument of `landau_analysis_from_polynomial` that keeps that symbol out of
  the surfaces.
- A test that compiles the report of the massive bubble, the massless triangle
  and the massless box with pdflatex and fails on errors, overfull lines or
  undefined references. It is skipped when pdflatex is not installed, and for
  the box when Singular is not. CI installs TeX Live so that it runs there.
- `fk` has two subcommands: `fk analyse CNICKEL` for one diagram and `fk compare A B` for two.
  The bare forms `fk CNICKEL` and `fk A B` still work and run them. `fk --version` prints the
  version.
- `fk --no-db` skips the database.
- `fk analyse --latex FILE` and `--text FILE` write the analysis report of
  `FeynmanIntegral.to_latex` and `to_text`, with `--sections` choosing its sections, and
  `fk analyse --json` prints the report's summary and the CNickel string as JSON. The report is
  built once however many of the three are asked for.
- `fk --verbose` prints the time of each stage on stderr, and `fk` flushes its output after each
  section.
- `CITATION.cff`, so that GitHub and reference managers can cite feynkit.

### Changed

- The examples keep their output out of the repository root: the survey
  databases and text reports, and the TikZ files of
  `examples/visualisation_example.py`, are written under `examples/output/`.
  `examples/toric_survey.py` now has a database of its own rather than sharing
  `examples/feynkit_survey.py`'s, because its equivalence analysis iterates
  every record in the database and sharing a file would report classes over
  `feynkit_survey.py`'s conformal configurations too.
- The two slow examples have an `--all` flag, as `examples/landau_analysis.py`
  already had, and their default runs take seconds rather than minutes.
  `examples/equivalence_survey.py` surveys the triangle, box and pentagon and
  takes the 64 hexagon variants only under `--all`;
  `examples/higher_analogues_search.py` keeps the lattice invariants and the
  finite-index and unimodular verdicts of phases 1 and 2, and takes the
  phase-2 affine-equivalence searches and the phase-3 sweep over 2,825,761
  candidate maps only under `--all`.
- `FeynkitDatabase.summary()` now shows `0` for a cached toric ideal with no
  generators, rather than the same `?` it shows for one that was never
  computed.
- `feynkit.io.latex.to_latex` writes products by juxtaposition (`2 x y`) rather
  than with `\cdot`.
- `one_loop_landau_surfaces` lists the first-type (Cayley) factors before the
  second-type (Gram) ones, as `one_loop_landau_surfaces_by_type` splits them.
  The set of factors is unchanged.
- `graph_to_tikz` places each external leg outward from its own vertex and
  lays out one-loop graphs as polygons along the walk of internal edges.
  It draws a self-loop as a loop, where it used to draw a line from the vertex
  to itself, and several self-loops at one vertex take the sides above, below,
  left and right in turn.
- `docs/triangle_analysis.tex`, `.pdf` and `.txt` are regenerated from the new
  report. They describe the same graph as before, the triangle with three
  distinct masses (`12e|2e|e|:nnn`).
- The `fk` help quotes every CNickel example, since an unquoted | is a shell pipe, and no longer
  repeats the section flags in its epilog.
- `fk` reports a CNickel string that does not parse, a feynkit error or a database error in one
  line on stderr, without a traceback, and exits with status 1; a parse error also gives the
  CNickel grammar and a quoted example. Usage errors still exit with status 2. When the reader of
  the output closes the pipe early, as `head` does, `fk` stops quietly with status 141, which is
  128 + SIGPIPE. The database is closed on every path.
- The guide's CLI section and the README describe `fk analyse` and `fk compare`, the report and
  JSON options, `--no-db`, `--verbose` and the exit status, and quote every CNickel string. A test
  takes each `fk` command in the `bash` blocks of those two sections and checks that it quotes its
  CNickel strings, that `fk` accepts its arguments and that its CNickel strings parse; it also
  checks that the guide's section names every long option other than `--help`.
- The error for a CNickel string with the wrong number of mass codes says "Mass-colour" rather
  than "Mass-color".

### Fixed

- Section 4.5 of the mathematics reference no longer says that resonant parameters raise the
  holonomic rank. It now says that the rank can exceed the volume only when the toric ring is
  not Cohen-Macaulay (Matusevich, Miller, Walther 2005), gives the hypotheses of Klausen's
  Theorem 3.4.2 under which Feynman configurations are Cohen-Macaulay, notes that
  Cohen-Macaulayness can fail outside them (Michaelsen, Tellander 2025), and that resonance
  means reducibility (Schulze, Walther 2012). The literature review is corrected in the same
  way, and the 1989 Gelfand, Zelevinsky, Kapranov paper now has its correct title.
- `examples/dissertation_overview_enhanced.py` died part way through: section 13
  passed an `AConfiguration` to `hull_vertex_indices`, which wants an array of
  points, and section 20 looked up a malformed cnickel string. It now runs to
  the end.
- A cached integral with an empty toric ideal (e.g. the massless bubble) no
  longer raises on lookup.
- The symmetry-pair identity was stated with the permutation on the wrong side,
  as I_A(beta, z) = I_A(T beta, z_P), which fails whenever P is not an
  involution. It now reads I_A(beta, z_P) = I_A(T beta, z), with
  z_P = (z_{P(1)}, ..., z_{P(N)}) and I_A the integral without Gamma prefactors.
- For a map between two diagrams, `fk` and `examples/feynkit_survey.py` printed
  I_A(beta, z_P) = I_A(T*beta, z), which lacks the factor |det M| and names A
  where the target B is meant. They also gave the exponent map
  u_i = sum_j M_ij v_j + t_i as the change of variables. They now print P, the
  substitution u_i = prod_k v_k^(M_ki) and I_A(beta, z_P) = |det M| I_B(T beta, z),
  and only for maps of every column (point_config, finite_index); a map of the
  hull vertices gives no identity. The docstrings no longer say that
  `symmetry_pairs` can return maps with |det M| > 1.
- The LaTeX document did not escape its title, so a title with `_`, `&` or `%`
  failed to compile. It showed at most five Euler equations; the report writes
  one Euler operator per row of A.
- The documents called the toric generators IBP relations and the Euler
  equations Horn-type. The report describes the toric operators as an analogue
  of IBP relations (Chestnov et al. 2022) and makes no Horn claim. The guide,
  the README, the mathematics reference and the `feynkit.algebra.toric`
  docstrings no longer call the toric generators IBP relations either.
- `fk` called the toric ideal "IBP relations in z-space" and, for a trivial
  toric ideal, printed "no IBP relations, single master integral", which does
  not follow: the massless tadpole has a trivial toric ideal and no master
  integral. It now prints the toric ideal, glosses its generators as an
  analogue of IBP relations and calls a trivial one the zero ideal. The
  symmetry section no longer counts finite-index symmetry pairs, since every
  self-map has |det M| = 1.
- The LaTeX document pointed to github.com/feynkit/feynkit, which does not
  exist.
- `graph_to_tikz` drew parallel propagators, as in the bubble and the sunrise,
  as one line; it now bends them apart.
- `to_latex_split` could break a line inside a `\left( ... \right)` pair and
  leave its delimiters unbalanced.
- The mathematics reference gave the exponent map u -> Mu + t as the change of
  integration variables, called point-configuration equivalence unimodular
  although it allows a rational M, and said the conformal companion maps to
  BMS_n with |det M| = 2 for every n and unimodularly for n = 3. It now gives
  the substitution u_i = prod_k v_k^(M_ki), the identity
  I_A(beta, z_P) = |det M| I_B(T beta, z) for a map of every column, and
  |det M| = 2/(n-2), an integer map only for n = 3. The de la Cruz (2024)
  reference has its arXiv number, 2404.03564.
- The guide listed a `witness_matrix` field of `SymmetryPair`, which is
  `linear_map`, split symmetry pairs into unimodular and finite-index ones,
  although every self-map has |det M| = 1, wrote the Lee-Pomeransky integrand
  with G^(d/2 - E/2) instead of G^(-D/2), and said the Newton polytope of the
  triangle is a line segment; its dimension is 3.
- `examples/symmetry_pairs_example.py` and the mathematics reference put
  |det M| = 1 for self-maps down to Smith invariants equal to 1; it holds for
  every full-dimensional configuration, since P has finite order k and so
  M^k = I. The `finite_index_map` docstring had the index the wrong way round:
  the image M Z^n of the source lattice has index |det M| in the target
  lattice Z^n.
- `docs/automorphism_groups.md` and the `coefficient_preserving_indices`
  docstring claimed functional equations I(z) = I(sigma . z) between kinematic
  points. A coefficient-preserving automorphism gives
  I_A(beta, z) = I_A(T beta, z), a relation between parameter vectors at one
  kinematic point. The literature review no longer says the IBP-like relations
  are the polytope-symmetry relations, the Euler equations are no longer called
  Horn equations, and the conformal companion's map to BMS_n is no longer said
  to have det 2 for every n. The mathematics reference now defines the
  normalised volume in the lattice the points span, as feynkit computes it.
  The guide cited de la Cruz (2019) as arXiv:1907.01007; it is 1907.00507.
- `FeynkitDatabase` left its SQLite connection open when opening failed, for example on a file
  that is not a database; it now closes it before raising.
- `fk compare` printed the canonical CNickel string for both diagrams, so two inputs with the
  same canonical form looked identical. `fk` now prints each string as given, with the
  canonical form beside it when they differ.
- The Newton polytope section of `fk analyse` calls the normalised volume the holonomic rank
  for generic beta, not the holonomic rank.
- The examples `dissertation_overview.py`, `dissertation_overview_enhanced.py`,
  `complete_analysis.py`, `feynkit_survey.py`, `toric_ideal_example.py`,
  `automorphism_survey.py`, `conformal_simplex_comparison.py`,
  `dissertation_configuration_analysis.py` and `symmetry_pairs_example.py` called the toric
  generators IBP relations or identities, inferred a single master integral from a trivial toric
  ideal or a volume of 1, equated the holonomic rank with the number of master integrals,
  counted finite-index symmetry pairs, of which there are none, and stated functional equations
  I(z) = I(sigma z). They now call the toric operators an analogue of IBP relations (Chestnov et
  al. 2022), say that a trivial toric ideal gives no master-integral count and that the volume
  only bounds it, and give I_A(beta, z) = I_A(T beta, z) for a coefficient-preserving
  automorphism. They also give A with E+1 rows rather than L+1, use the library's
  beta = (-D/2, -nu_1, ..., -nu_n), and write the finite-index identity as one term with the
  factor |det M|. They no longer cite an epsilon-expansion or resonance module, which feynkit
  does not have, or claim conformal invariance of the triangle in D = 2. An audit of the nine
  scripts removed further unsupported statements, among them a database index on volume and
  Smith invariants that does not exist, a parity projection in the triangle to triple-K map,
  and a sunrise with two thresholds where the output lists four.
- The guide said that a trivial toric ideal often indicates a master integral; such an ideal
  gives no master-integral count.
- The mathematics reference gave the polytope automorphism group of the L-loop massless banana
  as S_{L+1}, of order (L+1)!. Its L + 1 propagators give L + 2 monomials forming a unimodular
  simplex, so the group is S_{L+2}, of order (L+2)!: 6 for the bubble, 24 and 120 for three and
  four propagators.
- The symbol tables of the mathematics reference swapped the Schwinger and Feynman parameters.
  U, F and the Feynman representation use the Schwinger parameters `a_{e.idx}` of
  `graph.schwinger_parameters`, and the Schwinger representation renames them `alpha_{e.idx}`.
- The mathematics reference labelled Klausen's thesis, arXiv:2302.13184, as Klausen (2022) in
  two places, a clash with the 2022 JHEP paper, and the literature review and the analysis
  report's bibliography dated the thesis 2022. All three now give 2023, and
  `docs/triangle_analysis.tex`, `.pdf` and `.txt` are regenerated.
- The Symanzik section of `fk analyse` listed the Schwinger representation's parameters alpha_e
  above U and F, which are written in the a_e. It now lists the Schwinger parameters a_e of
  `graph.schwinger_parameters`.
- `fk` ended in an uncaught ValueError traceback on the massive tadpole `0|:n`, part way through
  the Newton section: the tadpole's Newton polytope is a segment, on which the convex hull of
  `AConfiguration` fails. When the Newton polytope has dimension below 2, `fk analyse` now takes
  its vertices and volume from `polytope_data` and leaves out the symmetries, as the report does,
  and `fk compare` skips the unimodular and affine-polytope checks. `fk` builds a graph with fewer
  than two external legs without the Mandelstam invariants, which need two legs, and reports an
  integral that cannot be built in one line, apart from parse errors and without the CNickel
  grammar.
- `fk analyse -g` wrote every term of an Euler equation with coefficient 1, so the equations of
  a massive diagram, whose A-matrix has entries 2, were wrong: for `11e|e|:nn` it printed
  `z_1 d_1 + z_2 d_2 + z_4 d_4 = -nu_1` for the row (2, 1, 0, 1, 0). It now writes each term as
  A_rj z_j d_j, here `2 z_1 d_1 + z_2 d_2 + z_4 d_4 = -nu_1`, in the order of the columns rather
  than with z_10 before z_2.
- `fk analyse -n` ended in a traceback when the Newton polytope was not full-dimensional, as for a
  graph with a massless self-loop: at the lattice base point for `01e|e|:zn`, whose polytope is a
  segment in R^2, after giving its volume as 0, and at the normalised volume for `011e|e|:znn` and
  `012e|2e|e|:zzzz`, whose polytopes have dimension 2 in R^3 and 3 in R^4. `fk compare` failed at
  the normalised volume or in the finite-index check, and printed `unimodular no` even for a
  diagram and itself. For such a polytope `fk analyse` and `fk compare` now take the vertices and
  the volume from `polytope_data`, as the report does, the Newton section leaves out the lattice
  base point, which the report does not show either, and `fk compare` prints n/a for the
  unimodular, affine-polytope and finite-index checks. Nor does the Newton section call the
  normalised volume the holonomic rank for generic beta there: the rows of A are dependent, so for
  generic beta the Euler equations contradict each other and the rank is 0. The analysis report
  says the same.
- `fk analyse -S` gave |Aut(P)| = 1 when the Newton polytope was not full-dimensional, as for
  `012e|2e|e|:znnn`, although all six permutations of u_2, u_3 and u_4 preserve the monomials of
  G: the automorphism computation is built for full-dimensional polytopes. The symmetry section
  now leaves out the symmetries of such a polytope and says why, as the report does.

### Removed

- The unused `matplotlib` dependency.
- `test_installation.py`, superseded by the test suite.
- The empty `feynkit.utils` package.
- The tracked LaTeX build files under `docs/` (`.aux`, `.fdb_latexmk`, `.fls`,
  `.out`, `.toc`); these are now git-ignored.

## 0.2.0 (2026-09-19)

### Breaking changes

- Python 3.10 or newer is required; 3.9 reached end of life in October 2025.
- The `method` keyword has been removed from `is_affinely_equivalent`,
  `is_point_config_equivalent` and `FeynmanIntegral.is_affinely_equivalent_to`.
  The `"sage"` option it accepted was a placeholder that ran the SymPy search anyway.

### Added

- `feynkit.algebra.ideal_quotient` and `feynkit.algebra.intersect_ideals`, computed
  by elimination.
- `feynkit.algebra.compute_syzygy_module`, which returns generators of the first
  syzygy module via Buchberger's algorithm with cofactor tracking.
- `SchwingerParametrisation.dehomogenised_symanzik_polynomials` and
  `SchwingerParametrisation.get_A_matrix`, giving the GKZ form of the Schwinger
  representation.
- `reduce_polynomial` and `is_in_ideal` accept an `order` argument so the reduction
  can match the monomial order of the supplied Gröbner basis.
- `feynkit.io.latex.to_latex_split` now splits long expressions at top-level
  plus and minus signs instead of returning them unchanged.
- A GitHub Actions workflow runs the test suite on Python 3.10 and 3.14 and the
  full pre-commit hook set.
- Tests for the database, the `fk` command line, the section exporters and the
  3D geometry helpers.

### Fixed

- `import feynkit` failed on Python 3.13 and earlier because of an invalid
  annotation in the Schwinger module.
- The SQLite toric-ideal cache raised an `IndexError` on every hit, so results
  were never reused.
- `FeynkitDatabase.find_equivalent` computed the toric ideal of the query
  integral without using it.
- An undefined name in `examples/higher_analogues_search.py`.

### Removed

- The Sphinx configuration entries and the readthedocs URL. `docs/guide.md` is the
  documentation.
- Saved CLI output files from the repository root.
- Unreachable planar-polytope code in `feynkit.visualisation.geometry`.

## 0.1.0

Initial release.
