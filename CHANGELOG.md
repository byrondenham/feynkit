# Changelog

## Unreleased

### Breaking changes

- Kinematic invariants. With `use_mandelstam=True` the external dot products
  are now written in the standard invariants: the external masses p_i^2 and
  the planar invariants s_{i...j-1} = (p_i + ... + p_{j-1})^2, so that for four
  legs the variables are s, t and p_1^2 ... p_4^2. For two legs the single
  invariant is s = p^2 with p_1 . p_2 = -s, so the massive bubble's threshold
  sits at s = (m_1 + m_2)^2 as in the literature. Previously s_ij meant
  2 p_i . p_j and the bubble threshold sat at s = -2 (m_1 + m_2)^2.
- `feynkit.landau` now computes the reduced principal A-determinant over all
  faces of the Newton polytope, not only its edges, and the edge computation
  used a wrong exponent (the dot product with the direction instead of the
  lattice coordinate), which squared the Kallen factor of the bubble and
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
- An `AConfiguration` without points raises `ValidationError` from
  `normalized_volume`, where it returned 1. `AConfiguration.normalized_volume`,
  `faces` and `polytope_data` raise `ComputationError` if a consistency check of
  the exact computation fails, which would be a bug; the volume used to be 0
  whenever Qhull failed.
- `faces`, `polytope_data`, `lattice_coordinates` and `intrinsic_lattice_model`
  raise `ValidationError` on a non-integer coordinate, which `faces` used as a
  float and the others truncated, and on points with different numbers of
  coordinates, which raised `ValueError` or `TypeError`. Integral floats such as
  `2.0` are still accepted. `intrinsic_lattice_model` also raises
  `ValidationError` when there are no points, where it raised `IndexError`.
- `PolytopeData` gains four fields and `Facet` three (see Added). The new
  `PolytopeData` fields have no defaults, so code that builds a `PolytopeData`
  by hand must pass them. The new `Facet` fields have defaults but take part in
  equality, so a `Facet(normal, offset, point_indices)` built by hand no longer
  equals the facet `polytope_data` returns.

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
- The report leaves the symmetries out, and says so, when the Newton polytope
  has dimension below 2, as for the massive tadpole `0|:n`, whose polytope is
  a segment: the automorphism computation is built for two dimensions and
  above.
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
- `feynkit.polytope.normalized_volume(points)`, the exact normalised volume, and
  a keyword `backend` of `faces`, `polytope_data` and `normalized_volume` that
  chooses where facet candidates come from: `"python"` (integer beneath-beyond,
  the reference, which the default `"auto"` uses), `"qhull"` or `"normaliz"`.
  Every candidate is verified exactly and the list certified complete, so all
  backends give the same result. `"normaliz"` needs PyNormaliz, which feynkit
  detects at run time and does not depend on. `faces` and `lattice_coordinates`
  also accept a sequence of points, not only an array.
- Facets of lower-dimensional polytopes and lattice-primitive facet forms.
  `PolytopeData` gains `relative_facets`, the facets relative to the affine hull
  for a polytope of any dimension at least 1, as primitive integer inequalities
  that cut out the polytope together with the equations in the new field
  `affine_hull`; `chart`, a `LatticeChart` x = o + B c in which the points
  generate Z^d affinely, as the new `feynkit.polytope.lattice_chart` returns it;
  `smith_invariants`, the non-zero Smith invariants of the difference matrix;
  and the property `sublattice_index`, their product. `Facet` gains
  `lattice_normal` and `lattice_offset`, the facet in the chart, and
  `lattice_index`, the gcd g_F of b - m . alpha_j over the points, and the
  properties `homogenised_form`, (b, -m), and `lattice_form`, (b, -m) / g_F,
  the facet form primitive with respect to the lattice ZA.
- A `slow` pytest marker, deselected by default like `examples`, and acceptance
  tests against the f-vectors of the principal Landau determinant database of
  Fevola, Mizera and Telen (2024). Three entries are committed under
  `tests/data/pld/` with their attribution and CC BY 4.0 licence; set
  `FEYNKIT_PLD_DATA` to the unpacked database to check all 114.

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
- `faces` and `polytope_data` work in integer arithmetic. The facets come from
  an integer beneath-beyond construction, and the list is certified complete,
  from the face lattice it generates, before anything is derived from it. They
  used to come from Qhull with a tolerance of 1e-7 and were not checked. Qhull,
  and Normaliz when installed, are optional sources of facet candidates, each
  verified exactly; when either fails, or its list fails the certificate,
  beneath-beyond takes over. `polytope_data` is faster on large polytopes:
  about 0.13 s instead of 1 s for the massless planar double box.
- The normalised volume comes from a pulling triangulation of the certified
  face lattice. `AConfiguration.normalized_volume` delegates to it and is
  cached; on large polytopes it is slower than the floating hull was, about
  0.12 s instead of 0.01 s for the massless planar double box.
  `AConfiguration.affine_dim` and `AConfiguration.smith_invariants` are computed
  exactly in pure Python.
- `intrinsic_lattice_model` and `AConfiguration.intrinsic_model` give
  `intrinsic_coords` in the Hermite normal form basis of the lattice L spanned
  by the differences of the points, the basis `lattice_chart` uses, with the
  first point at 0. Where the old coordinates were right they can differ from
  the new ones by a unimodular change of basis, and for the massless triangle
  they do. The new field `IntrinsicModel.basis`, which defaults to (), holds
  the basis, so that point i is base_point + sum_t intrinsic_coords[i][t]
  basis[t]. The field takes part in equality, so a four-field `IntrinsicModel`
  built by hand, or unpickled from an older version, no longer equals a
  computed one whose basis is non-empty. For a single point, or copies of one,
  each coordinate is the empty tuple, where it was the zero vector of length n.
- `faces` lists every point on a face, so every copy of a repeated endpoint of a
  segment is now in its vertex face, where only one was kept, and the two
  vertices of a segment are sorted by index like all other faces. As a result
  `landau_analysis_from_polynomial` can list the two vertex `face_discriminants`
  of a polynomial whose Newton polytope is a segment in the other order, as for
  x + s x^2 y^2 in x and y. Feynman polytopes keep their order, and the
  discriminants and surfaces do not change.

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
- `AConfiguration.normalized_volume` was wrong on configurations that are not
  full-dimensional: it projected onto the wrong singular vectors and divided by
  the product of the Smith invariants where the covolume of the difference
  lattice was needed. It raised a NumPy shape error for (0, 0), (1, 1), (2, 2),
  whose volume is 2; gave 0 for (0, 0), (2, 2) and for (0, 0, 0), (1, 0, 0),
  (0, 1, 1), and 2 for (0, 0, 0), (1, -1, 0), (1, 0, -1), whose volumes are 1;
  and gave 1 for the unit square in a coordinate plane of R^4, whose volume
  is 2. `examples/bms_g_polynomial_analysis.py` therefore prints 1 for the
  lower and upper sub-configurations of BMS_n, which are unimodular simplices,
  where it printed 0 for n = 2 and 2 for n >= 3.
- `AConfiguration.normalized_volume` gave 0 for a segment in Z^1, such as the
  Newton polytope (1), (2) of the massive tadpole `0|:n`, which now gives 1. On
  full-dimensional input with large coordinates its floating hull could be
  wrong: for the pentagon (-177677, 2), (266518, -3), (266516, -3),
  (-177679, 2), (-10^10, 0) it gave 50000000005, where the volume is
  50000000015.
- `faces` and `polytope_data` could return a wrong face lattice for large
  coordinates, where the floating tolerances of Qhull and of the rank
  computation misplaced points. For the pentagon above `faces` found three of
  its five vertices and `polytope_data` raised, and the cube [0, 2]^3 sheared by
  x -> (x_1 + 10^9 x_2, x_2 + 10^9 x_3, x_3) was taken to be 2-dimensional.
- `AConfiguration.affine_dim` used a floating rank and gave 1 for (0, 0, 0),
  (1, 10^17, 0), (2, 2 * 10^17 + 1, 0), whose affine dimension is 2.
  `AConfiguration.smith_invariants` and `lattice_coordinates` took the
  differences of the points in int64, which overflows silently for coordinates
  near 2^62, and returned wrong invariants and coordinates there.
- `intrinsic_lattice_model`, and so `AConfiguration.intrinsic_model`, took as
  its basis the first linearly independent differences of the points and
  truncated each coordinate towards zero, so its coordinates were wrong
  whenever those differences were not a basis of the lattice all the
  differences span: for the points 0, 2, 3 in Z the basis was 2 and the last
  point got the coordinate 1, where it is 3/2. When the points were neither
  full-dimensional nor all equal, as for the Newton polytopes of `01e|e|:zn`
  and `012e|2e|e|:zzzz`, it raised `NonSquareMatrixError`, and so did the
  Newton section of `fk`. Its rank came from a floating-point `matrix_rank`,
  and it took the differences in int64. It is now built on `lattice_chart` in
  integer arithmetic: the coordinates are integers that reproduce every point,
  it works in every dimension, and `intrinsic_rank` and `smith_invariants` are
  exact, the latter equal to `AConfiguration.smith_invariants`.
- The documents said that the holonomic rank equals the normalised volume
  without limiting this to full-dimensional configurations: section 5.3 of the
  mathematics reference, the `AConfiguration` table of the guide
  ("= holonomic rank"), the `feynkit.polytope` module docstring ("whatever the
  ambient dimension") and the `AConfiguration.normalized_volume` docstring.
  Below full dimension the rows of the homogenised A are linearly dependent,
  and for generic beta the system has no non-zero solutions. They now state
  the hypothesis, and section 4.5 states it as full row rank of the
  homogenised A.
- Section 6.3 of the mathematics reference said that a toric operator relates
  integrals with shifted propagator exponents. Since A u = A v, d^u I_A and
  d^v I_A are the same multiple of I_A(beta - A u, z), so a toric operator is a
  differential equation in z, not a reduction between different integrals.
- Section 5.4 of the mathematics reference said that the intrinsic coordinates
  W^-1 (alpha_j - alpha_1) are integers for any r independent rows W of the
  difference matrix. They are integers only when the columns of W form a basis
  of the lattice L the differences span. The section now describes the
  Hermite normal form basis of L that `AConfiguration.intrinsic_model` uses,
  and says that the product of the Smith invariants is the index of L in its
  saturation, which is [Z^n : L] only for a full-dimensional configuration.
- The intrinsic lattice model example in section 16 of the guide raised
  `AttributeError`: it printed `model.basis` and `model.intrinsic_points`, and
  `IntrinsicModel` had neither. It now prints the fields `IntrinsicModel` has,
  for three points on a line in Z^2, and a test runs it and checks its output
  against the guide.
- `examples/bms_g_polynomial_analysis.py` printed that the holonomic rank of
  BMS_n was "confirmed" to be 2^{n-1} for n = 2, 3, 4, 5, although feynkit
  computes only the volume. It now prints whether vol_0(BMS_n) = 2^{n-1} and
  says that for generic beta the holonomic rank is vol_0(BMS_n); its other
  statements about the rank now say "for generic beta" too.

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
