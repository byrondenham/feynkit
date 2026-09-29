# Changelog

## Unreleased

### Breaking changes

- `FeynmanIntegral.gkz` now has one column per monomial of G. Earlier versions
  kept monomials whose coefficients cancel at special kinematics, such as
  on-shell legs or p_1^2 = m_1^2, so the A-matrix, the toric ideal, the symmetry
  pairs and the report's z-table, monomial count, codimension and Schwinger
  column check described a larger configuration than the Newton polytope: for
  the massless box with p_i^2 = 0, 10 columns and 10 toric generators instead of
  6 and 1. Integrals whose coefficients do not cancel, among them every integral
  at generic kinematics and so every integral `fk` builds, are unchanged.
  Database files are repaired when first opened for writing, and toric
  generators that releases up to 0.4.0 store in them afterwards for the larger
  configuration are recomputed rather than read. Those releases read repaired
  rows inconsistently.
- `fk compare` exits with status 3 when none of its checks finds an
  equivalence, including when the ambient dimensions differ; it exited with 0
  whatever the verdict. Status 0 means that some check found a map, not that a
  GKZ identity follows: a `unimodular` or `affine_polytope` map alone gives 0.
  Status 1 still means an error. A script that runs `fk compare` under
  `set -e`, or treats any non-zero status as a failure, must now allow 3, and
  a Python caller of `feynkit.cli.main` now gets `SystemExit(3)`.
- The bare form needs a `|` in its first argument, which every CNickel string
  that fk prints or documents has; a first argument without one is read as a
  command.
  `fk 0:n` and `fk -- 0:n`, which analysed the massive tadpole in 0.4.0, and
  `fk 0:n 00:nn`, which compared two diagrams, now exit with status 2 as usage
  errors. So do `fk e` and `fk abc`, which exited with 1 after reading their
  argument as a CNickel string. A first argument such as `0:n`, a CNickel
  string without a `|`, gets a hint to name the command, as in
  `fk analyse "0:n"`.
- `Graph.cnickel()` minimises over the names of shared masses as well as over
  the vertex labellings. It names them `a`, `b`, `c` in order of first
  appearance and writes a mass that one propagator alone carries as `n`,
  whatever its symbol: `12e|2e|e|:aab` gives `12e|2e|e|:aan`, `:sss` gives
  `:aaa`, `:bbn` gives `:aan` and `:nan` gives `:nnn`. Strings with only `z`
  and `n` do not change. The change reaches `FeynmanIntegral.cnickel`, the
  analysis report, `fk analyse` and `fk compare`. A graph with more than 23
  classes of equal masses, which the letters cannot name, raises
  `NotImplementedError`. Database rows keep the string they were stored with,
  since a store never overwrites the `cnickel` column; lookups go by the
  fingerprint of the Newton polytope and still find them, but a stored string
  can differ from what `cnickel()` now returns for the same graph.
- `Graph.from_cnickel` rejects the mass codes `1` to `9`. The code `1` gave the
  symbol `m_1`, which `n` also gives propagator 1, so two different masses
  could become one. `0` still means massless. `cnickel()` never wrote digits.
- `Graph.from_cnickel` strips only the terminating `|` of the topology. A
  superfluous one, as in `12e|2e|e||`, describes a vertex without propagators
  or legs and raises `ValueError`, where it was ignored. So does any vertex
  without propagators or legs, wherever its entry sits, when there are two or
  more vertices; the message names the vertex. `e||e|` gave a graph with an
  isolated vertex.

### Added

- `fk analyze` is an alias of `fk analyse`.
- `feynkit.generate`, with `generate_graphs` (also exported from `feynkit`) and
  `mass_colourings`. `generate_graphs` yields the connected, bridgeless graphs
  with given loops, legs and propagators in which every vertex has degree at
  least 3, counting a self-loop twice and a leg once, as canonical CNickel
  strings: each graph once, with every mass colouring up to its automorphisms.
  Colourings are massless or with a mass of their own (`"zn"`, the default),
  massless only (`"z"`) or with equal masses as well (`"shared"`). Options allow
  self-loops, several legs at a vertex and only one-vertex-irreducible graphs.
  One loop with 2 to 6 legs gives 34 graphs, two loops with 2 to 4 legs 675.
  The counts are checked against an independent enumeration, closed forms and
  OEIS A000029.
- `generate_graphs` checks its arguments when it is called, not at the first
  string: loops, legs or propagators that are not integers or are out of
  range, a `max_legs_per_vertex` other than a positive integer or None, an
  unknown alphabet or a block of more than 10 vertices raise `ValidationError`.
- `nickel_index` and `cnickel()` accept graphs with 10 vertices, the most that
  one-digit labels allow; the limit was 9.
- Property-based tests over relabellings of the generated graphs, with
  `hypothesis` as a development dependency. `HYPOTHESIS_PROFILE=feynkit-dev`
  runs a larger random search than the default, derandomised profile. Running
  the tests needs the development dependencies (`uv sync`), since
  `tests/conftest.py` imports `hypothesis`.
- `FeynmanIntegral.is_scaleless`: whether G satisfies Lee's criterion of a
  zero sector (R. N. Lee, arXiv:1310.1145, section 3), that is whether the
  origin lies outside the affine hull of the Newton polytope. Dimensional
  regularisation sets such an integral to zero. A scaleless integral's Newton
  polytope is not full-dimensional, but not every such integral is scaleless:
  `1ee|1|:zn`, whose massless line carries no momentum, is not.

### Changed

- `nickel_index`, `cnickel()` and `compute_graph_automorphisms` find the
  canonical labelling by an exact branch and bound over breadth-first
  labellings instead of a scan of all V! labellings, and give the same strings
  and the same automorphisms, in lexicographic order. Graphs with 9 or 10
  vertices take milliseconds unless their automorphism group is large: the
  9-gon takes about a millisecond instead of about 6 s. The Notes of
  `Graph.nickel_index` give the slow cases, such as the complete graph with 10
  vertices, which takes minutes.
- Below full dimension `polytope_automorphisms` is the group of the Newton
  polytope as a lattice polytope in its affine hull: the affine maps of the
  affine hull that preserve its integer points and the polytope. `maps` holds
  one extension (U, t) of each to Z^n, whose linear part U fixes a complement
  of the direction space of the affine hull, and `order` counts them; the
  unimodular maps of Z^n that take the polytope to itself form an infinite
  group there. The witnesses of the equivalence tests are such extensions. In
  full dimension nothing changes.

### Fixed

- `maximal_pairing_matrix` did not return the lexicographic maximum under row
  and column permutations: it placed one column per row, never branched on
  rows that tie, and kept only the last of the columns that tie. It returned a
  smaller matrix for most small random matrices, and for the A-matrix of the
  massless triangle a matrix that `is_canonical` rejected. It now searches
  row by row and keeps every arrangement that ties, so `PM_max` is the maximum
  and changes for most inputs. `row_permutation` and `col_permutation` now give
  `PM_max`, with `PM_max[i, j] == PM[row_permutation[i], col_permutation[j]]`;
  the row permutation used to drop a swap. `is_canonical(PM)` holds exactly
  when PM is the maximum of its orbit, and so for every `PM_max`. The search
  costs more the more arrangements tie, as for an identity matrix.
  `examples/dissertation_overview.py` and
  `examples/dissertation_overview_enhanced.py` print the new maximum of the
  triangle, which is canonical, where the second reported a failed check.
- `fk` reported an unknown option or a surplus argument with its top-level
  usage; it now shows the usage of `fk analyse` or `fk compare`. As a first
  argument without a `|` is read as a command, `fk analyze "12e|2e|e|"` no
  longer runs `fk compare` on `analyze` and `12e|2e|e|`, and a misspelt
  command such as `fk anlyse` gets argparse's invalid-choice message.
  `fk --help` shows the bare form `fk A B` beside `fk CNICKEL`. An option of
  the subcommand given before it, as in `fk --no-db analyse "12e|2e|e|"`, is
  reported as one that must follow the command.
- `fk analyse --latex FILE` and `--text FILE` found that the file could not be
  written only after the analysis, which can take minutes. A file in a missing
  or read-only directory, or a directory given as the file, now stops `fk`
  before the analysis, with the same one-line error and exit status 1.
- `fk analyse -S` printed `|Aut(P)|  (polytope automorphisms)` as a key wider
  than the column of the others, which pushed its value out of line. The key
  is now `|Aut(P)|`, and the note follows the number.
- `fk compare` reported `finite_index` as YES, with det = 0, when the search
  found only a singular map, which gives no identity. It now reports `no` with
  the note `only a singular map found, det = 0`, and prints no witness map.
- `Graph.from_cnickel` parses strings with an empty last entry, which
  `cnickel()` returns for the sunrise with one leg (`111e||:zzz`), the vacuum
  sunrise (`111||`) and three chains of two propagators (`123|4e|4e|4e||`), and
  which raised `ValueError`. `Graph.from_cnickel(s).cnickel() == s` now holds
  for the canonical string of every graph in which each vertex has a
  propagator or a leg.
- Section 1.4 of the mathematics reference listed the digits 1 to 9 as shared
  mass labels, and `docs/automorphism_groups.md` said that
  `compute_graph_automorphisms` tries all V! vertex permutations; both now
  describe what the code does. The `nickel_index` docstring gave the box as
  `13e|2e|3e|e|`, which is not canonical; it is `12e|3e|3e|e|`.
- `polytope_automorphisms`, `symmetry_pairs` and the equivalence tests took the
  vertices and edges of a Newton polytope from a floating convex hull, which
  could take points that are not vertices for vertices and diagonals of facets
  for edges. The groups then came out too small, and depended on the order of
  the points: `112|3|4e|5e|5e|e|:nnnnnzz`, `:nnnnzzz` and `:zznnnzz` got 4, 12
  and 12 automorphisms instead of 24, 24 and 72, and a relabelled support or a
  relabelled massive double box was not unimodularly equivalent to the original.
  `hull_vertex_indices` and `vertex_edge_graph` now take the vertices and edges
  from the certified face lattice of `polytope_data`, and `vertex_permutations`
  and `vertex_orbits` index `polytope_data(points).vertices`, the list the
  report numbers v_1, v_2, .... The massive tadpole's segment gets its
  reflection, where `polytope_automorphisms` and `symmetry_pairs` raised
  `ValueError`. The exact labels separate the vertices of some polytopes less
  well than the floating ones did: `symmetry_pairs` on the columns of A of
  `112|3|4e|5e|5e|e|:nnnnzzz` takes about 15 s, where it took under a second.
- Below full dimension `polytope_automorphisms` found only the identity,
  `is_unimodular_equivalent` found a Newton polytope not equivalent even to
  itself, and `is_affinely_equivalent` and `is_point_config_equivalent`
  returned witnesses of arbitrary determinant, singular for three points
  compared with themselves. They now work in the lattice chart of the vertices,
  or of all the points, where the polytope is full-dimensional:
  `012e|2e|e|:znnn`, the massive triangle with a massless self-loop, gets the 6
  automorphisms of the triangle and is unimodularly equivalent to
  `12e|12e|e|:nnzn`, and two segments are unimodularly equivalent exactly when
  their lattice lengths agree. A single point repeated, compared with another,
  gets the translation between them, with M the identity.
- `is_affinely_equivalent` read SymPy rationals as their integer parts, so a
  triangle with rational vertices was not equivalent to the unit triangle. It
  now scales each configuration by the least common denominator of its
  coordinates and scales the witness back, and `is_point_config_equivalent`
  does the same below full dimension, so both accept rational points in every
  dimension. `is_unimodular_equivalent` and `compute_polytope_automorphisms`
  raise `ValidationError` on a coordinate that is not an integer: they
  truncated SymPy rationals, and raised `ValueError` on other floats.
- The basis searches of `compute_polytope_automorphisms`,
  `is_unimodular_equivalent` and `symmetry_pairs` tested linear independence
  with a floating rank, which fails for large coordinates. For the triangle
  with a point on an edge (0, 0), (1, 0), (2, 0), (k, 1), whose group has
  order 2, `polytope_automorphisms` found the identity alone from k = 8 * 10^7,
  and `symmetry_pairs` found no pairs or recursed without end. They now use an
  exact integer rank.
- `polytope_automorphisms`, `symmetry_pairs` and `is_unimodular_equivalent`
  computed each candidate map W_b W_a^-1 in floating point, rounded it and
  filtered it by a floating determinant, so the images of polytopes under
  unimodular maps with large entries lost automorphisms. Under
  [[1, k], [k, k^2 + 1]], in GL_2(Z), the unit triangle kept 2 of its 6
  automorphisms and symmetry pairs at k = 1000, and at k = 10^4
  `symmetry_pairs` found none, the identity included. Each candidate is now
  found in integer arithmetic, as W_b adj(W_a) / det(W_a), and kept only when
  it is integral; the products, and those of the Liu-Cai labels, use Python
  integers where int64 could overflow. The searches are also faster:
  `polytope_automorphisms` of the massless pentagon takes about 60 s instead
  of about 140 s, and `symmetry_pairs` on the columns of A of
  `112|3|4e|5e|5e|e|:nnnnzzz` about 15 s instead of about 35 s.
- `symmetry_pairs` returned no pairs, not even the identity, for a
  configuration that is not full-dimensional. It now finds them in the
  lattice chart of the points and extends each to Z^n as
  `polytope_automorphisms` does: 6 for `012e|2e|e|:znnn`, as for the massive
  triangle.

## 0.4.0 (2026-09-27)

### Breaking changes

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
- `feynkit.io.report.SECTION_NAMES` gains `torus`, the finite-field point
  counts (see Added), which take seconds for five propagators; seven exceed
  the default budget. Code that passes `SECTION_NAMES` to build every section
  now counts points too, and raises `ValidationError` where the count cannot
  run, as for an integral with kinematic constraints or seven propagators. The
  new `DEFAULT_SECTIONS`, every section but `torus`, is what `from_integral`,
  `to_latex`, `to_text` and `fk analyse --sections` use by default.
- `faces`, `polytope_data`, `lattice_coordinates` and `intrinsic_lattice_model`
  raise `ValidationError` on a non-integer coordinate, which `faces` used as a
  float and the others truncated, and on points with different numbers of
  coordinates, which raised `ValueError` or `TypeError`. Integral floats such as
  `2.0` are still accepted. `intrinsic_lattice_model` also raises
  `ValidationError` when there are no points, where it raised `IndexError`.
- An `AConfiguration` without points raises `ValidationError` from
  `normalized_volume`, where it returned 1. `AConfiguration.normalized_volume`,
  `faces` and `polytope_data` raise `ComputationError` if a consistency check of
  the exact computation fails, which would be a bug; the volume used to be 0
  whenever Qhull failed.
- `PolytopeData` gains four fields and `Facet` three (see Added). The new
  `PolytopeData` fields have no defaults, so code that builds a `PolytopeData`
  by hand must pass them. The new `Facet` fields have defaults but take part in
  equality, so a `Facet(normal, offset, point_indices)` built by hand no longer
  equals the facet `polytope_data` returns.

### Added

- Finite-field point counts of G = 0 in the torus, in the new module
  `feynkit.point_count`. `count_torus_points` counts the points of V = {G = 0}
  in (F_p^*)^N for finitely many primes p at one rational kinematic point, drawn
  with a seed or given, and fits a polynomial P(q) exactly. The draw prefers a
  point where the square-test factors, the irreducible factors of the principal
  discriminants of the faces of dimension 1 or more, are rational squares.
  When the fit passes the tests below, it reports P(q), the candidate Euler
  characteristic chi(X) = -P(1) of the complement X of V in the torus, and the
  candidate master count C = (-1)^N chi(X) of Bitoun, Bogner, Klausen and
  Panzer (Corollary 37). These are candidates, not proofs: Katz's theorem needs
  every finite field of all but finitely many characteristics. The result is a
  frozen `TorusCount`; its `max_prime` is the largest prime the count could
  use, up to which `excluded_primes` lists the primes left out.
  `backend="flint"` counts with python-flint, which feynkit does not depend on,
  instead of numpy: a cross-check 75 to 180 times slower.
- The fit is tested in this order: integer coefficients; no q^N term;
  0 <= C <= N! Vol(Newt G), with Vol the Euclidean volume of the Newton
  polytope Newt G of G at the point (Bitoun et al., Theorem 44, after
  Kouchnirenko), computed as `normalized_volume * sublattice_index`, or 0 when
  the polytope is not full-dimensional; a guard; and a check at further
  primes. The guard fails when the Newton polytope has an edge of lattice
  length 3 or more, or a face of dimension 2 or more whose lattice quotient has
  exponent (largest Smith invariant) 3 or more, since the counts may then
  depend on characters of order above 2. The check takes `verification`
  primes, 4 by default, and more until every non-trivial product of the
  quadratic characters of -1 and of the non-zero values of the vertex
  coefficients, the face-discriminant factors, written in the squared masses,
  and the edge discriminants has taken both signs; the constants of the
  discriminants of faces of dimension 2 or more, and the discriminants of
  skipped faces, are not covered. The divisors of the edge discriminants are
  among the primes left out, and a factor f odd in a mass m enters through its
  norm f(m) f(-m). A fit that fails a test gives no candidate, and `reason`
  names the test. Massive graphs and off-shell legs often give no candidate at
  the drawn point, since many factors must be squares at once: the off-shell
  massless box `12e|3e|3e|e|:zzzz` gives none at seeds 0 to 149, and 11 at
  p_i^2 = (-12, -6, 5, -4), s12 = 20, s23 = -41, where its square-test factors
  are squares.
- `fk analyse --torus-count` prints the point counts and the candidates, or why
  the counts give no candidate. Runs without section flags leave it out.
  `--seed N`, 0 or more, and `--torus-budget N`, 1 or more, set the seed of the
  kinematic point and the most evaluations of G (2 * 10^9 by default), for
  `--torus-count` or a report whose `--sections` names `torus`, which is how
  `--json` gets the counts; the JSON summary then gains
  `candidate_master_count`, which is null when the counts give no candidate. A
  count over the budget stops with an error naming `--torus-budget`. Given
  `--torus-count` and a report, the count runs once if the report has `torus`,
  and otherwise uses the report's Landau analysis if it has `landau`.
- The analysis report gains the section `torus`, "Candidate Euler characteristic
  from point counts", after the Newton polytope, and a "Candidate master count"
  row in its summary. With it the Newton polytope section points to the
  candidate, or says that the counts give no candidate, instead of saying that
  the Euler characteristic is not computed. `AnalysisReport` gains the field
  `torus`, and `from_integral` the arguments `torus_seed` and `torus_budget`;
  the `torus` and `landau` sections share one Landau analysis.
- `FeynmanIntegral.torus_count` substitutes `on_shell` into the momentum
  products and counts the points of G. The values of `on_shell` are exact
  expressions without floats and contain none of its keys, and a symbol in them
  that shares its name with one of the integral's symbols must be that symbol.
  String values are parsed by SymPy, so "p1^2" means p1**2.
- `critical_point_count`, also in `feynkit.point_count`, counts the critical
  points of sum_e nu_e log u_e - (D/2) log G on X with Singular, which number
  |chi(X)| for generic exponents (Fevola, Mizera and Telen 2024, proof of
  Theorem 3.1, after Huh 2013). It counts modulo two primes near 2^31. A count
  modulo a prime can differ from the count over Q, so it is a cross-check, not
  a certificate, and requiring the two primes to agree guards against an
  unlucky prime. It raises `ComputationError` after `timeout` seconds, 300 by
  default. feynkit does not depend on Singular.
- `feynkit.polytope.normalized_volume(points)`, the exact normalised volume, and
  a keyword `backend` of `faces`, `polytope_data` and `normalized_volume` that
  chooses where facet candidates come from: `"python"` (integer beneath-beyond,
  the reference, which the default `"auto"` uses), `"qhull"` or `"normaliz"`.
  Every candidate is verified exactly, and when Qhull or Normaliz fails, or its
  list fails the completeness certificate, beneath-beyond takes over, so all
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
  `FEYNKIT_PLD_DATA` to the unpacked database to check all 114, three of them
  only with the `slow` marker.
- Acceptance tests of the candidate master counts. At seeds 0 and 1 the
  massless, one-mass and two-mass bubbles, the massless triangle, the massless
  box with p_i^2 = 0 and the massless kite give the master counts of Fevola,
  Mizera and Telen (2024, Tables 1 and 2) and Bitoun et al. (Example 53); the
  one-mass bubble, which neither lists, gives the C = 2 derived in section 4.7
  of the mathematics reference. The off-shell massless box gives 11 at the
  point above. The massless triangle at p_i^2 = (1, 4, 9), where the Källén
  function vanishes, raises unless `allow_singular` is set, and then gives 3.
  The two-mass bubble, the massless triangle and the three-mass sunrise give no
  candidate at a point where a square-test factor is not a square, and there
  `critical_point_count` gives their generic master counts, 3, 4 and 7. It
  also gives the candidate at the seed-0 point of each of the first six graphs.
  The committed entry `A4_zero_generic` of the database, and `A4_zero_zero`
  when `FEYNKIT_PLD_DATA` is set, give the Euler characteristic the database
  lists for generic kinematics. With the `slow` marker as well, every entry
  with at most five variables is counted at seed 0: the eight that give a
  candidate there must give that Euler characteristic after a check at four
  primes or more, and any other candidate must equal it.

### Changed

- `faces` and `polytope_data` work in integer arithmetic. The facets come from
  an integer beneath-beyond construction, and the list is certified complete,
  from the face lattice it generates, before anything is derived from it. They
  used to come from Qhull with a tolerance of 1e-7 and were not checked; Qhull
  and Normaliz are now optional sources of candidates (see Added).
  `polytope_data` is faster on large polytopes: about 0.13 s instead of 1 s for
  the massless planar double box.
- The normalised volume comes from a pulling triangulation of the certified
  face lattice. `AConfiguration.normalized_volume` delegates to it and is
  cached; on large polytopes it is slower than the floating hull was, about
  0.12 s instead of 0.01 s for the massless planar double box.
- `faces` lists every point on a face, so every copy of a repeated endpoint of a
  segment is now in its vertex face, where only one was kept, and the two
  vertices of a segment are sorted by index like all other faces. As a result
  `landau_analysis_from_polynomial` can list the two vertex `face_discriminants`
  of a polynomial whose Newton polytope is a segment in the other order, as for
  x + s x^2 y^2 in x and y. Feynman polytopes keep their order, and the
  discriminants and surfaces do not change.

### Fixed

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
  near 2^62, and returned wrong invariants and coordinates there. All three
  are now computed exactly in pure Python.
- `intrinsic_lattice_model`, and so `AConfiguration.intrinsic_model`, took as
  its basis the first linearly independent differences of the points and
  truncated each coordinate towards zero, so its coordinates were wrong whenever
  those differences were not a basis of the lattice all the differences span:
  for the points 0, 2, 3 in Z the basis was 2 and the last point got the
  coordinate 1, where it is 3/2. When the points were neither full-dimensional
  nor all equal, as for the Newton polytopes of `01e|e|:zn` and
  `012e|2e|e|:zzzz`, it raised `NonSquareMatrixError`; `fk` 0.3.0 therefore left
  the lattice base point out of its Newton section, and now prints it. Its rank
  came from a floating-point `matrix_rank`, and it took the differences in
  int64. It is now built on `lattice_chart` in integer arithmetic (see Breaking
  changes): the coordinates are integers that reproduce every point, it works
  in every dimension, and `intrinsic_rank` and `smith_invariants` are exact,
  the latter equal to `AConfiguration.smith_invariants`.
- Section 4.5 of the mathematics reference and section 4.3 of the literature
  review called |chi|, the number of master integrals, the Lee-Pomeransky
  count. Lee and Pomeransky count the critical points of G itself; the master
  count is due to Bitoun et al. (2019), and both documents now attribute it to
  them. Both, the Newton polytope section of the analysis report, the docstring
  of `compute_toric_ideal_generators`, `docs/automorphism_groups.md` and two
  example scripts also bounded it by the normalised volume without a
  hypothesis. The bound is N! Vol, with Vol the Euclidean volume, which equals
  the normalised volume only when the exponent differences span Z^N: for
  1 + x^2, |chi| = 2 and the normalised volume is 1. The README and the guide
  gave the title of Lee and Pomeransky (2013) as "Critical points and master
  integrals"; it is "Critical points and number of master integrals".
- The documents said that the holonomic rank equals the normalised volume
  without limiting this to full-dimensional configurations: section 5.3 of the
  mathematics reference, the `AConfiguration` table of the guide ("= holonomic
  rank"), the `feynkit.polytope` module docstring ("whatever the ambient
  dimension"), the `AConfiguration.normalized_volume` docstring, two passages of
  `docs/automorphism_groups.md` and the statement of Klausen's Theorem 2.2 and
  section 4.3 of the literature review. Below full dimension the rows of the
  homogenised A are linearly dependent, and for generic beta the system has no
  non-zero solutions. They now state the hypothesis, and section 4.5 states it
  as full row rank of the homogenised A.
- `examples/bms_g_polynomial_analysis.py` printed that the holonomic rank of
  BMS_n was "confirmed" to be 2^{n-1} for n = 2, 3, 4, 5, although feynkit
  computes only the volume. It now prints whether the normalised volume
  vol_0(BMS_n) is 2^{n-1} and says that for generic beta the holonomic rank is
  vol_0(BMS_n); its other statements about the rank now say "for generic beta"
  too.
- The intrinsic lattice model example in section 16 of the guide raised
  `AttributeError`: it printed `model.basis` and `model.intrinsic_points`, and
  `IntrinsicModel` had neither. It now prints the fields `IntrinsicModel` has,
  for three points on a line in Z^2, and a test runs it and checks its output
  against the guide.
- Section 5.4 of the mathematics reference said that the intrinsic coordinates
  W^-1 (alpha_j - alpha_1) are integers for any r independent rows W of the
  difference matrix. They are integers only when the columns of W form a basis
  of the lattice L the differences span. The section now describes the
  Hermite normal form basis of L that `AConfiguration.intrinsic_model` uses,
  and says that the product of the Smith invariants is the index of L in its
  saturation, which is [Z^n : L] only for a full-dimensional configuration.
- Section 6.3 of the mathematics reference said that a toric operator relates
  integrals with shifted propagator exponents. Since A u = A v, d^u I_A and
  d^v I_A are the same multiple of I_A(beta - A u, z), so a toric operator is a
  differential equation in z, not a reduction between different integrals.

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
