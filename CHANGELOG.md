# Changelog

## Unreleased

### Added

- `FeynmanIntegral.schwinger_gkz`: the two-block Cayley GKZ system of the
  Schwinger representation (Jimenez-Santacruz, Lopez-Arcos, Quintero Velez
  2026; Klausen 2023, section 3.4), with its toric ideal and the face
  subsystem on the F block. `SchwingerParametrisation.get_A_matrix` now
  delegates to it.

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

## 0.2.0 (2026-09-19)

### Breaking changes

- Python 3.10 or newer is required; 3.9 reached end of life in October 2025.
- The `method` keyword has been removed from `is_affinely_equivalent`,
  `is_point_config_equivalent` and `FeynmanIntegral.is_affinely_equivalent_to`.
  The `"sage"` option it accepted was a placeholder that ran the SymPy search anyway.

### Added

- `one_loop_landau_surfaces` and `one_loop_principal_a_determinant`: the
  one-loop closed form of Dlapa, Helmer, Papathanasiou and Tellander (2023)
  from the principal minors of the modified Cayley matrix, used to check the
  face computation.
- Singular is used for the elimination ideals of non-simplex faces when the
  `Singular` binary is installed; SymPy is the fallback.
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
