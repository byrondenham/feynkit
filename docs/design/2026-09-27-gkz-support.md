# GKZ support at special kinematics

Design note for building the columns of `FeynmanIntegral.gkz` from the monomials of $G$ at every
kinematic point, as `newton_polytope` already does.

## Purpose

`gkz_exponent_vectors` in `polynomials/spanning_trees.py` lists the exponent vectors of
$G = U + F$ without expanding $G$. It keeps the complement of a spanning two-forest $T_2$ when a
single product $p_j \cdot p_l$, with $j$ and $l$ on different trees, is nonzero. It also keeps
$u_e \prod_{f \notin T} u_f$ for every massive edge $e$ and spanning tree $T$. The coefficient of
a two-forest monomial is a sum, and it can vanish while its terms do not. On shell, a tree
carrying one leg $j$ has $s_{T_2} = p_j^2 = 0$. With $p_1^2 = m_1^2$ on the one-mass triangle,
the mass term cancels $s_{T_2}$.

| Integral | Columns of `gkz` | Monomials of $G$ |
|---|---|---|
| `12e\|3e\|3e\|e\|:zzzz`, $p_i^2 = 0$ | 10 | 6 |
| `12ee\|22e\|e\|:zzzz`, $p_i^2 = 0$ | 9 | 7 |
| `15e\|24\|3e\|4e\|5\|e\|:zzzzzzz`, $p_i^2 = 0$ | 46 | 26 |
| `12e\|2e\|e\|:nzz`, $p_1^2 = m_1^2$ | 7 | 6 |

At generic kinematics nothing cancels and the two agree. With two or three legs on shell every
product vanishes, and the rule gives the right support by accident.

## Effects

For the on-shell massless box, whose Newton polytope has 6 points, f-vector (6, 15, 18, 9) and
normalised volume 3:

- `gkz` has 10 columns, 10 z-variables, and Euler operators over all 10; $\beta$ is unchanged.
- `toric_ideal` has 10 generators, against one.
- `symmetry_pairs` gives 120 pairs, against 72. Only 72 equals the order of
  `polytope_automorphisms`, with which its docstring says the pairs coincide.
- The report's z-table has four entries with coefficient 0, and its summary gives 10 monomials of
  $G$, codimension 5 and 10 toric generators, against 6, 1 and 1. `Schwinger.columns_match`
  (`io/report.py`, line 574) is False, since the Cayley matrix, built from $\tilde U$ and
  $\tilde F$, has 6 columns.
- The printing code of `fk analyse` shows the same for this integral: `-g` prints the Euler
  equations and z-variables of the 10 columns and `-t` the ten generators. `-n` builds
  `AConfiguration` from `gkz.a_matrix` and prints 10 monomials, 10 vertices and volume 11,
  against 6, 6 and 3; `-S` prints the 120 pairs. `fk compare` also builds `AConfiguration` from
  `gkz.a_matrix`. `fk` itself builds every integral from a CNickel string at generic kinematics,
  so on the command line the larger configuration appears only through a database row stored
  from Python.
- The database stores `a_matrix`, `n_rows`, `n_cols` and the toric generators from `gkz`, beside
  the six Newton points and their fingerprint. `find_equivalent` prefilters candidates on
  `(n_rows, n_cols)`, so it compares the wrong shape.
- The `ZEntry` docstring's "zero when the monomial cancels" describes a case that no longer
  arises and should go.

## Design

- Each two-forest monomial $\prod_{e \notin T_2} u_e$ is kept when its coefficient in $F$, up to
  the factor $1/\mu^2$,
  $$c_{T_2} = -s_{T_2} + \sum_{e \in C(T_2)} m_e^2, \qquad
  s_{T_2} = -\sum_{j \in A,\, l \in B} p_j \cdot p_l,$$
  is nonzero. Here $A$ and $B$ are the legs on the two trees and $C(T_2)$ the edges joining them;
  $s_{T_2}$ is the square of the momentum from one tree to the other, by conservation. A product
  is looked up under $(j, l)$ or $(l, j)$, as now.
- The mass loop adds only $u_e^2 \prod_{f \notin T} u_f$ for $e \notin T$, with coefficient
  $m_e^2$. The monomials with $e \in T$ are the two-forest monomials of $T \setminus \{e\}$,
  decided above.
- "Nonzero" uses the same zero test as the support of $G$ in `extract_monomial_support`: the
  coefficient after `sp.expand` is compared with 0. The test is complete for polynomial
  coefficients. For rational functions both can keep a monomial whose coefficient is 0, as for
  $m_1^2 = s/x - s/(x+1)$ against $p_1^2 = s/(x(x+1))$, but they agree with each other.
- A mass `Float(0.0)` still gives a mismatch, since `Float(0.0) != 0` in SymPy: the mass loop
  adds $u_e^2$ monomials that the expansion of $G$ drops. Normalising float zeros is left to a
  separate change.
- Filtering `gkz` by the support of $G$ inside `integral.py` would also work, but it touches a
  core module and makes `gkz` expand $G$.

The rule matches the support of $G$ in 64 cases of graph and kinematics on 25 CNickel strings
up to the planar double box, and at 300 seeded random special points (vanishing invariants,
integer values, $p_i^2 = m_e^2$, numerical and shared masses), where the current rule disagrees
in 66; an independent implementation agrees. With the new rule on a copy of 0.4.0, the full test
suite passes, the on-shell box gives a $5 \times 6$ matrix, one toric generator, 72 symmetry
pairs, the report numbers 6, 1 and 1 with `columns_match` True, and `fk analyse -n` volume 3. The
cost of `gkz` is unchanged within measurement: 0.01 s, 0.13 s and 0.10 s, against 0.02 s, 0.15 s
and 0.09 s before, for the massless and massive planar double boxes and the pentagon-box
`145|26|3e|4e|e|6e|e|:zzzzzzzz`.

## Database

- `_migrate` repairs old rows once. When `PRAGMA user_version` is 0, every row whose `n_cols`
  differs from its number of Newton points gets `a_matrix` rebuilt from those points in the
  graded order of `_create_gkz_system_direct` (descending degree, then descending exponents),
  under a row of ones, with `n_rows` and `n_cols` to match, and `toric_gens`, `n_toric_gens` and
  `is_binomial` set to NULL; then `user_version` becomes 1. The whole repair is one transaction.
  Repairs are numbered steps, step $k$ taking `user_version` from $k$ to $k + 1$, so that a later
  repair is added as the next step.
  The fingerprint, the automorphism data and `equivalences` depend on the points alone and need
  no repair. The comparison runs in Python, since feynkit supports SQLite builds without JSON
  functions.
- The upserts of `store` and `_store_toric` also refresh `a_matrix`, `n_rows` and `n_cols`.
- `_lookup_toric` treats a row as uncached when its `n_cols` differs from its number of points,
  or when a cached generator names a variable beyond $z_{n_\mathrm{cols}}$. Both cases arise when
  0.4.0 or earlier writes to a repaired file: a new row with the larger matrix, or, on a repaired
  row whose generators are NULL, ten generators in $z_1, \ldots, z_{10}$ over its six columns
  (checked).
- 0.3.0 and 0.4.0 read a repaired row inconsistently: a toric ideal indexed for 6 columns sits
  on their own 10-column A, or they recompute from NULL. Old releases cannot be fixed; the
  changelog says so.

## Release

The changelog entry, under Breaking changes:

> `FeynmanIntegral.gkz` now has one column per monomial of G. Earlier versions kept monomials
> whose coefficients cancel at special kinematics, such as on-shell legs or p_1^2 = m_1^2, so the
> A-matrix, the toric ideal, the symmetry pairs and the report's z-table, monomial count and
> codimension described a larger configuration than the Newton polytope: for the massless box
> with p_i^2 = 0, 10 columns and 10 toric generators instead of 6 and 1. Integrals whose
> coefficients do not cancel, among them every integral at generic kinematics and so every
> integral `fk` builds, are unchanged. Database files are repaired when first opened; releases up
> to 0.4.0 read repaired rows inconsistently.

## Testing

The suite has no test on this path. New tests:

- The columns of `gkz.a_matrix` are the Newton points for a list of graphs with on-shell legs,
  equal masses and generic kinematics, and at seeded random special points.
- The on-shell box: a $5 \times 6$ A-matrix; the report gives 6 monomials of $G$, codimension 1
  and one toric generator; `columns_match` is True; `_print_newton` gives volume 3.
- `symmetry_pairs` has as many elements as `polytope_automorphisms` has, 72 for the on-shell box.
- Database: a stale row written with the 0.4.0 statements is repaired on open, and only once;
  a row carrying generators beyond $z_{n_\mathrm{cols}}$ is ignored by `_lookup_toric`, then
  rewritten.
- At default kinematics the A-matrix is unchanged.

## Changes to core modules

- `database.py`: the repair in `_migrate` and `PRAGMA user_version`; the guard in
  `_lookup_toric`; the refreshed columns in both upserts.
- `integral.py`, `polytope.py`, `a_configuration.py`, `landau.py`: none.

Elsewhere: the rule in `polynomials/spanning_trees.py`, the `ZEntry` docstring in
`io/report.py`, and the changelog.

## Decisions

- The fix is announced under Breaking changes in the changelog's Unreleased section, as the
  correction of $\beta$ was in 0.3.0.
- The repair leaves the toric generators of the rows it fixes NULL, to be recomputed on demand,
  which keeps the first open fast. It is the step from `user_version` 0 to 1.
