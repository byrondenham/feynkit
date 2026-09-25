# Exact polyhedral core

Design note for replacing the floating-point hull computations behind `faces`,
`polytope_data` and `AConfiguration.normalized_volume` with exact integer ones.

## Purpose

The Landau analysis and the report both read the Newton polytope. The Landau analysis takes one
discriminant per face, and the principal A-determinant is a product over every face
(Gelfand, Kapranov and Zelevinsky 1994, chapter 10), so a missing or misassigned face means a
wrong factor, not a rounding error. The report prints the facets as convergence inequalities and
prints the volume. Results used by the Landau analysis and the report must not depend on floating
tolerances. This note specifies:

- facets computed and certified complete in integer arithmetic, and the face lattice built from
  them;
- an exact normalised volume that raises on failure instead of returning 0;
- facet forms primitive with respect to the lattice $\mathbb{Z}A$ as well as the ambient lattice;
- facets for lower-dimensional polytopes.

## Current state

- `faces(pts)` projects onto the affine hull with a floating SVD and runs scipy's `ConvexHull`.
  A point lies on a facet when $|w \cdot x + c| < 10^{-7}$ for Qhull's unit-normalised equation
  $w \cdot x + c = 0$. The face lattice comes from repeated pairwise intersection of these index
  sets, and face dimensions from a floating rank with tolerance $10^{-9}$. Nothing checks that the
  facet list is complete or that the index sets are the true tight sets.
  `landau_analysis_from_polynomial` iterates over this list, and `polytope_data` wraps it.
- `polytope_data` computes each facet normal exactly (SymPy nullspace, made primitive, oriented by
  the centroid) from the index set `faces` returned, so the normal is right only if the set is.
  Completeness is not checked. `facets` is empty for a lower-dimensional polytope. The volume
  comes from `_normalized_volume`, which rewrites a lower-dimensional polytope in lattice
  coordinates and calls `AConfiguration.normalized_volume`.
- `AConfiguration.normalized_volume` returns $d!\,\mathrm{Vol}(P)$, for $P$ of affine
  dimension $d$, divided by the product of the Smith invariants, with $\mathrm{Vol}$ from a
  floating hull, rounded to an integer. Any exception from the hull returns 0. For
  lower-dimensional input it has two faults. It projects with `u_mat[:, :n]`, the left singular
  vectors, which are indexed by points, where `vt[:n].T` is meant. The product
  `deltas @ u_mat[:, :n]` is then a shape error unless the number of points equals the ambient
  dimension, and meaningless when it does. Replacing it with `vt` is not enough: an orthonormal
  chart measures Euclidean volume, and the divisor must then be the covolume
  $\sqrt{\det(B^T B)}$ of the difference lattice $L$ with basis $B$. The product of the Smith
  invariants is only the index of $L$ in its saturation $\mathrm{sat}(L) = \mathrm{span}(L) \cap
  \mathbb{Z}^n$, and the two agree only when $\mathrm{sat}(L)$ has covolume 1, for instance when
  $P$ is full-dimensional.

| Points | Correct | `AConfiguration` now |
|---|---|---|
| $(0,0), (1,1), (2,2)$ | 2 | NumPy shape error |
| $(0,0), (2,2)$ | 1 | 0 |
| $(0,0,0), (1,-1,0), (1,0,-1)$ | 1 | 2 |
| $(0,0,0), (1,0,0), (0,1,1)$ | 1 | 0 |
| unit square in a coordinate plane of $\mathbb{R}^4$ | 2 | 1 |

With `vt` in place of `u_mat`, rows 1 and 2 still give 0, because Qhull cannot build a
one-dimensional hull; measuring the length instead would give $2\sqrt{2} \approx 2.83$ on row 1,
which rounds to 3. Row 3 gives $2! \cdot \sqrt{3}/2 = \sqrt{3}$, which rounds to 2.
`polytope_data` gets all five right because it passes lattice coordinates, which is why the report
has not been affected.

## Design

Notation: $P = \mathrm{conv}(\alpha_1, \ldots, \alpha_N) \subset \mathbb{R}^n$ with
$\alpha_j \in \mathbb{Z}^n$, affine dimension $d$, and $L$ the lattice spanned by the differences
$\alpha_j - \alpha_1$. A face is stored as its tight set, the indices of all points on it, as a
Python integer bitmask. All arithmetic is on Python integers; NumPy arrays are converted on entry.

### Coordinates

A full-dimensional $P$ is handled in ambient coordinates. A lower-dimensional one is handled in
the lattice chart $x = o + Bc$ that `lattice_coordinates` already builds: the columns of $B$ are
the Hermite-normal-form basis of $L$, $o \in \mathbb{Z}^n$, and the coordinates
$c_j \in \mathbb{Z}^d$ are non-negative. In the chart $P$ is full-dimensional and its points
affinely generate $\mathbb{Z}^d$. The hull code below therefore only ever sees full-dimensional
input. Faces are index sets and need no mapping back; facets are mapped back as described under
lower-dimensional polytopes.

`_exact.py` provides a pure-Python Hermite normal form with the unimodular transform recorded,
and Smith invariants. The chart, the affine hull, the lift of facet forms and the Smith
invariants all use it. It follows SymPy's normal-form convention, so `lattice_coordinates`
returns what it returns now.

### Facets

Assume $d \ge 2$; smaller dimensions are handled in closed form under Faces. A candidate facet is
verified against a set of points, all points or, inside beneath-beyond, the points inserted so
far:

1. Choose $d$ affinely independent points among those the generator associates with it, by exact
   rank. If there are none, reject it.
2. The normal $m$ is the vector of signed maximal minors of the $(d-1) \times d$ matrix of their
   differences (Bareiss elimination), divided by the gcd of its entries. Put $b = m \cdot \alpha$
   for any of the chosen points.
3. If $m \cdot \alpha_j - b$ takes both signs over the points, reject the candidate. Otherwise
   orient it so that $m \cdot \alpha_j \le b$ for every $j$.
4. The tight set is $\{j : m \cdot \alpha_j = b\}$. It has affine dimension $d - 1$ by step 1.
5. Discard duplicates by $(m, b)$.

An accepted candidate is a facet of the hull of those points: a valid inequality whose equality
set has dimension $d - 1$. What remains to be shown is that no facet is missing.

The reference generator is beneath-beyond over the integers, in pure Python. It starts from the
first $d + 1$ affinely independent points in index order, found greedily by exact rank, and their
simplex, then inserts the remaining points in index order, so the run is reproducible. For a point
$p$ and every current facet $F$ let $s_F = m_F \cdot p - b_F$.

- If no $s_F$ is positive, $p$ lies in the current hull; add it to the tight sets of the facets
  with $s_F = 0$.
- Otherwise keep the facets with $s_F < 0$, extend those with $s_F = 0$ by adding $p$ to their
  tight sets, and drop those with $s_F > 0$. For every ridge $R = T_F \cap T_G$ of affine
  dimension $d - 2$ with $s_F > 0$ and $s_G < 0$, add the facet through $R$ and $p$, verified
  against the inserted points. Its tight set is exactly $T_R \cup \{p\}$. Its hyperplane $H$
  supports the old hull in a face containing $R$; were that face a facet, it would be $F$ or $G$,
  the two facets through $R$, but $p$ lies in $H$ and in neither of their hyperplanes. So $H$
  meets the old hull in $R$.

This is the beneath-beyond theorem. Ridges are found here by pairwise intersection, which is
correct because the facet list of the current hull is complete at every step; the remark under
Certificate explains why the same shortcut fails as a certificate. A ridge needs at least $d - 1$
common points, which is checked before any rank is computed.

Qhull, through scipy, is an opt-in candidate generator. Each Qhull facet contributes the points
within $10^{-7}$ of its equation together with its simplex, and step 1 picks the $d$ points from
those. Qhull triangulates non-simplicial facets and may return flat simplices, so the simplex
vertices alone will not do. Rejected candidates are dropped. Beneath-beyond takes over when Qhull
raises or the certificate fails.

Qhull is not the default. Feynman polytopes have few facets with many points on each, and there
beneath-beyond is faster than Qhull plus verification. With hundreds of facets the order reverses,
so Qhull remains available for large general configurations.

PyNormaliz, when installed, is a third generator (see Backends).

### Certificate

Remark. Pairwise ridge closure, the condition that every $(d-2)$-dimensional intersection of two
candidate tight sets lies in exactly two candidates, is not a certificate, because an incomplete
list passes it. Such an intersection is a ridge of $P$, and a ridge lies in exactly two facets of
$P$, so for verified, distinct candidates the condition always holds. Two edges of a triangle pass
it. So does the square pyramid without one of its triangular facets; the intersections of the
remaining four facets have f-vector $(3, 5, 4, 1)$ with alternating sum 1, so that list satisfies
Euler-Poincaré as well. The ridges of a candidate whose neighbour is missing are not intersections
of two candidates, so the condition never examines them. The certificate below supplies the
ridges of every candidate by induction on dimension.

Let $\mathcal{C}$ be the verified candidates and $\mathcal{L}$ the set of non-empty intersections
of their tight sets, together with $P$, each with its exact affine dimension. The proof uses three
facts:

- (F1) every member of $\mathcal{L}$ is a face of $P$, being an intersection of faces, and a face
  of $P$ contained in a face $Q$ is a face of $Q$;
- (F2) a face is determined by its tight set, since it is the convex hull of the points on it, so
  containment and equality of faces can be read off the sets;
- (F3) the members of $\mathcal{L}$ of dimension $d - 1$ are exactly the candidates, since two
  distinct facets meet in dimension at most $d - 2$.

For $Q \in \mathcal{L}$ of dimension $k \ge 1$ let
$\Phi(Q) = \{Q \cap F : F \in \mathcal{C},\ \dim(Q \cap F) = k - 1\}$, so $\Phi(P) = \mathcal{C}$
by (F3). By (F1) every member of $\Phi(Q)$ is a facet of $Q$. Accept $\mathcal{C}$ if and only if:

- (C1) $\Phi(Q)$ has exactly two members for every $Q$ of dimension 1;
- (C2) $\Phi(Q)$ is non-empty for every $Q$ of dimension at least 2;
- (C3) for every $Q$ of dimension at least 2, every $F \in \Phi(Q)$ and every $R \in \Phi(F)$,
  exactly two members of $\Phi(Q)$ contain $R$.

(C3) is the diamond property along the chains $R < F < Q$, and (C1) is its case with the empty
face at the bottom.

Soundness. Claim: for every $Q \in \mathcal{L}$ of dimension $k \ge 1$, $\Phi(Q)$ is the set of
all facets of $Q$. For $k = 1$ this is (C1), since a segment has two vertices. For $k \ge 2$,
(C2) gives a facet $F \in \Phi(Q)$. By induction $\Phi(F)$ holds every facet of $F$, that is every
ridge of $Q$ in $F$, and by (C3) each lies in a second member of $\Phi(Q)$. So $\Phi(Q)$ is a
non-empty set of facets of $Q$ closed under crossing ridges. The facet-ridge graph of a polytope
is connected, so $\Phi(Q)$ contains every facet of $Q$. At $Q = P$ the candidate list is complete.
Conversely a complete list always passes, since the face lattice of a polytope is graded and has
the diamond property.

Building the sets $\Phi(Q)$ takes about $|\mathcal{L}| \cdot |\mathcal{C}|$ intersections, each
of whose dimensions is looked up in $\mathcal{L}$; checking (C3) along chains avoids testing all
pairs of faces two dimensions apart. The check needs no geometry beyond $\mathcal{L}$, which
`faces` builds anyway.

Euler-Poincaré holds for every complete list but also for some incomplete ones, such as the
pyramid above, so it is a test, not the certificate. A failed certificate on the `"python"` path
is a bug and raises `ComputationError` naming the condition and the faces involved.

### Faces

`faces` returns $\mathcal{L}$ as a list of (dimension, sorted indices), sorted by dimension and
then indices, as now. $\mathcal{L}$ is built by intersecting the frontier with the candidates
only, since every face is an intersection of facets. Repeated points lie on the same faces. Small
dimensions are closed-form:

- no points: `faces([])` returns `[]`, as now; `polytope_data` and `normalized_volume` raise
  `ValidationError`;
- $d = 0$: one face of dimension 0 holding every index;
- $d = 1$: in the chart coordinate $c$, which runs from 0 to $c_{\max}$, the two vertex faces are
  the points with $c = 0$ and with $c = c_{\max}$, and the relative facets are $-c \le 0$ and
  $c \le c_{\max}$, mapped back like any other.

### Volume

The volume comes from the pulling triangulation, computed on the certified lattice with no
further hull work. Order the vertices by their ambient coordinates, lexicographically, so the
triangulation does not depend on input order. For a face $Q$ let $v(Q)$ be its least vertex.
The simplices of a vertex are that vertex alone. For $\dim Q \ge 1$ the simplices of $Q$ are
$v(Q)$ joined to each simplex of each facet $F \in \Phi(Q)$ with $v(Q) \notin F$. This is a
triangulation of $Q$ by vertices of $Q$. Simplices are memoised per face, and there are at most
as many as the normalised volume.

The normalised volume is $\sum_\sigma |\det(\sigma_1 - \sigma_0, \ldots, \sigma_d - \sigma_0)|$
over the simplices of $P$, divided by the sublattice index. For a full-dimensional $P$ the
determinants are taken in ambient coordinates and the sublattice index is $[\mathbb{Z}^n : L]$,
the product of the Smith invariants of the difference matrix. Each simplex has edges in $L$, so
its determinant is a multiple of that index. A remainder, or a zero determinant, raises
`ComputationError`. For a lower-dimensional $P$ the determinants are taken in the chart, whose
points generate $\mathbb{Z}^d$, so there is nothing to divide by. A point has volume 1, as now.

`AConfiguration.normalized_volume` delegates to the same function on its affine points. The
function raises `ValidationError` on empty or non-integer input and never returns 0; every
non-empty configuration has positive volume.

### Lattice-primitive facet forms

Let $P$ be full-dimensional with facet $F$ given by $m \cdot x \le b$, $m$ primitive in
$\mathbb{Z}^n$. The homogenised form is
$$l(y_0, y) = b\,y_0 - m \cdot y,$$
so that $l(1, \alpha_j) = b - m \cdot \alpha_j \ge 0$ with equality exactly on $F$. Its
coefficient vector $(b, -m)$ is primitive in $\mathbb{Z}^{n+1}$ because $m$ is.

The columns $a_j = (1, \alpha_j)$ of the homogenised $A$ generate a lattice
$\mathbb{Z}A \subseteq \mathbb{Z}^{n+1}$. The $\mathbb{Z}A$-primitive form of $F$ is the generator
of the forms that vanish on the columns of $F$ and take integer values on $\mathbb{Z}A$, chosen
non-negative on $A$. It is
$$l_F = l / g_F, \qquad g_F = \gcd_j\, l(1, \alpha_j),$$
and $g_F$ is called the lattice index of the facet. Proof: the forms vanishing on the columns of
$F$ are the real multiples $t\,l$, since those columns span a hyperplane of $\mathbb{R}^{n+1}$. As
the $a_j$ generate $\mathbb{Z}A$, $t\,l$ is integral on $\mathbb{Z}A$ exactly when
$t\,l(a_j) \in \mathbb{Z}$ for all $j$. Each $l(a_j)$ is a multiple of $g_F$ and $g_F$ is an
integer combination of them, so this holds exactly when $t\,g_F \in \mathbb{Z}$. The admissible
multiples are therefore $(1/g_F)\,\mathbb{Z}\,l$, generated by $l / g_F$. Its values on the
columns are non-negative integers with gcd 1, zero exactly on $F$. Since some point lies on $F$,
$g_F$ is also the gcd of $m \cdot \lambda$ over $\lambda \in L$, the index of $m(L)$ in
$\mathbb{Z}$.

When the two differ. Subtracting the first column of the homogenised $A$ from the others and
clearing the first column by row operations shows that its Smith invariants are 1 followed by
those of the difference matrix. So $\mathbb{Z}A = \mathbb{Z}^{n+1}$ exactly when
$L = \mathbb{Z}^n$, exactly when every Smith invariant is 1. Then
$m(L) = m(\mathbb{Z}^n) = \mathbb{Z}$ and $g_F = 1$ for every facet, and the two normalisations
coincide. A Smith invariant above 1 is necessary for a difference but not sufficient, and the
difference is per facet:

- $(0,0), (2,0), (0,1)$: $L$ has index 2; $g_F = 1$ for $y \ge 0$ and $g_F = 2$ for $x \ge 0$ and
  for $x + 2y \le 2$.
- $(0,0), (4,0), (2,2), (2,1)$: $L = 2\mathbb{Z} \times \mathbb{Z}$ has index 2, yet $g_F = 1$
  for all three facets.

In the lattice chart of a lower-dimensional polytope the points generate $\mathbb{Z}^d$ affinely,
so $g_F = 1$ there and the chart form is already $\mathbb{Z}A$-primitive. The convergence
inequalities in the report are unchanged by positive scaling and keep using $(m, b)$.

### Lower-dimensional polytopes

Facets are computed in the chart as $m' \cdot c \le b'$ with $m'$ primitive in $\mathbb{Z}^d$, by
the full-dimensional code. They are mapped back as follows.

- Affine hull. The integer forms on $\mathbb{Z}^n$ that vanish on $L$ form a saturated lattice of
  rank $n - d$. For a basis $E$ of it, the rows $h = (-E_i \cdot \alpha_1, E_i)$ are forms on
  homogenised coordinates with $h(1, x) = 0$ on $\mathrm{aff}(P)$, in the same constant-first
  order as the homogenised facet form. $E$ comes from the Hermite normal form of $B^T$ with its
  transform, and is itself put into Hermite normal form.
- Facet forms. An ambient form $\mu$ restricts to $B^T \mu$ on the chart. The image of
  $\mathbb{Z}^n \to \mathbb{Z}^d$, $\mu \mapsto B^T \mu$, has index $[\mathrm{sat}(L) : L]$, the
  sublattice index, so $m'$ need not lift to an integer form. Take the least $t \ge 1$ with
  $t\,m' \in B^T \mathbb{Z}^n$ (read off the Hermite normal form of $B^T$), an integer $\mu$ with
  $B^T \mu = t\,m'$, and $\beta = \mu \cdot o + t\,b'$. Then $\mu \cdot x \le \beta$ holds on $P$
  with equality exactly on the facet, and together with the affine-hull equations these
  inequalities cut out $P$.
- $\mu$ is primitive: if $\mu = h\,\mu''$ with $h > 1$, then $B^T \mu'' = (t/h)\,m'$, and
  $t/h$ is an integer because $m'$ is primitive, contradicting the choice of $t$. $\mu$ is unique
  modulo the forms vanishing on $L$ and is fixed by reducing it against the Hermite normal form
  of $E$. So for lower-dimensional $P$ the homogenised form, and with it `lattice_form`, is
  defined modulo the affine-hull forms; the reduction picks one representative.
- $t$ is the lattice index $g_F$ of the lifted form, since its values on $L$ are
  $t\,m'(\mathbb{Z}^d) = t\,\mathbb{Z}$. So in every dimension the ambient form is $g_F$ times the
  $\mathbb{Z}A$-primitive one. For a full-dimensional $P$ the same construction reproduces
  $(m, b)$ with $t = g_F$, which is a test.

### Backends

`backend` follows `compute_toric_ideal_generators` and the Landau elimination:

- `"python"`: beneath-beyond, the reference path. Always available.
- `"qhull"`: Qhull candidates, verified and certified, with beneath-beyond as the fallback.
- `"normaliz"`: support hyperplanes from PyNormaliz, given the coordinates the hull code works in
  (ambient for a full-dimensional $P$, the lattice chart otherwise); only the points on each
  hyperplane are used. Raises `ComputationError` if `_pynormaliz_available()` is false.
  Falls back to `"python"` if the import or any call into PyNormaliz fails, or the certificate
  fails. The face lattice, certificate and volume are the shared code, so the volume never
  depends on Normaliz's normalisation.
- `"auto"` (default): `"python"`. It will prefer `"normaliz"` only once Normaliz has been timed
  against the pure-Python path.
- Anything else raises `ComputationError`, as in `compute_toric_ideal_generators`.

The helper `_pynormaliz_available()` checks `importlib.util.find_spec("PyNormaliz")`, the
analogue of `_4ti2_binary()` and `_singular_binary()`. The core never needs PyNormaliz, and CI
does not install it. Every backend's facets pass through the same verification and certificate,
so all backends return equal `PolytopeData`.

## Interfaces

In `feynkit/polytope.py`. The integer routines (Bareiss determinant and rank, hyperplane through
$d$ points, Hermite normal form with transform, Smith invariants, beneath-beyond, lattice
closure, certificate, pulling simplices) go in a private module `feynkit/_exact.py`.

```python
def faces(
    pts: np.ndarray | Sequence[Sequence[int]], *, backend: str = "auto"
) -> list[tuple[int, tuple[int, ...]]]
def polytope_data(points: Sequence[Sequence[int]], *, backend: str = "auto") -> PolytopeData
def normalized_volume(
    points: Sequence[Sequence[int]] | np.ndarray, *, backend: str = "auto"
) -> int
def lattice_chart(points: Sequence[Sequence[int]] | np.ndarray) -> LatticeChart
def lattice_coordinates(pts: np.ndarray) -> list[tuple[int, ...]]  # unchanged


@dataclass(frozen=True)
class LatticeChart:
    origin: tuple[int, ...]                     # o in Z^n
    basis: tuple[tuple[int, ...], ...]          # the d columns of B, each of length n
    coordinates: tuple[tuple[int, ...], ...]    # c_j >= 0 with alpha_j = o + B c_j

    def to_ambient(self, c: Sequence[int]) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class Facet:
    normal: tuple[int, ...]                     # m (or mu), primitive in Z^n
    offset: int                                 # b (or beta)
    point_indices: tuple[int, ...]
    lattice_normal: tuple[int, ...] | None = field(default=None, kw_only=True)  # m'
    lattice_offset: int | None = field(default=None, kw_only=True)             # b'
    lattice_index: int = field(default=1, kw_only=True)                        # g_F

    @property
    def homogenised_form(self) -> tuple[int, ...]: ...   # (b, -m_1, ..., -m_n)
    @property
    def lattice_form(self) -> tuple[Fraction, ...]: ...  # homogenised_form / g_F
```

`lattice_coordinates` returns `list(lattice_chart(pts).coordinates)`. `lattice_form` is the
$\mathbb{Z}A$-primitive form; for lower-dimensional $P$ it is defined modulo the affine-hull
forms. `polytope_data` always fills the three new `Facet` fields. The defaults keep
`Facet(normal, offset, point_indices)` working: the chart fields are then `None` and
`lattice_form` equals `homogenised_form`, as for a configuration whose points generate the
ambient lattice. The chart of a full-dimensional $P$ is computed too, so
every facet has `lattice_normal` and `lattice_offset`: the $\mathbb{Z}A$-primitive form in chart
coordinates, where it is integral. The invariants are $B^T m = g_F\,m'$ and
$b - m \cdot o = g_F\,b'$.

New fields on `PolytopeData`, after the existing ones:

| Field | Meaning |
|---|---|
| `relative_facets` | Facets relative to the affine hull, for every $P$ of dimension at least 1; equal to `facets` when $P$ is full-dimensional |
| `affine_hull` | Rows $(h_0, h_1, \ldots, h_n)$ with $h_0 + h_1 x_1 + \cdots + h_n x_n = 0$ on $P$; empty when full-dimensional |
| `chart` | The `LatticeChart` |
| `smith_invariants` | All non-zero Smith invariants of the difference matrix, computed from $B^T$, whose rows span the same lattice |

and a property `sublattice_index`, the product of `smith_invariants`, which is
$[\mathrm{sat}(L) : L]$. `facets` keeps its meaning, empty unless $P$ is full-dimensional, so the
report's convergence inequalities and the existing tests are unaffected. New names use British
spelling, like `homogenise_polynomial`, except the function `normalized_volume`, which takes the
name of the attribute it computes.

## Testing and acceptance

- All existing tests pass unchanged, including the massless bubble's facet inequalities, the
  one-mass triangle's 27 faces and volume 5, the segment with `facets == ()`, and the BMS family
  with volume $2^{n-1}$.
- Euler-Poincaré: $\sum_{k=0}^{d} (-1)^k f_k = 1$ for every polytope in the samples below.
- Expected values, f-vectors in feynkit's convention with the polytope itself as the final 1.
  Volumes are regression values of the current code. The database entries are named where one
  matches the CNickel string (see below).

| Diagram | CNickel | Database entry | f-vector | Volume |
|---|---|---|---|---|
| Massless triangle | `12e\|2e\|e\|:zzz` | | (6, 12, 8, 1) | 4 |
| Massless box | `12e\|3e\|3e\|e\|:zzzz` | `A4_zero_generic` | (10, 30, 30, 10, 1) | 11 |
| Massive box | `12e\|3e\|3e\|e\|:nnnn` | `A4_generic_generic` | (8, 16, 14, 6, 1) | 15 |
| Massless sunrise | `111e\|e\|:zzz` | | (4, 6, 4, 1) | 1 |
| Massive sunrise | `111e\|e\|:nnn` | | (9, 15, 8, 1) | 10 |
| Massless kite | `12e\|23\|3\|e\|:zzzzz` | | (16, 54, 78, 54, 16, 1) | 42 |
| Massive kite | `12e\|23\|3\|e\|:nnnnn` | `kite_generic_generic` | (24, 66, 73, 39, 10, 1) | 136 |
| `par`, massless | `12ee\|22e\|e\|:zzzz` | `par_zero_generic` | (9, 24, 24, 9, 1) | 8 |
| `par`, massive | `12ee\|22e\|e\|:nnnn` | `par_generic_generic` | (15, 33, 27, 9, 1) | 35 |
| Planar double box | `15e\|24\|3e\|4e\|5\|e\|:zzzzzzz` | `dbox_zero_generic` | (46, 282, 636, 706, 421, 133, 20, 1) | 903 |
| Pentagon-box | `145\|26\|3e\|4e\|e\|6e\|e\|:zzzzzzzz` | `pentb_zero_generic` | (68, 504, 1327, 1778, 1373, 630, 166, 22, 1) | 3148 |

- PLD database. The acceptance source is the database of Fevola, Mizera and Telen, Principal
  Landau determinants, Comput. Phys. Commun. 303 (2024) 109278, arXiv:2311.16219, published on
  MathRepo (https://mathrepo.mis.mpg.de/PLD/, `PLD_database.zip`, content under CC BY 4.0). The
  MathRepo page states that `f_vector` is that of the Newton polytope of the graph polynomial
  $G = U + F$, listing $f_0, \ldots, f_{d-1}$ without the polytope itself. Each entry file gives
  $U$, $F$ and the variables in Julia syntax. For each file the test asserts
  `polytope_data(exponents of U + F).f_vector == tuple(f_vector) + (1,)`. It reads the unpacked
  database from the environment variable `FEYNKIT_PLD_DATA` and skips when that is unset. The
  current code passes on all 114 files, so for the exact code this is a regression test. A small
  fixture of a few entry files may be committed under `tests/data/pld/` with the attribution and
  licence. The three 9-dimensional entries (`dpent`, `npl-dpent`, `npl-dpent2`) and the pentb
  cross-check are marked with a new `slow` marker, registered in `pyproject.toml` and deselected by
  default alongside `examples`.
- The CNickel rows of the table that name a database entry are checked against the table
  always and, when `FEYNKIT_PLD_DATA` is set, also against the entry: the exponents of $U + F$
  must equal feynkit's Newton polytope points up to a permutation of the variables. feynkit's
  default kinematics correspond to the `*_zero_generic` entries for massless strings; the
  all-massive strings in the table match the `*_generic_generic` entries. The database's
  `chi_generic` is an Euler characteristic, not the normalised volume (for `dbox_zero_generic` it
  is 75 against a volume of 903), and is not used.
- Certificate: incomplete lists are rejected: two edges of a triangle, a cube missing one facet, a
  cube missing four, and the square pyramid without a triangular facet, which satisfies
  Euler-Poincaré.
- Exact against floating. The current algorithm moves into the test module as a floating
  reference. On every CNickel string in the test suite and 200 seeded random lattice
  configurations of dimension 2 to 5, some embedded in higher dimension by integer maps, the two
  must give the same faces, facets $(m, b)$ and volume wherever the floating code runs without
  error. `AConfiguration`'s floating volume is compared only on full-dimensional input, because it
  is wrong on lower-dimensional input; the five configurations in Current state become regression
  tests with their exact values.
- Lattice forms: for every facet the values of `lattice_form` on the columns are non-negative
  integers with gcd 1, zero exactly on `point_indices`; `lattice_index` is 1 on every facet when
  the Smith invariants are trivial; the two examples above give $(1, 2, 2)$ and $(1, 1, 1)$; the
  chart invariants hold.
- Lower-dimensional polytopes: $(1,0,0), (0,1,0), (0,0,1), (1,1,-1)$ span the plane
  $x + y + z = 1$, with four relative facets, affine hull $x + y + z = 1$ and volume 2;
  $(0,0,0), (2,0,0), (0,2,0)$ has chart $(0,0), (1,0), (0,1)$ and lifts $c_1 \ge 0$ to
  $-x \le 0$ with $g_F = 2$; every lifted inequality holds on $P$ with equality exactly on its
  facet.
- Volume: invariant under integer unimodular maps and translations; the ambient determinant sum
  over the pulling triangulation equals the sublattice index times the volume; for
  full-dimensional $P$ the volume times the sublattice index over $d!$ matches the floating
  Euclidean volume to $10^{-9}$ relative; `normalized_volume([])` raises `ValidationError`.
- Backends: `"python"` and `"qhull"` give equal `PolytopeData` on the samples. A test that
  removes one facet from the Qhull candidates checks that the certificate rejects the list and the
  fallback returns the right answer. `"normaliz"` is compared in the same way, in tests that skip
  when PyNormaliz is absent, as the Singular and 4ti2 tests do.

## Performance

Targets: on the massless and massive box, the kite and the sunrise, exact `polytope_data` is no
slower than the floating version and takes under 20 ms; the planar double box takes under a
second. These are checked by hand before release, not by timed tests.

Cost: the lattice closure and the certificate take about $|\mathcal{L}| \cdot |\mathcal{C}|$ set
operations on bitmasks and dominate once faces number in the thousands. Beneath-beyond compares
visible and invisible facets in pairs, which is quadratic in the facets of the current hull; the
common-point filter skips the rank computation for pairs sharing fewer than $d - 1$ points. The
pulling triangulation has at most as many simplices as the volume, one $d \times d$ determinant
each. The Landau analysis keeps the same faces, so its cost is still set by the eliminations.

## Migration of callers

- `landau.py`: no change. `_faces(exps)` keeps its signature and, for Feynman polytopes, its
  output order, so the order of `face_discriminants` and the report's grouping by dimension are
  unchanged there. For a polynomial whose Newton polytope is a segment the two vertex faces are
  now sorted by index and can come out in the other order. A certificate failure raises instead
  of silently misassigning points.
- `io/report.py`: no change. `polytope_data` is called as now; `_representations` still gates the
  convergence inequalities on `is_full_dimensional` and uses `normal` and `offset`.
  `_affine_dimension` can switch to the shared exact rank.
- `a_configuration.py`: `normalized_volume` becomes a call to `polytope.normalized_volume` on
  `affine_points`; `affine_dim` uses exact rank; the `try`/`except` returning 0 goes;
  `_compute_smith_invariants` moves to `_exact.py`. The docstring of `normalized_volume` describes
  the triangulation, the lattice it measures in and the exceptions, in place of the Euclidean
  formula. `polytope.py` stops importing `AConfiguration`, which removes the import cycle. The two
  uses in `cli.py` now see the exact volume where they saw 0, and an exception only if a
  consistency check fails.
- `polytope.py`: `_facet`, `_normalized_volume` and `_affine_rank` are removed. The module
  docstring describes the certificate, the relative facets and the lattice forms, and no longer
  says that lower-dimensional polytopes have no facet description.
- `examples/bms_g_polynomial_analysis.py`: its lower and upper sub-configurations are
  $(n-1)$-simplices, unimodular in their difference lattices. Their volumes print 0 at $n = 2$ and
  2 for $n \ge 3$, and print 1 after the change.
- Docs: section 5.3 of the mathematics reference describes the triangulation and the
  lower-dimensional case, and a new subsection covers the certificate and the lattice forms; the
  `PolytopeData` table in the guide gains the new fields; the changelog records the new fields,
  and that `AConfiguration.normalized_volume` now raises rather than returning 0 and is correct on
  lower-dimensional input.

## Out of scope

`normal_forms/_invariants.py` keeps its floating hulls: `hull_vertex_indices` and
`vertex_edge_graph`, with tolerance $10^{-9}$, feed the automorphisms, the report's symmetry
section and `AConfiguration.newton_polytope_points`. Moving them onto the certified lattice, which
already holds the vertices and the edge graph, is a follow-up. The visualisation keeps floating
hulls, since it only draws.

## Open questions

- Normaliz: confirm the coordinate order and sign convention of its support hyperplanes on
  lattice-coordinate input, and time it against the pure-Python path before `"auto"` prefers it.
  The cross-check tests settle the conventions once PyNormaliz is installed.
- Whether `"auto"` should choose `"qhull"` over `"python"` for large inputs, and by what measure.
  The facet count that decides it is not known in advance, while point count and dimension are.
