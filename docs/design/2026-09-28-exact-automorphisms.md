# Exact polytope automorphisms and scaleless integrals

Design note for a property `FeynmanIntegral.is_scaleless`, and for computing the automorphisms,
symmetry pairs and equivalences of Newton polytopes from the exact polyhedral core, in every
dimension.

## Purpose

This note specifies:

- a property `FeynmanIntegral.is_scaleless` for Lee's criterion of a zero sector, shown in the
  report summary, in `fk analyse` and in its JSON output;
- automorphisms, symmetry pairs and equivalence tests that take their vertices and edges from
  `polytope_data` instead of floating hulls, which give wrong groups for ordinary two-loop
  graphs;
- the same computations for a Newton polytope that is not full-dimensional, done in its lattice
  chart, so that the report and the CLI show its symmetries;
- a database repair step that drops the results the old code cached.

`finite_index_map` below full dimension uses the same chart and is left to a later change.

## Current state

### The criterion

The report and the CLI test whether the Newton polytope $P \subset \mathbb{R}^n$ of $G = U + F$ is
full-dimensional:

- `io/report.py`: `_symmetries_omitted(data)` returns `"not full-dimensional"` when
  `not data.is_full_dimensional`, and `"dimension below 2"` below dimension 2. Both renderers
  then print one sentence instead of the symmetries, and replace the rank paragraph of the Newton
  polytope section with "Since P is not full-dimensional, the rows of A are linearly dependent.
  For generic beta the Euler equations are then inconsistent, and the GKZ system has no non-zero
  solutions: its holonomic rank is 0, not the normalised volume."
- `cli.py`: `_print_newton` calls the volume the holonomic rank only in full dimension;
  `_print_symmetries` prints "Not computed for a Newton polytope that is not full-dimensional"
  or "... of dimension below 2" instead of `|Aut(P)|`; `fk compare` prints n/a for its
  `unimodular` and `affine_polytope` checks in both cases, and for `finite_index` catches the
  `NonSquareMatrixError` that `finite_index_map` raises below full dimension.
- No property names the case. Test comments call such integrals scaleless, which is right for
  most of them but not all (see Mathematics).

### Floating hulls in full dimension

`_invariants.hull_vertex_indices` runs Qhull in floating point on the points, projected to their
affine hull when needed. It can return points that are not vertices. The basis search then finds
only the automorphisms that also permute those points. For three two-loop graphs, built with
`FeynmanIntegral.from_cnickel`:

| CNickel | Vertices, exact / floating | Order now | Order on exact vertices |
|---|---|---|---|
| `112\|3\|4e\|5e\|5e\|e\|:nnnnnzz` | 43 / 46 | 4 | 24 |
| `112\|3\|4e\|5e\|5e\|e\|:nnnnzzz` | 45 / 49 | 12 | 24 |
| `112\|3\|4e\|5e\|5e\|e\|:zznnnzz` | 36 / 38 | 12 | 72 |

The result depends on the order of the points, since Qhull's does. `symmetry_pairs` gives 4, 12
and 12 on the points in the order of `fi.newton_polytope.points`, and 24, 24 and 72 on the
A-matrix columns, which `fi.symmetry_pairs` passes in graded order. The support of
`123e|4e|4e|4e||:nzznnn` has order 4. After a permutation of its coordinates and of its points
the floating hull adds the non-vertex $(1,1,0,0,1,0)$, the order drops to 1, `symmetry_pairs`
gives 1, and `is_unimodular_equivalent` says that the two supports are not equivalent. For the
same reason a property test finds the massive non-planar double box
`123|4e|4e|5e|5|e|:nnnnnnz` not equivalent to a relabelling of itself: the floating hull finds
53 and 55 vertices where both polytopes have 52.

`vertex_permutations` and `vertex_orbits` index the floating vertex list. The report numbers the
vertices $v_k$ from `data.vertex_indices`, so for the three graphs above its orbit labels are
shifted.

`vertex_edge_graph` also counts as edges the diagonals of the non-simplicial facets that Qhull
triangulates. The massive triangle `12e|2e|e|:nnn` has 9 edges and the floating graph 15; for
`12e|3e|3e|e|:nzzz` the counts are 28 and 55. The Liu-Cai labels are then not guaranteed to be
invariant.

### Below full dimension

The basis search needs $n$ linearly independent vertex differences. Below full dimension there
are only $d$, so `_select_basis_indices_by_label` returns `None` and the function returns the
identity alone. The Liu-Cai labels $\det \sum_w (w - v)(w - v)^T$ vanish as well, since the
$n \times n$ matrix has rank at most $d$: for `012e|2e|e|:znnn` every label is 0. On a segment in
$\mathbb{R}^1$ the floating hull raises.

The correct orders below come from a brute-force search. It takes the vertex set from
`polytope_data`, fixes an affine basis, tries every ordered choice of $d + 1$ distinct vertices
as its image, solves for the affine map in rational arithmetic, and keeps the maps that permute
the vertices and preserve the integer points of the affine hull. The configuration group is found
the same way on all the points.

| Diagram | CNickel | $d$ in $\mathbb{R}^n$ | Now | Correct |
|---|---|---|---|---|
| Massive bubble with a massless self-loop | `011e\|e\|:znn` | 2 in $\mathbb{R}^3$ | 1 | 2 |
| Massless triangle with a massless self-loop | `012e\|2e\|e\|:zzzz` | 3 in $\mathbb{R}^4$ | 1 | 48 |
| Massive triangle with a massless self-loop | `012e\|2e\|e\|:znnn` | 3 in $\mathbb{R}^4$ | 1 | 6 |
| One-mass triangle with a massless self-loop | `012e\|2e\|e\|:znzz` | 3 in $\mathbb{R}^4$ | 1 | 6 |
| Massless sunrise at $s = 0$ | `111e\|e\|:zzz` | 2 in $\mathbb{R}^3$ | 1 | 6 |
| Massless triangle at $p_i^2 = 0$ | `12e\|2e\|e\|:zzz` | 2 in $\mathbb{R}^3$ | 1 | 6 |
| Massless box at vanishing invariants | `12e\|3e\|3e\|e\|:zzzz` | 3 in $\mathbb{R}^4$ | 1 | 24 |
| Massless bubble at $s = 0$ | `11e\|e\|:zz` | 1 in $\mathbb{R}^2$ | 1 | 2 |
| Massless self-loop on a massive line | `01e\|e\|:zn` | 1 in $\mathbb{R}^2$ | 1 | 2 |
| Massive tadpole | `0\|:n` | 1 in $\mathbb{R}^1$ | `ValueError` | 2 |
| Massless tadpole | `0\|:z` | 0 in $\mathbb{R}^1$ | 1 | 1 |

The special kinematics are set with `with_(momentum_products=...)`. For the first four rows the
correct order is that of the graph without the self-loop, which the current code gets right in
full dimension: 2 for `11e|e|:nn`, 48 for `12e|2e|e|:zzz`, 6 for `12e|2e|e|:nnn` and 6 for
`12e|2e|e|:nzz`. The configuration group has the same order in every row, and the sublattice
index is 1 in every row.

The equivalence tests fail below full dimension as well:

- `is_unimodular_equivalent` returns `False` for a polytope and itself, and for the segments
  $(0,0), (1,1)$ and $(0,0), (1,-1)$, although
  $U = \bigl(\begin{smallmatrix}1&0\\-2&1\end{smallmatrix}\bigr)$ maps one onto the other. A
  property test finds the scaleless graph `012e|3e|3e|e|:znnnn` not equivalent to itself.
- `is_affinely_equivalent` and `is_point_config_equivalent` give the right verdict, but they solve
  an under-determined system and set its free parameters to 0, so the witness depends on that
  choice. For the support of `012e|2e|e|:znnn` and its images under unimodular maps of
  $\mathbb{Z}^4$ the witnesses have determinants 0, $\pm 1$, 2 and $\pm 3$, and for three points
  compared with themselves the witness is singular.
- `symmetry_pairs` returns `[]`, although the identity is always a pair. For the massive tadpole
  `0|:n`, a segment in $\mathbb{R}^1$, it raises `ValueError` from the floating hull.
- `finite_index_map` raises `NonSquareMatrixError`: it inverts the $n \times d$ matrix of basis
  differences.

### The database

`FeynkitDatabase.store(..., compute_automorphisms=True)` records $|\mathrm{Aut}(P)| = 1$ for
`012e|2e|e|:znnn` and raises `ValueError` for `0|:n`. The automorphism columns are refreshed only
by another store with `compute_automorphisms=True`.

`find_equivalent` and `find_equivalent_record` read the `equivalences` cache first and write both
directions with `INSERT OR IGNORE`, so a cached verdict is never recomputed. Storing
`012e|2e|e|:znnn` and `12e|12e|e|:nnzn`, the same graph with the self-loop on another vertex,
caches `equivalent = 0` in both directions, and `find_equivalent` returns nothing.
`clear_equivalence_cache(relation=...)` deletes the negative verdicts of one relation, but only
when called.

## Mathematics

### Newton polytopes that are not full-dimensional

Write $G = \sum_j z_j u^{\alpha_j}$, with $\beta = (-D/2, -\nu_1, \ldots, -\nu_n)$ and Euler
operators $\hat{E}_r = \sum_j A_{rj} z_j \partial_j$, so that the Euler equations read
$\hat{E}_r \Phi = \beta_r \Phi$. Call $G$ quasi-homogeneous with weights $h \in \mathbb{Z}^n$,
$h \ne 0$, of any signs, and degree $c \in \mathbb{Z}$, possibly 0, when
$G(\lambda^{h_1} u_1, \ldots, \lambda^{h_n} u_n) = \lambda^c G(u)$. The following are
equivalent:

1. $P$ is not full-dimensional;
2. there is a non-zero integer vector $(h_0, h)$ with $h_0 + h \cdot \alpha_j = 0$ for every $j$
   (the rows of `PolytopeData.affine_hull` span these vectors);
3. the rows of the homogenised $A$ are linearly dependent, since $(h_0, h)^T A = 0$;
4. $G$ is quasi-homogeneous, with weights $h$ and degree $-h_0$ (here $h \ne 0$, because $h = 0$
   would force $h_0 = 0$).

Section 4.5 of the mathematics reference states that $A$ has full row rank exactly when $P$ is
full-dimensional, and that the rank is then the normalised volume for very generic $\beta$ (GKZ
1989; Saito, Sturmfels and Takayama 2000). Section 5.3 states the other case: the rows are
dependent, and for generic $\beta$ the system has no non-zero solutions. The reason is one line:
$\sum_r h_r \hat{E}_r = \sum_j (h_0 + h \cdot \alpha_j)\, z_j \partial_j = 0$, so every solution
satisfies $(h_0 \beta_0 + h \cdot \beta')\,\Phi = 0$. Here
$h_0 \beta_0 + h \cdot \beta' = -(h_0 D/2 + h \cdot \nu)$, and it is non-zero exactly when $\beta$
lies outside the column span of $A$, which holds for generic $\beta$.

The same vector gives a rescaling. Substituting $u_i = \lambda^{h_i} v_i$ in
$I = \int_{\mathbb{R}_+^n} u^{\nu - 1} G^{-D/2}\, du$ gives

$$I = \lambda^{h_0 D/2 + h \cdot \nu}\, I \qquad \text{for every } \lambda > 0.$$

For Euclidean kinematics the guide takes the region of absolute convergence from Klausen (2023,
section 3.3), one inequality per facet, and states that the integral converges for no $D$ and
$\nu$ when $P$ is not full-dimensional. The rescaling gives the same for any coefficients. The
integral of the absolute value of the integrand is multiplied by
$\lambda^{\mathrm{Re}(h_0 D/2 + h \cdot \nu)}$; splitting $\mathbb{R}_+^n$ into the orbits
$\lambda \mapsto \lambda^h v$ leaves a factor $\int_0^\infty \lambda^{a - 1}\, d\lambda$, which
diverges for every real $a$. So the integral converges absolutely for no $D$, $\nu$ and
coefficients.

### Scaleless integrals

The paper describing LiteRed (R. N. Lee, arXiv:1310.1145, section 3) calls an integral
scaleless when a linear transformation of the loop momenta multiplies it by a factor other than
1; dimensional regularisation sets such integrals to zero. Its criterion of a zero sector,
stated there as a sufficient condition, is that $\sum_e k_e u_e\, \partial G / \partial u_e = G$,
its eq. (16), has a solution $k$ independent of $u$. That is $k \cdot \alpha_j = 1$ for every $j$, so
$(h_0, h) = (-1, k)$ satisfies item 2. The criterion therefore holds exactly when some equation of
the affine hull has $h_0 \ne 0$, that is when the origin does not lie in the affine hull of the
support of $G$. Then the exponent $h_0 D/2 + h \cdot \nu$ involves $D$, and the rescaling makes
the integral vanish in dimensional regularisation for generic $D$.

The criterion implies that $P$ is not full-dimensional; the converse fails. When every equation
has $h_0 = 0$, the exponent is $h \cdot \nu$ and does not involve $D$. Two examples:

- `1ee|1|:zn`: a massive tadpole joined to the external vertex by a massless line that carries no
  momentum. $u_1$ does not occur in $G$, the only equation is $\alpha_1 = 0$, and
  $\int_0^\infty u_1^{\nu_1 - 1}\, du_1$ factors out.
- `01e|e|:nz` at $s = 0$: the massless line is a bridge carrying $p$ with $p^2 = 0$.

In both, a propagator $1/(0)^{\nu}$ makes the integral ill-defined rather than zero, and
dimensional regularisation does not regulate it. The integral still converges absolutely for no
$D$ and $\nu$, and the GKZ system still has rank 0 for generic $\beta$.

So `is_scaleless` is Lee's criterion. The rank and convergence statements stay keyed on
full-dimensionality. On every example tried the flag matches the physics term:

| Integral | CNickel and kinematics | $d$ in $\mathbb{R}^n$ | Full-dimensional | Scaleless |
|---|---|---|---|---|
| Massless tadpole | `0\|:z` | 0 in $\mathbb{R}^1$ | no | yes |
| Massless self-loop on a massive line | `01e\|e\|:zn` | 1 in $\mathbb{R}^2$ | no | yes |
| Massive times massless tadpole | `00\|:nz` | 1 in $\mathbb{R}^2$ | no | yes |
| Two massless tadpoles | `00\|:zz` | 0 in $\mathbb{R}^2$ | no | yes |
| Massless triangle with a massless self-loop | `012e\|2e\|e\|:zzzz` | 3 in $\mathbb{R}^4$ | no | yes |
| Massive triangle with a massless self-loop | `012e\|2e\|e\|:znnn` | 3 in $\mathbb{R}^4$ | no | yes |
| Massless bubble, $p^2 = 0$ | `11e\|e\|:zz`, $s = 0$ | 1 in $\mathbb{R}^2$ | no | yes |
| Massless sunrise, $p^2 = 0$ | `111e\|e\|:zzz`, $s = 0$ | 2 in $\mathbb{R}^3$ | no | yes |
| Massless triangle, all legs light-like | `12e\|2e\|e\|:zzz`, $p_i^2 = 0$ | 2 in $\mathbb{R}^3$ | no | yes |
| Massless box, all invariants 0 | `12e\|3e\|3e\|e\|:zzzz`, $p_i^2 = s_{12} = s_{23} = 0$ | 3 in $\mathbb{R}^4$ | no | yes |
| Tadpole on a massless line at zero momentum | `1ee\|1\|:zn` | 1 in $\mathbb{R}^2$ | no | no |
| Massive self-loop on a light-like massless bridge | `01e\|e\|:nz`, $s = 0$ | 1 in $\mathbb{R}^2$ | no | no |
| Massive tadpole | `0\|:n` | 1 in $\mathbb{R}^1$ | yes | no |
| Massive self-loop on a massless line | `01e\|e\|:nz` | 2 in $\mathbb{R}^2$ | yes | no |
| Two massive tadpoles | `00\|:nn` | 2 in $\mathbb{R}^2$ | yes | no |
| Massless bubble | `11e\|e\|:zz` | 2 in $\mathbb{R}^2$ | yes | no |
| One-mass bubble, $p^2 = 0$ | `11e\|e\|:nz`, $s = 0$ | 2 in $\mathbb{R}^2$ | yes | no |
| Massive bubble on shell | `11e\|e\|:nn`, $s = m_1^2$ | 2 in $\mathbb{R}^2$ | yes | no |
| One-mass triangle | `12e\|2e\|e\|:zzz`, $p_1^2 = p_2^2 = 0$ | 3 in $\mathbb{R}^3$ | yes | no |
| On-shell massless box | `12e\|3e\|3e\|e\|:zzzz`, $p_i^2 = 0$ | 4 in $\mathbb{R}^4$ | yes | no |

A massless self-loop $e$ makes the integral scaleless because $G = u_e\, G_{\Gamma \setminus e}$,
so $P$ lies in the hyperplane $\alpha_e = 1$.

### Automorphisms of a lattice polytope in its affine hull

Let $S \subset \mathbb{Z}^n$ be a finite set: the vertices of $P$ for `polytope_automorphisms`,
all the points for `symmetry_pairs`. Let $d$ be its affine dimension, $L$ the lattice spanned by
its differences, $x = o + Bc$ its lattice chart with chart points $C = \{c_j\} \subset
\mathbb{Z}^d$, $\mathrm{sat}(L) = \mathbb{R}L \cap \mathbb{Z}^n$, and

$$\Lambda = \{c \in \mathbb{Q}^d : Bc \in \mathrm{sat}(L)\} \supseteq \mathbb{Z}^d, \qquad
[\Lambda : \mathbb{Z}^d] = [\mathrm{sat}(L) : L],$$

the `sublattice_index`. The chart describes the affine lattice
$\mathrm{aff}(S) \cap \mathbb{Z}^n = o + \mathrm{sat}(L)$ as $o + B\Lambda$.

Frame. The Hermite normal form of $B^T$ with its transform, which `polytope._lift_data` already
computes, gives $B^T W = [0 \mid H]$ with $W \in GL_n(\mathbb{Z})$ and $H$ of size $d \times d$.
Put $Q = W^{-T}$. Then $B = Q \bigl(\begin{smallmatrix}0\\H^T\end{smallmatrix}\bigr)$, so the last
$d$ columns $S_Q$ of $Q$ satisfy $B = S_Q H^T$. As columns of a unimodular matrix they are a
basis of $\mathrm{sat}(L)$, the first $n - d$ columns span a complement, and
$\Lambda = H^{-T}\mathbb{Z}^d$.

There are two candidate groups:

$$\mathrm{Aut}_{\mathbb{Z}^n}(S) = \{(U, t) \in GL_n(\mathbb{Z}) \ltimes \mathbb{Z}^n :
US + t = S\}, \qquad \mathrm{Aut}_L(S) = \{(M, s) \in GL_d(\mathbb{Z}) \ltimes \mathbb{Z}^d :
MC + s = C\}.$$

They relate as follows.

- (a) $\mathrm{Aut}_L(S)$ is the group of all affine bijections of $\mathrm{aff}(S)$ that permute
  $S$. Such a map is determined by its permutation, since $C$ spans $\mathbb{R}^d$ affinely, and
  it maps $\mathbb{Z}^d$ onto itself, since the $c_j$ generate $\mathbb{Z}^d$ affinely. It is
  finite and acts faithfully on $S$.
- (b) An element of $\mathrm{Aut}_{\mathbb{Z}^n}(S)$ permutes $S$, so it maps $L$ onto $L$ and
  restricts to an element $\rho(U, t)$ of $\mathrm{Aut}_L(S)$. The kernel $K$ of $\rho$ consists
  of the maps that fix $\mathrm{aff}(S)$ pointwise. It is trivial when $d = n$ and infinite when
  $d < n$ and $n \ge 2$: it contains the shears $x \mapsto x + (e \cdot (x - o))\, v$ for a
  non-zero integer form $e$ vanishing on $L$ and $v \in \mathbb{Z}^n$ with $e \cdot v = 0$. For
  `00|:zz` the swap of $u_1$ and $u_2$ lies in $K$.
- (c) The image of $\rho$ is $\{(M, s) \in \mathrm{Aut}_L(S) : M\Lambda = \Lambda\}$. An integral
  $U$ maps $\mathrm{sat}(L)$ onto itself, which gives one inclusion. Conversely, if
  $M\Lambda = \Lambda$ then $M_S = H^T M H^{-T}$, the matrix of $BMB^{-1}$ on $\mathrm{sat}(L)$
  in the basis $S_Q$, lies in $GL_d(\mathbb{Z})$, and
  $$U = Q \begin{pmatrix} I_{n-d} & 0 \\ 0 & M_S \end{pmatrix} Q^{-1} \in GL_n(\mathbb{Z}),
  \qquad t = \alpha_{\sigma(1)} - U \alpha_1$$
  restricts to $(M, s)$, where $\sigma$ is its permutation.
- (d) Hence $\mathrm{Aut}_{\mathbb{Z}^n}(S) / K$ is the subgroup of $\mathrm{Aut}_L(S)$ that
  preserves $\Lambda$, and the two groups are equal when the sublattice index is 1.

$\mathrm{Aut}(P)$ is the group of automorphisms of $P$ as a lattice polytope in the affine
lattice $\mathrm{aff}(P) \cap \mathbb{Z}^n$: the affine bijections of $\mathrm{aff}(P)$ that
preserve that lattice and $P$. By (c) this is the image of $\rho$ for $S$ the vertex set,
computed in the chart as the part of $\mathrm{Aut}_L(S)$ that preserves $\Lambda$. The reasons:

- For $d = n$ it is the group the code computes now: $K$ is trivial, and preserving
  $\Lambda = B^{-1}\mathbb{Z}^n$ is the requirement that $U$ be integral.
- It depends on $P$ and $\mathbb{Z}^n$ only, not on the embedding. The triangle
  $(0,0), (2,0), (0,1)$, whose differences span a sublattice of index 2, has
  $|\mathrm{Aut}_L| = 6$. In $\mathbb{Z}^2$ its group has order 2, and so has the group of
  $(0,0,0), (2,0,0), (0,1,0)$ in $\mathbb{Z}^3$, where $\mathrm{Aut}_L$ would give 6.
- Its elements act on the ambient support. `maps` holds, for each permutation, the lift in (c),
  which acts as the identity on the complement spanned by the first $n - d$ columns of $Q$. For
  `012e|2e|e|:znnn` the six lifts are the permutations of $u_2, u_3, u_4$.

The lift is a convention. Two lifts differ by an element of $K$, which fixes every point of
$\mathrm{aff}(P)$ and so every column $(1, \alpha_j)$ of $A$. `coefficient_preserving_indices`
evaluates $U\alpha + t$ at support points only, so it does not depend on the lift. The
homogenised maps $T = \bigl(\begin{smallmatrix}1&0\\t&U\end{smallmatrix}\bigr)$ of two lifts
agree on the column span of $A$, but not outside it, so $T\beta$ at the physical
$\beta = (-D/2, -\nu)$ does depend on the lift. That $\beta$ lies outside the column span for
generic $D$ and $\nu$, where the GKZ system has no non-zero solutions.

When the sublattice index is 1, as for every Feynman polytope checked, $\mathrm{Aut}_L$ and
$\mathrm{Aut}(P)$ coincide, so for Feynman integrals the choice is not visible.

Polytope and configuration. `polytope_automorphisms` preserves the vertex set, as it does now.
Section 7.1 of the mathematics reference describes maps that permute every support point, and
step 5 of section 7.3 says this is verified, but the code checks the vertices only.
`symmetry_pairs` does check every column, and the report's coefficient-preserving count requires
every support point to map to a support point. The two groups coincide on every Feynman polytope
checked, twelve full-dimensional and twelve below full dimension, and on the three two-loop graphs
above. They differ in general: the square $[0,2]^2$ with the extra point $(1,0)$ has 8 polytope
automorphisms and 2 symmetry pairs. `polytope_automorphisms` keeps the vertex semantics, and
sections 7.1 and 7.3 of the reference are corrected.

### Equivalences below full dimension

The same frames reduce every ambient question to the charts. For $S, S' \subset \mathbb{Z}^n$ with
frames $(Q, H)$ and $(Q', H')$, the vertex sets for the polytope tests and all the points for the
configuration test:

- Unimodular equivalence. A pair $(U, t) \in GL_n(\mathbb{Z}) \ltimes \mathbb{Z}^n$ with
  $US + t = S'$ exists if and only if $d = d'$ and there is an affine bijection $\varphi$ of
  $\mathbb{Z}^d$ with $\varphi(C) = C'$ and $M\Lambda = \Lambda'$ for its linear part $M$. For
  the converse, $\mathrm{sat}(L)$ and $\mathrm{sat}(L')$ are direct summands of the same rank, so
  $U = Q' \,\mathrm{diag}(I_{n-d}, M_S)\, Q^{-1}$ with $M_S = H'^T M H^{-T}$ is a witness. The
  condition on $\Lambda$ matters: every segment has vertex chart $\{0, 1\}$ and
  $\Lambda = g^{-1}\mathbb{Z}$ for its lattice length $g$, so $(0,0), (1,1)$ and $(0,0), (2,2)$
  are not unimodularly equivalent, while $(0,0), (1,1)$ and $(0,0), (1,-1)$ are. Two points are
  always equivalent, and two segments are when their lattice lengths agree.
- Affine equivalence over $\mathbb{Q}$. The chart map is found as now, but in $\mathbb{Q}^d$,
  where the basis matrix is square and the map unique for a given basis correspondence. All points
  are affinely equivalent, and so are all segments. The lift
  $Q' \,\mathrm{diag}(I_{n-d}, M_S)\, Q^{-1}$ is invertible for every invertible rational $M_S$,
  so the witness is never singular. Its determinant, $\pm \det M_S$, is not intrinsic below full
  dimension: another lift changes it through the complement block.
- Symmetry pairs. As for automorphisms, with $S$ all the points: the chart map permutes the
  columns, and the lift is a pair $(M, t, P)$ with $\det M = \pm 1$.
- Finite-index maps. An integral $x \mapsto Mx + t$ taking every source point onto a target point
  restricts to a chart map $\varphi$ with $\varphi(C) \subseteq C'$. It is integral on
  $\mathbb{Z}^d$ automatically, since it maps differences of source points to differences of
  target points, and it lifts to an integral map exactly when $M_S = H'^T M H^{-T}$ is integral.
  The lift has $|\det| = |\det M_S| = [\mathrm{sat}(L') : \psi(\mathrm{sat}(L))]$, with
  $\psi = B' M B^{-1}$. This is the analogue of $[\mathbb{Z}^n : M\mathbb{Z}^n]$ in full
  dimension, and the only index that does not depend on the lift.

## Design

### `FeynmanIntegral.is_scaleless`

A cached property in `integral.py`:

```python
@cached_property
def is_scaleless(self) -> bool:
    """Whether G satisfies Lee's criterion of a zero sector. ..."""
    points = [tuple(int(x) for x in p) for p in self.newton_polytope.points]
    origin = (0,) * len(self.newton_polytope.parameters)
    return _exact.affine_rank([*points, origin]) > _exact.affine_rank(points)
```

This equals `any(row[0] != 0 for row in polytope_data(points).affine_hull)`, which the tests
check. The docstring states Lee's criterion and its equivalent forms: a solution $k$ of
$\sum_e k_e u_e\, \partial G / \partial u_e = G$, an affine-hull equation with $h_0 \ne 0$, and
the origin outside the affine hull. It also states that such integrals vanish in dimensional
regularisation, that the flag implies the polytope is not full-dimensional, and the converse
counterexample `1ee|1|:zn`. The flag depends on the support of $G$ only, not on $D$ or $\nu$.

### Exact vertices and edges in every dimension

`_invariants.hull_vertex_indices(points)` returns `polytope_data(points).vertex_indices`, and
`vertex_edge_graph(vertices)` takes its edges from the two vertices of each face of dimension 1
of `polytope_data(vertices)`. `_facet_incidence` and the floating hulls go. Every caller then works
on exact vertices and the certified 1-skeleton: `compute_polytope_automorphisms`,
`symmetry_pairs`, `is_unimodular_equivalent`, `is_affinely_equivalent` and
`AConfiguration.newton_polytope_points`. In full dimension the search is otherwise unchanged and
gives 24, 24 and 72 for the three graphs of Current state.

`vertex_permutations` and `vertex_orbits` then index `data.vertex_indices`, the list from which
the report numbers its vertices $v_k$.

The label of a vertex pairs its Liu-Cai label over its neighbours with the same determinant over
all the other vertices, $\det \sum_{w \ne v} (w - v)(w - v)^T$, which is a unimodular invariant
for the same reason. The pair separates vertices that the neighbours alone leave together: the 45
vertices of `112|3|4e|5e|5e|e|:nnnnzzz` fall into 6 classes of the first and into 9 of the pair,
the orbits of its group. Both determinants are non-negative, and Cantor's pairing function stores
the pair as one integer, so the labels stay integers.

### Automorphisms below full dimension

`compute_polytope_automorphisms(points)`:

1. `data = polytope_data(points)`, once, for the vertices and the 1-skeleton.
2. A single vertex: the identity.
3. $d = n$: the basis search on the exact vertices, with Liu-Cai labels on the exact skeleton.
   On a segment in $\mathbb{R}^1$ it finds the reflection, so `0|:n` gets order 2 and no longer
   raises.
4. $1 \le d < n$: build the chart of the vertices, `chart_frame(data.vertices)`, and compute the
   labels on the exact skeleton in chart coordinates, where they are $d \times d$ determinants
   invariant under $GL_d(\mathbb{Z})$. For `012e|2e|e|:znnn` they are $(1, 1, 1, 16, 16, 16)$
   where the ambient ones are all 0. Run the same search on the chart vertices, which are
   full-dimensional in $\mathbb{Z}^d$, keep the maps whose linear part preserves $\Lambda$, and
   lift them as in (c). A segment is handled the same way: its chart is $\{0, 1\}$ and its
   reflection always preserves $\Lambda$.

`order` counts permutations, and `maps` holds one lift per permutation.

A private module `normal_forms/_chart.py` holds the frame and the lift, shared by the
automorphisms, the equivalence tests and `symmetry_pairs`, and later by `finite_index_map`:

```python
@dataclass(frozen=True)
class ChartFrame:
    chart: LatticeChart
    frame: sp.ImmutableMatrix    # Q in GL_n(Z); its last d columns are a basis of sat(L)
    coframe: sp.ImmutableMatrix  # Q^-1 = W^T
    hermite: sp.ImmutableMatrix  # H, with B = S_Q H^T

def chart_frame(points: Sequence[Sequence[int]] | np.ndarray) -> ChartFrame: ...
def lift_linear(
    source: ChartFrame, target: ChartFrame, m: sp.Matrix, *, integral: bool = True
) -> sp.ImmutableMatrix | None:
    """Q' diag(I, H'^T m H^-T) Q^-1, or None when integral and H'^T m H^-T is not integral."""
```

It uses `_exact.hermite_normal_form_with_transform`, as `polytope._lift_data` does, and SymPy for
the inverse of $W$. `polytope.py` does not change; in `_exact.py` only `_as_int` does (see Exact
arithmetic in the searches).

### Symmetry pairs and equivalence tests

- `symmetry_pairs` below full dimension runs on the chart of all the points, whose pairs are
  found by the full-dimensional search, keeps those that preserve $\Lambda$, and returns the
  lifted pairs. This replaces the `return []`. A segment needs no special case, and
  `0|:n` no longer raises. The docstring's argument that $M^k = I$ holds for the chart map, and
  the lift is a homomorphism that is the identity on the complement.
- `is_unimodular_equivalent` and `is_affinely_equivalent` reduce to the charts of the exact vertex
  sets, with the exact skeletons for the labels, when the polytopes are below full dimension. They
  require equal affine dimension and, for the unimodular test, equal sublattice index.
  `_direct_basis_search` takes an acceptance test, the $\Lambda$ condition, and goes on past a
  rejected candidate. `is_point_config_equivalent` does the same on the charts of all points.
  The witnesses are the lifts. `FeynmanIntegral.is_unimodular_equivalent_to` and
  `is_affinely_equivalent_to` delegate, so their code does not change.
- `finite_index_map` below full dimension is left to a later change with the same approach.
  Charts of both configurations with equal $d$ make the basis matrix square. The loop becomes a
  generator, so that a candidate that fails the $\Lambda$ test does not end the search. The
  witness is the lift and the determinant is $|\det M_S|$.

### Exact arithmetic in the searches

Every decision in the automorphism, symmetry-pair and equivalence searches is exact: floating
point is not used, not even to pre-filter. `finite_index_map` keeps its floating code until the
later change.

- Ranks. The greedy choices of a basis, `_select_basis_indices_by_label` and
  `_select_basis_indices` in `polytope_automorphisms.py` and `_basis_indices` in
  `a_configuration.py`, test each candidate row with `_exact.rank`, and `symmetry_pairs` chooses
  the chart path by the exact rank of the differences of the points. With a floating rank the
  triangle with a point on an edge, $(0,0), (1,0), (2,0), (k,1)$, got the identity alone from
  `polytope_automorphisms` for $k \ge 8 \cdot 10^7$, and `symmetry_pairs` recursed without end,
  since the floating rank sent the points to a chart that was the points themselves.
- Candidate maps. `_basis_search`, `_direct_basis_search` and the full-dimensional path of
  `symmetry_pairs` find each candidate $U = W_b W_a^{-1}$ as $W_b \operatorname{adj}(W_a) /
  \det W_a$ in integer arithmetic, and keep it only when every entry divides exactly;
  `_invariants.integer_inverse` and `_invariants.integral_candidate` hold this arithmetic. A
  combination of vertices is tried only when $|\det W_b| = |\det W_a|$, by an exact Bareiss
  determinant. That test only prunes: a singular candidate fails the exact check, which follows,
  that the candidate maps the vertices, or the points, bijectively onto themselves. The floating
  inverse, rounding and determinant filters that these replace lost maps: under the map
  $S = \bigl(\begin{smallmatrix}1&k\\k&k^2+1\end{smallmatrix}\bigr)$ of $GL_2(\mathbb{Z})$ the
  unit triangle kept 2 of its 6 automorphisms and symmetry pairs at $k = 1000$.
- Overflow. The arrays are int64 when $n^2 s^2 \max|\operatorname{adj}(W_a)| < 2^{62}$, where $s$
  bounds every coordinate and every difference of coordinates, since that bounds every integer the
  search forms; otherwise they hold Python integers (`_invariants.exact_dtype`). `vertex_label`
  forms its moment matrices the same way.
- Inputs. `to_integer_points` raises `ValidationError` on a coordinate that is not an integer,
  where it truncated SymPy rationals, and `_exact._as_int` accepts a float only when it equals an
  integer exactly, at its own precision. The affine tests accept rational points: each
  configuration is scaled by the least common denominator $q$ of its coordinates, and a witness
  $M', t'$ of the scaled configurations gives $M = (q_a / q_b) M'$ and $t = t' / q_b$. The
  unimodular test does not scale, since scaling changes the lattice. A point repeated, compared
  with another, gets the translation between them with $M = I$.
- The equivalence search takes the basis of the first polytope from its rarest label classes, as
  the automorphism search does, and generates only the combinations of vertices of the second
  that carry the labels of that basis, class by class. It used to try every combination of the
  right size and compare its sorted labels, the same set in a far longer loop.

### Neighbour-restricted search

An automorphism maps a vertex to a vertex with the same label and degree, and its neighbours in
the 1-skeleton onto the neighbours of the image, keeping their labels and the edges among them.
`_basis_search` therefore fixes an anchor in the rarest label class and a basis among its
neighbours, whose edge directions span the space at a vertex of a full-dimensional polytope. For
each vertex that can be the anchor's image it maps the basis only to neighbours of that vertex
with the same labels and the same edges among them, and every candidate is then decided exactly,
as above. Where every vertex carries one label, as for the massless pentagon and hexagon, this is
what prunes the search.

A symmetry pair permutes the points, so it permutes the vertices of their convex hull and is an
automorphism of the polytope. `symmetry_pairs` takes the pairs, in full dimension, as the
automorphisms that send every point to a point (`configuration_symmetries`), instead of a second
search over all the points.

The results come in the order in which a search over all the vertices, or all the points,
anchored at the first with a basis from the rarest label classes, finds them: a map's place in
that search follows from its permutation alone (`_search_order`), and the maps are sorted by it.
So the groups, the pairs and their order are those of the searches they replace.

### Report and CLI

The flag:

- `Polynomials.scaleless: bool`, from `integral.is_scaleless`. `Polynomials` is always built, so
  the summary always has the row `("Scaleless", "yes" | "no")`, placed after "Codimension".
- The Newton polytope paragraph keeps its rank sentence keyed on `data.is_full_dimensional`, and
  below full dimension adds one sentence, citing Lee:
  - when scaleless: the origin does not lie in the affine hull of $P$, so for an equation
    $h_0 + h \cdot x = 0$ of the affine hull with $h_0 \ne 0$ the rescaling multiplies the
    Lee-Pomeransky integral by $\lambda^{h_0 D/2 + h \cdot \nu}$, and dimensional regularisation
    sets it to zero;
  - otherwise: the origin lies in the affine hull of $P$, so the factor is
    $\lambda^{h \cdot \nu}$, which does not involve $D$; the integral is not scaleless by Lee's
    criterion, and dimensional regularisation does not regulate it.
- `fk analyse`: the Newton section gains a line after "Affine dimension": `Scaleless  yes` or
  `no`. The rank remark stays keyed on full-dimensionality, so it follows the polytope and not the
  flag. `fk analyse` reads `gkz.a_matrix` while the flag reads the support of $G$; with one GKZ
  column per monomial of $G$ the two agree.
- `fk analyse --json`: `_json_value` maps "yes" and "no" to `true` and `false`, so the summary
  gains `"scaleless": true` or `false`. No other summary value is "yes" or "no".

The symmetries. The report and the CLI stop leaving out the symmetries, in every dimension:

- `_symmetries_omitted`, `SymmetriesOmitted`, `AnalysisReport.symmetries_omitted` and
  `symmetries_omitted()` in `_report_shared.py` go. Removing the field is a breaking change for
  the changelog.
- `Symmetries` gains `full_dimensional: bool`. The symmetry section prints the order, the vertex
  orbits, the graph automorphisms and the coefficient-preserving count in every dimension. For
  `012e|2e|e|:znnn`: order 6, orbits $\{v_1, v_2, v_3\}$ and $\{v_4, v_5, v_6\}$, 2 graph
  automorphisms.
- Below full dimension the text calls $\mathrm{Aut}(P)$ the group of affine maps of the affine
  hull of $P$ onto itself that preserve its integer points and take $P$ to itself, since the
  unimodular maps of $\mathbb{Z}^n$ taking $P$ to itself form an infinite group.
- Below full dimension, one sentence replaces the paragraph listing the symmetry pairs and the
  identities they give. It gives their number and says that each identity
  $I_A(\beta, z_\sigma) = I_A(T\beta, z)$ holds only trivially: the integral converges absolutely
  for no $D$ and $\nu$, $T\beta$ depends on how $T$ is lifted, and for generic $D$ and $\nu$ the
  GKZ system has no non-zero solutions. The data still holds the pairs.
- `fk analyse -S` prints the same numbers in every dimension, with the `|Aut(P)|` key, and below
  full dimension a note after the count of symmetry pairs.
- `fk compare` runs the `unimodular` and `affine_polytope` checks in every dimension. Below full
  dimension it notes after the identity of a `point_config` map that it holds only trivially.
  `finite_index` stays n/a below full dimension until the later change. The exit statuses do
  not change: 0 when a check finds a map, 3 when none does.

### Database repair

`_migrate` runs numbered repair steps under `PRAGMA user_version`; the GKZ-column repair is the
step from 0 to 1. This change appends the step from 1 to 2, `_repair_automorphisms`, which
follows the contract of `_migrate`: it runs inside the step's transaction, neither commits nor
rolls back, and is harmless on the empty tables of a new file.

- `DELETE FROM equivalences`. The floating path produced false negatives in full dimension as
  well, whenever the floating hull returned extra points, and telling which cached verdicts are
  affected would mean recomputing every one; below full dimension a cached positive verdict can
  carry a singular affine witness. Emptying the cache costs only recomputation.
- `UPDATE integrals SET poly_aut_order = NULL, graph_aut_order = NULL, coeff_pres_order = NULL,
  vertex_orbits = NULL`. The next store with `compute_automorphisms=True` fills them again.

A file at version 0 runs both steps in turn, each in its own transaction.

## Interfaces

```python
# feynkit/integral.py
class FeynmanIntegral:
    @cached_property
    def is_scaleless(self) -> bool: ...

# feynkit/io/report.py
@dataclass(frozen=True)
class Polynomials:
    ...
    scaleless: bool

@dataclass(frozen=True)
class Symmetries:
    ...
    full_dimensional: bool
```

`SymmetriesOmitted` and `AnalysisReport.symmetries_omitted` are removed. Signatures of
`compute_polytope_automorphisms`, `hull_vertex_indices`, `vertex_edge_graph`,
`is_unimodular_equivalent`, `is_affinely_equivalent`, `is_point_config_equivalent`,
`symmetry_pairs` and `finite_index_map` do not change. `hull_vertex_indices` keeps returning
sorted input indices, now exact. The docstring of `PolytopeAutomorphisms` says that below full
dimension `maps` holds one lift per permutation and `order` counts permutations.

## Changes to core modules

- `integral.py`: the property `is_scaleless`. The docstring of `polytope_automorphisms` states the
  definition as lattice-polytope automorphisms in $\mathrm{aff}(P) \cap \mathbb{Z}^n$ and that
  below full dimension the maps are lifts. The docstring of `symmetry_pairs` no longer says that
  the pairs coincide with the polytope automorphisms without qualification. The docstring of
  `canonicalise` no longer refers to later work on the face lattice.
- `a_configuration.py`: `symmetry_pairs` takes the chart path below full dimension in place of
  `return []`, with its docstring and that of `AConfiguration.symmetry_pairs`. It chooses that
  path by an exact rank, `_basis_indices` tests independence exactly, and the full-dimensional
  path finds its candidate maps in integers. `_to_pts` raises `ValidationError` on a coordinate
  that is not an integer, where it truncated it. `AConfiguration.newton_polytope_points` and the
  labels in `symmetry_pairs` become exact through `hull_vertex_indices` and `vertex_edge_graph`,
  with no change to their code. `finite_index_map` keeps its floating rank and rounding.
- `database.py`: the repair step from `user_version` 1 to 2.
- `polytope.py`: none.

## Changes elsewhere

- `normal_forms/_invariants.py`: exact `hull_vertex_indices` and `vertex_edge_graph`, the skeleton
  and the paired labels of a `PolytopeData`, `to_integer_points` raising on non-integers, the
  integer candidate helpers `exact_dtype`, `integer_inverse` and `integral_candidate`, and
  `vertex_label` guarded against overflow.
- `normal_forms/polytope_automorphisms.py`: one `polytope_data`, the chart path, the factored
  search, and exact ranks and candidates.
- `normal_forms/affine_equivalence.py`: the chart reduction, the acceptance test, exact ranks and
  candidates, the combinations generated by label from a label-diverse basis, denominators
  cleared in the affine tests, and the witness of a repeated point.
- `_exact.py`: `_as_int` accepts a float only when it is an integer exactly.
- `normal_forms/_chart.py`: new.
- `types.py`: the docstring of `PolytopeAutomorphisms`.
- `io/report.py` and `io/_report_shared.py`: the flag, the summary row, `full_dimensional`, and
  the removal of the omission.
- `io/report_text.py` and `io/report_latex.py`: the Newton sentences, the symmetry wording and
  the replacement sentence; Lee in the bibliography.
- `cli.py`: the Newton line, the JSON boolean, `_print_symmetries`, `fk compare`, and
  `_vertices_and_volume`, which reads `polytope_data` in every dimension.
- `docs/triangle_analysis.tex`, `.txt` and `.pdf`: regenerated, since the summary gains a row.
- `docs/automorphism_groups.md`: the group below full dimension.
- Mathematics reference: sections 7.1 and 7.3 on vertex semantics and the lattice of
  $\mathrm{aff}(P)$; section 5.3 on the flag; section 8.3 below full dimension; a bibliography
  entry for Lee.
- The guide, the changelog and the tests.

## Testing and acceptance

- `is_scaleless` on every row of the table in Scaleless integrals, including the kinematic points
  set with `with_`. It agrees with `any(row[0] != 0 for row in data.affine_hull)`, and
  `is_scaleless` implies `not data.is_full_dimensional`.
- The summary has the row "Scaleless" for every report; `fk analyse --json` gives a JSON boolean;
  the Newton section prints the line; the rank remark follows full-dimensionality for `1ee|1|:zn`.
- Regressions in full dimension: `112|3|4e|5e|5e|e|:nnnnnzz`, `112|3|4e|5e|5e|e|:nnnnzzz` and
  `112|3|4e|5e|5e|e|:zznnnzz` give 24, 24 and 72 from `polytope_automorphisms`, and the same
  counts from `symmetry_pairs` in both the Newton and the A-matrix order of the points. The
  relabelled support of Current state, stored as a fixed point list, gives order 4 and is
  unimodularly equivalent to the original. The vertex permutations index `data.vertices`.
- Orders below full dimension: every row of the table in Current state. Each of the first four
  equals the order of the graph without the self-loop.
- Embedding: a full-dimensional lattice polytope and its image under
  $x \mapsto Q\,(x, 0) + o$ with $Q \in GL_n(\mathbb{Z})$ have the same order. This includes the
  index-2 triangle, 2 both ways, where $\mathrm{Aut}_L$ would give 6.
- Lifts: every `(U, t)` is integral with $|\det U| = 1$ and maps the vertex set onto itself, and
  `vertex_permutations` agrees with it. For the Feynman examples it maps the support onto itself.
  `coefficient_preserving_indices` on the lifts matches the graph without the self-loop.
- Equivalence: a polytope below full dimension and its image under a unimodular map are
  unimodularly equivalent with a witness in $GL_n(\mathbb{Z})$; the affine and point-configuration
  witnesses are invertible; the segment pairs of Current state give true and false, and all
  segments are affinely equivalent; `012e|2e|e|:znnn` and `12e|12e|e|:nnzn` are equivalent,
  `012e|2e|e|:znnn` and `012e|2e|e|:zzzz` are not. `symmetry_pairs` below full dimension has as
  many pairs as the configuration group.
- The two property tests of Current state, the double box against its relabelling and
  `012e|3e|3e|e|:znnnn` against itself, pass, and their expected-failure marks go.
- Database: a file with cached verdicts and automorphism columns, written by 0.4.0, is repaired
  once. After the repair, `find_equivalent` for `012e|2e|e|:znnn` returns `12e|12e|e|:nnzn`. A
  file at version 0 gets both repairs, and a file at version 2 none.
- The tests that pin the omission change: those in `tests/io/test_report.py`,
  `tests/io/test_report_text.py` and `tests/cli/test_cli.py` that expect "not computed" or n/a.

## Performance

The flag costs two exact ranks of $N$ points. The exact vertices cost one `polytope_data`, which
the report computes anyway: under 0.1 s for the two-loop graphs above (42 to 54 points). The chart
path adds a Hermite normal form of a $d \times n$ matrix and a few small exact inverses, and the
exact candidates cost no more than the floating ones did.

Measured on the final code and on 0.4.0, each case in a fresh process with other work running:

| Case | 0.4.0 | Final |
|---|---|---|
| `polytope_automorphisms`, massless box, order 120 | 1.8 s | 0.6 s |
| `polytope_automorphisms`, the three two-loop graphs of Current state | 0.14 to 0.19 s, orders 4, 12, 12 | 0.3 to 0.4 s, orders 24, 24, 72 |
| `polytope_automorphisms`, massless pentagon, order 720 | 131 s | 0.3 s |
| `symmetry_pairs`, massless pentagon | 222 s | 0.4 s |
| `polytope_automorphisms`, massless hexagon, order 5040 | over 20 minutes | 2.0 s |
| `symmetry_pairs`, massless hexagon | over 20 minutes | 2.3 s |
| `symmetry_pairs` on the columns of A, `112\|3\|4e\|5e\|5e\|e\|:nnnnzzz` | under 1 s | 0.4 s |
| `polytope_automorphisms`, `123\|4e\|4e\|5e\|5\|e\|:nnnnnnn` | 0.3 s, order 1 | 0.3 s, order 48 |
| `symmetry_pairs`, `123\|4e\|4e\|5e\|5\|e\|:nnnnnnn` | 0.2 s, 1 pair | 0.5 s, 48 pairs |
| `polytope_automorphisms`, `123\|4e\|4e\|4e\|\|:nnnnnn`, order 48 | over 10 minutes | 0.3 s |
| `is_unimodular_equivalent`, `123\|24\|e\|5e\|5e\|e\|:znnzzzz` and a relabelling | 66 s | 0.6 s |
| `is_unimodular_equivalent`, `112\|3\|4e\|5e\|5e\|e\|:nznnzzz` and a relabelling | 0.0 s, not equivalent | 0.5 s, equivalent |

Where 0.4.0 was faster it was wrong. Along the way the exact 1-skeleton made some searches slower:
its labels separated fewer vertices than the floating ones happened to, and the equivalence
search filtered every combination of vertices by its labels, so that seven-propagator two-loop
graphs took minutes against a relabelling, more than 600 s for `123|24|e|5e|5e|e|:znnzzzz`. The
paired labels, and the combinations generated by label from a basis in the rarest label classes
(Exact arithmetic in the searches), removed both.

Where the labels separate nothing the search over all vertices stayed slow: every vertex of the
Newton polytopes of the massless pentagon `12e|3e|4e|4e|e|:zzzzz` and hexagon
`12e|3e|4e|5e|5e|e|:zzzzzz` carries the same label, the pair included, since their groups, of
orders 720 and 5040, act transitively on the vertices. With the paired labels the pentagon's
automorphisms and pairs took 53 s and 100 s, and the hexagon's automorphisms did not finish in 25
minutes. The neighbour-restricted search takes the table's times. On the 228 generated two-loop
graphs with seven propagators, the automorphisms took 78 s in all instead of 129 s and the pairs
108 s instead of 317 s, with the same results in the same order.

## Out of scope

- `finite_index_map` below full dimension, left to a later change.
- Equivalence across ambient dimensions, for instance of `012e|2e|e|:zzzz` with `12e|2e|e|:zzz`,
  whose charts are unimodularly equivalent. `fk compare` still stops at an ambient mismatch.

## Decisions

- `FeynmanIntegral.is_scaleless` follows Lee's criterion: the origin is not in the affine hull of
  the support of $G$. It is false for `1ee|1|:zn`. There is no second public property for the
  weaker condition, which `polytope_data(...).is_full_dimensional` already gives.
- The group is the group of lattice-polytope automorphisms in $\mathrm{aff}(P) \cap \mathbb{Z}^n$,
  the part of $\mathrm{Aut}_L$ that preserves $\Lambda$.
- `polytope_automorphisms` keeps vertex semantics, and sections 7.1 and 7.3 of the mathematics
  reference are corrected to say so.
- Below full dimension one sentence replaces the pair identities in the report, and
  `symmetry_pairs` gains its chart path.
- The omission of symmetries below dimension 2 and below full dimension goes. This is a breaking
  change: `AnalysisReport.symmetries_omitted`, `SymmetriesOmitted` and `symmetries_omitted()` are
  removed.
- A database repair step empties the equivalence cache and nulls the automorphism columns. It is
  the step from `user_version` 1 to 2.
- The automorphism, symmetry-pair and equivalence searches decide everything in exact
  arithmetic, and a vertex label pairs the Liu-Cai label with the same determinant over all the
  other vertices.
