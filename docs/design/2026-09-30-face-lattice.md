# The decorated face lattice

Design note for a module `feynkit/face_lattice.py` that decorates every face of a GKZ
configuration, the empty face included, with the values of $\varepsilon$ at which it is resonant or
admissible, its lattice defect and whether $A$ is a pyramid over it, and decides from these the
resonance centres and the reducibility of the system; for `FeynmanIntegral.face_lattice`; and for
its place in the analysis report and in `fk analyse`.

References: Britto, Grimm and Hoefnagels, arXiv:2606.09978 (BGH26 below), and Schulze and
Walther, arXiv:1009.3569 (SW12). Page numbers are those of the arXiv versions. The design note
`docs/design/2026-09-30-resonance.md`, the resonance note below, treats the facets.

## Purpose

The resonance note classifies the facets. Faces of lower dimension need a test on several forms
at once, and BGH26 name lower-dimensional resonant faces, such as intersections of edge facets, as
a direction for further work (pp. 50-51). Reducibility needs them as well. The resonance centres
of SW12 are faces of any dimension, the empty face included, so the facets alone can show that a
system is reducible but never that it is irreducible: `FacetResonance.reducible` is True or None.
With every face decorated, reducibility is decided at every $\varepsilon$ whenever $A$ has full
rank.

## Conventions

$A$ is a $d \times n$ integer matrix with distinct columns $a_j$ and is homogeneous: some linear
form takes the value 1 on every column. Write $\Lambda = \mathbb{Z}A$,
$V = \operatorname{span}_{\mathbb{C}} A$ and $r = \operatorname{rank} A$, and for a set $G$ of
columns $V_G = \operatorname{span}_{\mathbb{C}} A_G$ and $\bar G$ for the columns off $G$.

A face of $A$ is a set $G$ of columns for which some linear functional on $\Lambda$ vanishes on $G$
and is positive on $\bar G$; $A$ is a face, and $A$ is positive, as a homogeneous $A$ is, exactly
when the empty set is a face (SW12 Def. 3.1, p. 5). The non-empty faces are those of the polytope
$\operatorname{conv}(A)$, which lies in the hyperplane where the homogenising form $\eta$ is 1: a
linear functional restricts to an affine function there, and an affine function $c + l$ there is
the restriction of the linear functional $c\eta + l$. `polytope_data` computes these faces, with
certified facets (section 5.5 of the mathematics reference). The facets are the faces $F$ with
$\operatorname{rank} A_F = r - 1$; when $r = 1$, the empty face is the only one.

The parameter is $\beta(\varepsilon) = \beta_0 + \varepsilon\beta_1$. For the Lee-Pomeransky
configuration, with columns $(1, \alpha_j)$ and $\beta = (-D/2, -\nu_1, \ldots, -\nu_N)$ at
$D = D_0 - 2\varepsilon$ and integer powers, $\beta_0 = (-D_0/2, -\nu)$ and $\beta_1 = e_0$. A face
$G$ is resonant when $\beta \in V_G + \Lambda$ (BGH26 Eq. 23, p. 12) and
admissible when $\beta \in V_G$ (BGH26 pp. 10-11). SW12's resonance centres are the minimal faces
with this property (p. 5), while their "$F$-resonant" (Def. 3.2, p. 5) means
$\beta \in \mathbb{Z}A + \mathbb{C}G$ for a proper subface $G$ of $F$.

## The face test

**Proposition 1.** Let $G$ be a face of $A$ and $N_G$ the group of linear forms on $V$ that take
integer values on $\Lambda$ and vanish on $A_G$. Then $N_G$ is free of rank
$k = r - \operatorname{rank} A_G$, and for a $\mathbb{Z}$-basis $h_1, \ldots, h_k$ of $N_G$ and
every $\beta \in \mathbb{C}^d$:

1. $G$ is resonant for $\beta$ if and only if $\beta \in V$ and $h_i(\beta) \in \mathbb{Z}$ for
   every $i$;
2. $G$ is admissible for $\beta$ if and only if $\beta \in V$ and $h_i(\beta) = 0$ for every $i$.

*Proof.* $\Lambda_G = \Lambda \cap V_G$ is saturated in $\Lambda$: if $x \in \Lambda$ and
$mx \in V_G$ for an integer $m \ne 0$, then $x \in V_G$. So $\Lambda/\Lambda_G$ is free, of rank
$\operatorname{rank}\Lambda - \operatorname{rank}\Lambda_G = r - \operatorname{rank} A_G$, since
$A_G \subseteq \Lambda_G \subseteq V_G$. A linear form on $V$ is fixed by its values on $\Lambda$,
which spans $V$, and every homomorphism $\Lambda \to \mathbb{Z}$ extends to one; the form vanishes
on $A_G$ exactly when it vanishes on $V_G$, and then on $\Lambda_G$. So restriction to $\Lambda$
identifies $N_G$ with $\operatorname{Hom}(\Lambda/\Lambda_G, \mathbb{Z})$, which is free of rank
$k$. Write $H = (h_1, \ldots, h_k)$. As a basis of the dual of the free group
$\Lambda/\Lambda_G$, $H$ maps $\Lambda/\Lambda_G$ isomorphically onto $\mathbb{Z}^k$, so
$H(\Lambda) = \mathbb{Z}^k$. Then $H$ maps $V$ onto $\mathbb{C}^k$, its kernel on $V$ has
dimension $\dim V - k = \dim V_G$, and it contains $V_G$, so it is $V_G$. This is part 2. For
part 1, if $\beta = v + w$ with $v \in V_G$ and $w \in \Lambda$, then $\beta \in V$ and
$H\beta = Hw \in \mathbb{Z}^k$. Conversely, if $\beta \in V$ and $H\beta \in \mathbb{Z}^k$, choose
$w \in \Lambda$ with $Hw = H\beta$, which $H(\Lambda) = \mathbb{Z}^k$ allows; then
$\beta - w \in V_G$. $\square$

For a facet, $k = 1$ and $h_1$ is the primitive form $l_F$ of the resonance note, whose
Proposition 1 is this case. For the empty face, $N_\emptyset$ holds every form integral on
$\Lambda$, so the empty face is resonant exactly when $\beta \in \Lambda$, as SW12 note (p. 5). For
$A$ itself, $k = 0$, and $A$ is resonant and admissible exactly when $\beta \in V$.

**In coordinates.** A column Hermite normal form of $A$ gives a basis $B$ of $\Lambda$ and integer
coordinates $c_j$ with $a_j = Bc_j$. A form is then a row in $\mathbb{Z}^r$, its values on the
basis, and $N_G$ is the integer kernel of the matrix with rows $c_j$, $j \in G$: a saturated
sublattice of $\mathbb{Z}^r$, whose basis a second Hermite normal form gives. $\mathbb{Z}A$ need
not be $\mathbb{Z}^d$.

**The sets in $\varepsilon$.** Where $\beta(\varepsilon) \in V$, write $p = H\beta_0$ and
$q = H\beta_1$; $G$ is resonant at $\varepsilon$ exactly when $p + \varepsilon q \in \mathbb{Z}^k$.
If $q = 0$ this holds for every $\varepsilon$ or for none. Otherwise $q = su$ with $s > 0$
rational and $u$ a primitive integer vector, and some $v \in \mathbb{Z}^k$ has $v \cdot u = 1$. If
$p + \varepsilon q$ is integral, put $t = s\varepsilon$: then $v \cdot (p + tu) = v \cdot p + t$ is
an integer, so $t \in t_0 + \mathbb{Z}$ with $t_0 = -v \cdot p$. For every integer $m$,
$p + (t_0 + m)u$ differs from $p + t_0 u$ by $mu \in \mathbb{Z}^k$. So the resonant set is
$(t_0 + \mathbb{Z})/s$ when $p + t_0 u$ is integral and empty when it is not. The admissible set
solves $p + \varepsilon q = 0$: every $\varepsilon$, one value, or none. Both are intersected with
the values at which $\beta(\varepsilon) \in V$, which are every $\varepsilon$, one value or none,
as in the resonance note. So each set is an `EpsilonSet` of one of its four kinds. The offset of a
resonant progression is the admissible value when there is one, as for the facets, and otherwise
lies in $[0, \text{period})$.

## Faces against facets

For a face $G$, write $F \supseteq G$ for the facets containing it and $l_F$ for their primitive
forms, which by Proposition 1 with $k = 1$ generate the groups $N_F$. They must be primitive on
$\Lambda$, not merely in $\mathbb{Z}^d$: the module computes them as the generators of $N_F$, and
`Facet.lattice_form` is one as well.

**Proposition 2.** Let $G$ be a face of $A$.

1. If $G \subseteq G'$ are faces, $G'$ is resonant and admissible wherever $G$ is.
2. $V_G = V \cap \bigcap_{F \supseteq G} \ker l_F$, so $G$ is admissible exactly where every facet
   containing it is.
3. The $l_F$, $F \supseteq G$, span a subgroup of finite index in $N_G$; call the quotient $T_G$,
   the lattice defect. Where $G$ is resonant, so is every facet containing it. If $\beta \in V$ and
   every facet containing $G$ is resonant for $\beta$, then $h \mapsto h(\beta) + \mathbb{Z}$ is a
   homomorphism $\chi_\beta : T_G \to \mathbb{Q}/\mathbb{Z}$, and $G$ is resonant for $\beta$
   exactly when $\chi_\beta = 0$. So $G$ is resonant wherever all the facets containing it are, for
   every $\beta \in V$, if and only if $T_G = 0$.

The last statement quantifies over all of $V$: along a single line $\beta(\varepsilon)$ a
non-trivial $T_G$ need not show.

*Proof.* (1) $V_G \subseteq V_{G'}$.

(2) Every $l_F$ with $F \supseteq G$ vanishes on $V_G$, so $V_G \subseteq W$, the right-hand side.
Conversely, let $x \in W$. The cone $C = \mathbb{R}_{\ge 0}A$ is the set of $y \in V$ with
$l_F(y) \ge 0$ for every facet $F$. For $r \ge 2$ these are the facet inequalities of
$\operatorname{conv}(A)$, which `polytope_data` certifies, made homogeneous; they imply
$\eta \ge 0$, since $\operatorname{conv}(A)$ is bounded and so some positive combination of the
$l_F$ is a positive multiple of $\eta$ on $V$. For $r = 1$ the description is plain. Let
$y = \sum_{j \in G} a_j$, which is 0 for the empty face. Every facet $F \supseteq G$ has
$l_F(y) = 0$, and every other facet has $l_F(y) > 0$, since some column of $G$ lies off it. So
$y + tx$ and $y - tx$ lie in $C$ for small $t > 0$. Let $\phi$ vanish on $G$ and be positive on
$\bar G$. It is non-negative on $C$ and zero at $y$, so $\phi(y \pm tx) \ge 0$ forces
$\phi(x) = 0$, and $y \pm tx$ lie in $C \cap \ker\phi$. A point $\sum_j \lambda_j a_j$ of $C$ with
$\lambda_j \ge 0$ is in $\ker\phi$ only when $\lambda_j = 0$ off $G$, so $C \cap \ker\phi$ lies in
$V_G$; as $y \in V_G$, so does $x$. For $G = A$ there are no facets, $\phi = 0$, and $W = V = V_A$.
The statement on admissibility follows from part 2 of Proposition 1, for $G$ and for each facet.

(3) The $l_F$ lie in $N_G$, and by (2) their common kernel in $V$ is $V_G$, so they span the space
of forms on $V$ vanishing on $V_G$, of dimension $k$, and a subgroup of rank $k$ of $N_G$; its
index is finite. By (1), every facet containing a resonant $G$ is resonant. Let $\beta \in V$ have
$l_F(\beta) \in \mathbb{Z}$ for every $F \supseteq G$, which by Proposition 1 says that these facets
are resonant. For $h \in N_G$, $mh$ lies in the span of the $l_F$ for $m = |T_G|$, so
$mh(\beta) \in \mathbb{Z}$; then $h \mapsto h(\beta) + \mathbb{Z}$ is a homomorphism
$N_G \to \mathbb{Q}/\mathbb{Z}$ that vanishes on the $l_F$, and it factors through $T_G$. By
Proposition 1, $G$ is resonant exactly when $h(\beta) \in \mathbb{Z}$ for every $h \in N_G$, that is
when $\chi_\beta = 0$. If $T_G = 0$, $\chi_\beta$ is always 0. If $T_G \ne 0$, the finite group
$T_G$ has a non-zero homomorphism $\chi$ to $\mathbb{Q}/\mathbb{Z}$. Choose $x \in \mathbb{Q}^k$ with
$x_i + \mathbb{Z} = \chi(h_i)$ and, since $H$ maps $V$ onto $\mathbb{C}^k$, some $\beta \in V$ with
$H\beta = x$. Each $l_F$ is $\sum_i b_i h_i$ with integers $b_i$, so
$l_F(\beta) + \mathbb{Z} = \chi(l_F) = 0$ and every facet containing $G$ is resonant; but $\chi$ is
not zero on some $h_i$, so $h_i(\beta) \notin \mathbb{Z}$ and $G$ is not resonant. $\square$

$|T_G|$ is the product of the non-zero Smith invariants of the matrix whose rows are the $l_F$,
$F \supseteq G$, in coordinates: that product is the index of their span in its saturation, and
the saturation is $N_G$, which is saturated and contains the span with the same rank.

**Examples.**

- SW12's quadric cone, the columns $(1, 0)$, $(1, 1)$, $(1, 2)$, at $\beta = (1/2, 1)$ (Ex. 3.3,
  p. 6). Here $\Lambda = \mathbb{Z}^2$. The two rays are the facets, with forms $(0, 1)$ and
  $(2, -1)$; both are integers on $\beta$, so both rays are resonant. The empty face is resonant
  only for $\beta \in \mathbb{Z}^2$, which this $\beta$ is not. The two forms span a sublattice of
  index 2 in $\mathbb{Z}^2 = N_\emptyset$, so $T_\emptyset = \mathbb{Z}/2$, and $\chi_\beta$ is its
  non-trivial character.
- The triangle $(0,0)$, $(2,1)$, $(1,2)$ with its centroid $(1,1)$, homogenised: $\Lambda =
  \mathbb{Z}^3$, the edge forms are $(0,-1,2)$, $(0,2,-1)$ and $(3,-1,-1)$, and $|T_G|$ is 3 at each
  vertex and 9 at the empty face.
- On the 21 Feynman graphs of the tests, from the bubble to the massless double box, every face has
  $T_G = 0$.

## Resonance centres and reducibility

What is used from SW12:

- A resonance centre for $\beta$ is a minimal face $F$ with $\beta \in \mathbb{Z}A + \mathbb{C}F$;
  every $\beta$ has one, and for positive $A$ the empty face is a centre exactly when
  $\beta \in \mathbb{Z}A$ (Def. 3.2 and the remarks after it, p. 5).
- $A$ is a pyramid over the face $F$ when $\operatorname{rank}\mathbb{Z}A = |\bar F| +
  \operatorname{rank}\mathbb{Z}F$ (Def. 3.4, p. 6).
- If $\beta \in \mathbb{C}A$ has a centre $F$ over which $A$ is a pyramid, $F$ is the only centre
  (Prop. 3.8, p. 6).
- Let $F$ be a centre for $\beta \in \mathbb{C}A$. If $A$ is not a pyramid over $F$, $M_A(\beta)$
  has reducible monodromy (Thm 4.1, p. 7); if $A$ is a pyramid over $F$, it has irreducible
  monodromy (Thm 5.1, pp. 7-8).

SW12 assume $\mathbb{Z}A = \mathbb{Z}^d$ (section 2.1, p. 2). By their Remark 2.1 (p. 3), when
$\mathbb{Z}A$ only has rank $d$, writing $A = BA'$ with $\mathbb{Z}A' = \mathbb{Z}^d$, the systems
$H_A(\beta)$ and $H_{A'}(B^{-1}\beta)$ are equivalent for $\beta \in \mathbb{C}A$. $B$ maps
$\mathbb{Z}A'$ onto $\mathbb{Z}A$ and each $\mathbb{C}F'$ onto $\mathbb{C}F$, and keeps ranks, so
faces, resonance, centres and pyramids correspond, and the statements above hold for every $A$ of
rank $d$.

**The centres.** By Proposition 2(1), the faces resonant at a given $\varepsilon$ are closed
upwards, and the centres are the minimal ones. `centres(eps)` takes the resonant faces by
increasing dimension and keeps each that contains no centre found before: a resonant face that is
not minimal contains a minimal one, of smaller dimension since it is a proper face of it, which was
found first.

**Reducibility.** Suppose $\operatorname{rank} A = d$. Then $V = \mathbb{C}^d$, $A$ itself is
resonant for every $\varepsilon$, and some centre exists. If $A$ is a pyramid over some centre, it
is the only centre and the system is irreducible; otherwise $A$ is a pyramid over no centre and
the system is reducible. So `reducible(eps)` is True or False at every rational $\varepsilon$, and
at generic $\varepsilon$, outside the countably many progressions and values of all the faces,
where exactly the faces resonant for every $\varepsilon$ are resonant.

Two guards remain.

- Full rank means $\operatorname{rank} A = d$, the number of rows of $A$, not the flag
  `PolytopeData.is_full_dimensional` of the columns: the columns of a homogeneous $A$ lie on an
  affine hyperplane, so the flag is false even for the Cayley matrix of full rank. For the
  Lee-Pomeransky $A$, $\operatorname{rank} A = \dim P + 1$, so $A$ has full rank exactly when $P$ is
  full-dimensional. Below full rank, `reducible` is None.
- When no face is resonant, `reducible` is None as well. By Proposition 1 this happens only for
  $\beta \notin V$, where the system has no non-zero solutions (resonance note) and is neither
  reducible nor irreducible; at full rank it cannot happen.

In coordinates, $\operatorname{rank}\mathbb{Z}A_G = r - k$, so $A$ is a pyramid over $G$ exactly
when $|\bar G| = k$. $A$ is always a pyramid over itself, and over a facet exactly when one column
lies off it.

**The massless bubble.** The columns $(1,1,0)$, $(1,0,1)$, $(1,1,1)$ have determinant $-1$, so
$|\bar G| + \operatorname{rank} A_G = 3$ for every set $G$, and $A$ is a pyramid over every face.
Whichever face is the centre, the monodromy is irreducible, at every $\varepsilon$. At
$\varepsilon = 0$, $\beta = (-2, -1, -1) \in \Lambda$, so every face is resonant and the empty face
is the only centre: the monodromy is irreducible although every face is resonant. The D-module itself is
reducible there: $A$ is unimodular and $A^{-1}\beta = (-1,-1,0)$ is integral, so it is a product of
modules $D/(x\partial - \gamma_j)$, of rank 1. The facets leave this open, since each
has a single column off it.

**BGH26's examples.** With unit powers and $D = 4 - 2\varepsilon$:

- For the bubble with $m_2 = 0$, the edge face $F_1$, the face of the terms free of $u_1$, is a
  vertex, no longer a facet, and resonant only for $D/2 \in \mathbb{Z}$, that is on $\mathbb{Z}$,
  while $F_2$ is a resonant facet (p. 29).
- The edge faces of the sunrise with three masses are facets resonant for every $\varepsilon$
  (Eq. 152, p. 41).
- For the sunrise with $m_3 = 0$, $F_1$ and $F_2$ are not facets and not resonant at generic
  $\varepsilon$, while $F_3$ is a resonant facet (p. 43).
- For the sunrise with $m_2 = m_3 = 0$, none of the five facets and none of the edge faces is
  resonant (p. 44). The face lattice finds only $P$ resonant at generic $\varepsilon$, a centre over
  which $A$ is a pyramid, so the system is irreducible there.

## The Lee-Pomeransky lattice and the Schwinger side

The faces are indexed by the points of `fi.newton_polytope`, those of `polytope_data`.
`fi.gkz` numbers the same columns in another order, which is kept; `gkz_columns[j]` is the column
of `fi.gkz.a_matrix` equal to $(1, \alpha_j)$. Each face carries `through_origin`, whether the
origin lies in the affine hull of its points, which for a facet $m \cdot x \le b$ says $b = 0$, and,
up to `identify_codimension` (2 by default), its identification from `fi.face_identification`.
Identifying every face takes about 1.5 s on the massless double box, against 0.36 s for those of
codimension at most 2.

$T$ = `lp_to_cayley(N, L)` is unimodular, maps the Lee-Pomeransky columns onto the Cayley columns
and $\beta_{\mathrm{LP}}$ to the Cayley parameter (resonance note). It therefore maps faces to faces,
$\Lambda$ to the lattice of the Cayley columns and each $V_G$ to the span of the corresponding
face, and takes a form $h$ to $hT^{-1}$, so that $N_G$, the facet forms, $p$, $q$, ranks and
columns off all correspond: the two sides agree face by face. `cayley_columns[j]` is the column of
`fi.schwinger_gkz.a_matrix` equal to $T(1, \alpha_j)$. `check_schwinger()` does not rely on the
argument. It computes the faces of the Cayley columns with `polytope_data`, on the relative-facet
path since those columns lie on an affine hyperplane, decorates them for
$T\beta_0 + \varepsilon T\beta_1$, and compares dimension, resonant and admissible sets, lattice
defect, columns off and pyramid under the column map. The comparison agrees on the 21 graphs of
the tests.

## Cost

Seconds on one core, the decoration with the identifications already computed:

| Graph | Faces | Decoration | `check_schwinger()` | Identification up to codimension 2 |
|---|---|---|---|---|
| massless pentagon | 214 | 0.04 | 0.04 | 0.07 |
| massless hexagon | 520 | 0.12 | 0.11 | 0.13 |
| massive kite | 214 | 0.05 | 0.05 | 0.07 |
| massless double box | 2246 | 0.78 | 0.68 | 0.36 |
| massless double box on shell | 1534 | 0.51 | 0.44 | 0.25 |

Each face costs a Hermite normal form and a Smith form; the centres at one $\varepsilon$ take a few
milliseconds.

## Interfaces

`feynkit/face_lattice.py`, not a core module:

- `DecoratedFace`, frozen: `point_indices`, `dimension` ($-1$ for the empty face), `codimension`,
  `facets` (positions in the lattice's `facets`), `resonant`, `admissible`, `lattice_defect`,
  `columns_off`, `pyramid`, `through_origin` and `identification`.
- `DecoratedConfiguration`, frozen: `columns`, `rank`, `span`, `faces` (by dimension, the empty
  face first), `facets` (positions in `faces`, in the order of `PolytopeData.relative_facets`), with
  `full_rank`, `face(points)`, `resonance(points)`, `admissible(points)`, `defect(points)`,
  `resonant_faces(eps)`, `centres(eps)` and `reducible(eps)`; `eps` is an integer, a Fraction or
  `"generic"`.
- `decorate_configuration(a_matrix, beta0, beta1=None)` for any homogeneous $A$.
- `DecoratedFaceLattice`, a `DecoratedConfiguration` with `data`, `d0`, `d0_source`, `nu`,
  `facet_resonance`, `identify_codimension`, `gkz_columns`, `cayley_columns`, `lp_to_cayley` and
  `cayley_matrix`, and the methods `beta()` and `check_schwinger()`, which returns a
  `SchwingerCheck` (`bijective`, `same_faces`, `mismatches`, `faces` and `agrees`).
- `decorate_faces(fi, d0=None, *, nu=None, identify_codimension=2)`, with $D_0$ and the powers
  chosen as for `facet_resonance`; `FeynmanIntegral.face_lattice` with the same arguments caches it.

## Report and CLI

A report section `face_lattice`, built by default after `faces`, at the $D_0$ and powers of the
resonance section: the number of faces of each dimension by the kind of their resonant set, the
centres at generic $\varepsilon$ and at $\varepsilon = 0$ with the facets containing them, their
graphs and the verdict on reducibility, the faces besides $P$ over which $A$ is a pyramid and those
with $|T_G| \ne 1$ when there are any, and the summary rows `Resonance centres at generic D` and
`Reducible at generic D`. The comparison with the Cayley side checks the code rather than
adding to the analysis, and it about doubles the cost of the section, so the report makes it only
when asked
(`AnalysisReport.from_integral(..., check_schwinger=True)`). `fk analyse -L` (`--face-lattice`)
prints the section on the terminal with the comparison, and adds the section, with the comparison,
to the reports.

## Tests

- Every facet of 19 graphs, from the bubble to the massless pentagon and the six boxes below, as
  `classify_facets` classifies it; on every face of them $T_G = 0$ and the resonant and admissible
  sets are the intersections of those of the facets, on sampled $\varepsilon$; the sets grow with
  the face.
- BGH26's statements above, pp. 29, 41, 43 and 44.
- The quadric cone, the triangle with defects 3 and 9, and the face test on a simplex worked by hand.
- The box with massless propagators and $p_i^2 \ne 0$, with $p_2^2 = 0$, with $p_1^2 = p_4^2 = 0$,
  with $p_1^2 = p_2^2 = 0$, with only $p_1^2 \ne 0$, and on shell: 10, 9, 7, 9, 8 and 9 facets, one
  for each facet of $\operatorname{Newt}(\mathcal U\mathcal F)$, the $\mathcal U$ layer, and the
  $\mathcal F$ layer when $\operatorname{Newt}(\mathcal F)$ has dimension 3, computed in the test;
  and 6, 5, 4, 4, 3 and 2 centres at generic $\varepsilon$, each identified as a contraction.
- The massless bubble, irreducible at every $\varepsilon$ tried while `FacetResonance.reducible`
  stays None.
- The guards: $\beta$ off the span of a segment in $\mathbb{Z}^3$, the same segment in the span
  but below full rank, and the Cayley matrix of the massive bubble, of full rank though its columns
  are not full-dimensional; `1ee|1|:zn` and the massless bubble on shell.
- `check_schwinger()` on the 19 graphs, and on the massless double box off and on shell in the slow
  tier.

## Decisions

- Faces are indexed by the points of the Newton polytope, with maps to the column orders of
  `fi.gkz` and `fi.schwinger_gkz`; reordering `fi.gkz` would be a breaking change.
- One Lee-Pomeransky object, with `check_schwinger()` for the Cayley side, rather than one object per
  system.
- The empty face is included: it is the centre whenever $\beta \in \mathbb{Z}A$, as at integer
  $D_0/2$ with integer powers.
- `reducible(eps)` is True or False at full rank and None below it or where no face is resonant.
- The report decorates every face, about 0.8 s on the massless double box, and identifies those up
  to codimension 2, as the faces section does.
- The lattice defect is in the API and in the report only when it is not 1.
