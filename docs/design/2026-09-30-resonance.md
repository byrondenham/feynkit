# Facet resonance

Design note for a module `feynkit/resonance.py` that decides, for each facet of a GKZ
configuration, at which $\varepsilon$ the facet is resonant or admissible when
$D = D_0 - 2\varepsilon$ and the propagator powers are integers, and when a resonant facet makes
the system reducible; for `FeynmanIntegral.facet_resonance`; for a warning in
`CayleyGKZSystem.restrict_to_f_block`; and for their place in the analysis report and in
`fk analyse`.

References: Britto, Grimm and Hoefnagels, arXiv:2606.09978 (BGH26 below), and Schulze and
Walther, arXiv:1009.3569 (SW12). Page numbers are those of the arXiv versions.

## Purpose

Feynman integrals satisfy GKZ systems at resonant parameters, and BGH26 build their reduction
operators from the resonant faces. `polytope_data` already gives every facet of the Newton
polytope with a functional primitive on $\mathbb{Z}A$, but nothing in feynkit says which facets are
resonant, whether a face subsystem is a true subsystem, or whether the system is reducible. The
report's convergence conditions are the same functionals evaluated on $\beta$; the new section
reads them modulo $\mathbb{Z}$.

## Conventions

feynkit's Lee-Pomeransky system has columns $a_j = (1, \alpha_j)$, Euler equations
$E\Phi = \beta\Phi$ and $\beta = (-D/2, -\nu_1, \ldots, -\nu_N)$ (section 4.2 of the mathematics
reference; `fi.gkz.beta_parameters`). BGH26 write $(E_e + \nu_e) f = 0$ with
$\nu = (D/2, \nu_1, \ldots, \nu_N)$ (Eqs. 14 and 36, pp. 9 and 15), so their $\nu$ is $-\beta$.
Every set below, $\operatorname{span}_{\mathbb{C}} F$, $\mathbb{Z}A$ and their sum, is closed under
negation, so resonance and admissibility do not depend on the sign; a functional $L_F$ of BGH26
is the $l_F$ here, and $L_F(\nu) = -l_F(\beta)$.

The powers $\nu_e$ are integers, as in BGH26 (Eq. 46, p. 17: dimensional regularisation but not
analytic regularisation), and $D = D_0 - 2\varepsilon$ with $D_0$ an integer or a fraction, 4 by
default.

## Resonance of a facet

A face $F$ of $A$ is a set of columns for which some linear functional vanishes on $F$ and is
positive on the other columns (BGH26 Eq. 19, p. 10; SW12 Def. 3.1, p. 5). It is resonant for
$\beta$ when
$$\beta \in \operatorname{span}_{\mathbb{C}} F + \mathbb{Z}A$$
(BGH26 Eq. 23, p. 12), and we call it admissible, a name BGH26 do not use, when $\beta \in \operatorname{span}_{\mathbb{C}} F$:
then $(A_F, \beta)$ is a true subsystem, its solutions solving $(A, \beta)$ (BGH26 pp. 10-11).
BGH26 show that a facet is resonant exactly when $L_F(\beta) \in \mathbb{Z}$, assuming
$\mathbb{Z}A = \mathbb{Z}^d$ (Eq. 24, p. 12, proved in appendix A, p. 52). The assumption is not
needed.

**Proposition 1.** Let $A$ be an integer $d \times n$ matrix of rank $r$, $F$ a facet of $A$, a
face whose columns span a subspace of dimension $r - 1$, and $l_F$ a linear functional that
vanishes on $F$, is positive on the other columns and maps $\mathbb{Z}A$ onto $\mathbb{Z}$. Then
for every $\beta \in \mathbb{C}^d$:

1. $F$ is resonant for $\beta$ if and only if $\beta \in \operatorname{span}_{\mathbb{C}} A$ and
   $l_F(\beta) \in \mathbb{Z}$;
2. $F$ is admissible for $\beta$ if and only if $\beta \in \operatorname{span}_{\mathbb{C}} A$ and
   $l_F(\beta) = 0$.

*Proof.* Write $V = \operatorname{span}_{\mathbb{C}} A$. $l_F$ does not vanish on $V$, being positive
on a column off $F$, so $V \cap \ker l_F$ has dimension $r - 1$ and contains
$\operatorname{span}_{\mathbb{C}} F$, which has the same dimension; the two are equal. This is
part 2. If $\beta = v + w$ with $v \in \operatorname{span}_{\mathbb{C}} F$ and $w \in \mathbb{Z}A$,
then $\beta \in V$ and $l_F(\beta) = l_F(w) \in \mathbb{Z}$. Conversely, if $\beta \in V$ and
$l_F(\beta) = k \in \mathbb{Z}$, choose $w \in \mathbb{Z}A$ with $l_F(w) = k$; then
$\beta - w \in V \cap \ker l_F = \operatorname{span}_{\mathbb{C}} F$. $\square$

For a Newton polytope the functional is `Facet.lattice_form`: for a facet $m \cdot x \le b$ with
lattice index $g_F$ it is $l_F = (b, -m)/g_F$, whose values on the columns are integers with
greatest common divisor 1, so that $l_F(\mathbb{Z}A) = \mathbb{Z}$; below full dimension the lifted
facet forms of section 5.5 of the mathematics reference have the same property, and are fixed only
modulo the forms vanishing on $A$, which vanish on $V$.

**Off the span.** If $\beta \notin V$, some $h$ with $h^T A = 0$ has $h \cdot \beta \ne 0$. Then
$\sum_r h_r E_r = \sum_j (h \cdot a_j)\, z_j \partial_j = 0$, while the Euler equations give
$\sum_r h_r E_r \Phi = (h \cdot \beta)\Phi$, so $\Phi = 0$: the system has no non-zero solutions.
For the Lee-Pomeransky configuration the forms vanishing on $A$ are the equations
$h_0 + h \cdot x = 0$ of the affine hull of the polytope, read as $(h_0, h)$, and
$\beta \in V$ means $h_0 D/2 + h \cdot \nu = 0$ for each. When some $h_0 \ne 0$, the case
`FeynmanIntegral.is_scaleless` reports, this fixes $\varepsilon$ at one value at most; when every
$h_0 = 0$ it holds for every $\varepsilon$ or for none, as $h \cdot \nu$ vanishes or not.

## The classification in $\varepsilon$

For a facet $m \cdot x \le b$ of a full-dimensional Newton polytope,
$$l_F(\beta) = \frac{m \cdot \nu - bD/2}{g_F} = \frac{c_F + b\varepsilon}{g_F},
\qquad c_F = m \cdot \nu - \frac{b D_0}{2},$$
since $-bD/2 = -bD_0/2 + b\varepsilon$. By Proposition 1:

- $b = 0$: $l_F(\beta) = m \cdot \nu / g_F$ does not depend on $\varepsilon$. The facet is
  resonant for every $\varepsilon$ when $g_F \mid m \cdot \nu$ and for none otherwise, and
  admissible for every $\varepsilon$ when $m \cdot \nu = 0$ and for none otherwise.
- $b \ne 0$: the facet is resonant exactly on the progression
  $$\varepsilon \in \varepsilon_F + \frac{g_F}{|b|}\mathbb{Z}, \qquad
  \varepsilon_F = \frac{D_0}{2} - \frac{m \cdot \nu}{b},$$
  and admissible exactly at $\varepsilon = \varepsilon_F$. It is resonant at $\varepsilon = 0$ if
  and only if $g_F \mid c_F$.

With $g_F = 1$ the "never" class is empty, and $\varepsilon = 0$ is resonant exactly when
$b D_0/2 \in \mathbb{Z}$: for every facet when $D_0$ is even, for the facets with even $b$ when
$D_0$ is odd. The configuration $(0,0), (2,0), (0,1)$, whose facets $y \ge 0$, $x \ge 0$ and
$x + 2y \le 2$ have $g_F = 1, 2, 2$, reaches the "never" class: $x \ge 0$ has $b = 0$ and
$l_F(\beta) = -\nu_1/2$.

These agree with BGH26: an edge facet has $l_{F_e}(\beta) = -\nu_e$ and is resonant for every
$D$ (Eq. 47, p. 17) and admissible at $\nu_e = 0$ (Eq. 53, p. 18); the facets $F_{\mathcal F}$ and
$F_{\mathcal U}$ have $b = -L$ and $b = L + 1$ and are resonant when $LD/2 - \nu$, respectively
$(L+1)D/2 - \nu$, is an integer, with $\nu$ the sum of the powers, and their spans are where these
vanish (p. 46).

`classify_configuration` treats any homogeneous configuration the same way. With $\beta$ linear in
$D$ and $\nu$, $l_F(\beta) = c + s\varepsilon$ at given powers; the facet is resonant for every
$\varepsilon$ or none when $s = 0$ and on $-c/s + |s|^{-1}\mathbb{Z}$ otherwise, and admissible
at $-c/s$.

Below full dimension $\beta(\varepsilon) \in V$ for every $\varepsilon$, for one value
$\varepsilon^*$, or for none (`span_epsilons`), and each facet's sets are intersected with it. A
facet can then be resonant or admissible at the single point $\varepsilon^*$.

The sets are `EpsilonSet` values of kind "all", "progression" (offset and period), "point" or
"never". The offset of a resonant progression is $\varepsilon_F$, not reduced modulo the period,
so it is also the point of admissibility. `EpsilonSet.window(low, high)` lists the values in an
interval.

## Admissibility of any face

`admissible(a_matrix, face, beta)` decides $\beta \in \operatorname{span}_{\mathbb{C}} A_F$ for
the columns $F$ given: it takes a basis of the integer forms vanishing on them, from a Hermite
normal form, and asks each to vanish on $\beta$. Entries of $\beta$ may be SymPy expressions; with
symbols the forms must vanish identically. It does not check that $F$ is a face.

## Reducibility

SW12 work with a $d \times n$ integer matrix and assume $\mathbb{Z}A = \mathbb{Z}^d$ (section 2.1,
p. 2); Remark 2.1 (p. 3) extends everything to "the rank of $\mathbb{Z}A$ is $d$" by a change of
basis, the systems being equivalent. What is used:

- Def. 3.2 (p. 5): "A resonance center is a minimal face $F$ for which
  $\beta \in \mathbb{Z}A + \mathbb{C}F$."
- Def. 3.4 (p. 6): "$A$ is a(n iterated) pyramid over the face $F$ if $d = \dim_{\mathbb{Z}}(\mathbb{Z}A)$
  equals $|\bar F| + \dim_{\mathbb{Z}}(\mathbb{Z}F)$", where $\bar F$ is the set of columns off
  $F$.
- Lemma 3.5 (p. 6): $F$ is a face and $A$ is a pyramid over $F$ if and only if
  $a_j \notin \mathbb{Q}(A \setminus \{a_j\})$ for every $j \notin F$.
- Theorem 4.1 (p. 7): "Let $F$ be a resonance center for $\beta \in \mathbb{C}A$. If $A$ is not a
  pyramid over $F$ then $M_A(\beta)$ has reducible monodromy."

BGH26 quote the theorem as reducibility for any resonant face over which $A$ is not a pyramid
(p. 12), where their Theorem 3.1 is SW12's Theorem 4.1 (the number in the first arXiv version of
BGH26). For facets this follows from SW12 as they state it.

**Proposition 2.** Let $A$ have rank $d$, let $F$ be a facet resonant for $\beta$, and let at
least two columns lie off $F$. Then $M_A(\beta)$ has reducible monodromy.

*Proof.* The faces $G \subseteq F$ with $\beta \in \mathbb{Z}A + \mathbb{C}G$ form a finite set
containing $F$; let $G$ be a minimal one. A face contained in $G$ is contained in $F$, so $G$ is a
minimal face with this property among all faces, a resonance centre. The columns of $F$ span
$\mathbb{Z}F$ of rank $d - 1$, so by Def. 3.4 $A$ is a pyramid over $F$ exactly when one column
lies off $F$; here it is not. By Lemma 3.5 some $j \notin F$ has
$a_j \in \mathbb{Q}(A \setminus \{a_j\})$. Then $j \notin G$, and by Lemma 3.5 again $A$ is not a
pyramid over $G$. Theorem 4.1 applies to $G$. $\square$

`FacetResonance.reducible` is True when these hypotheses hold, at every $\varepsilon$ in
`resonant`, and None otherwise: when the facet is never resonant, when $A$ is a pyramid over it
(the centre inside it may or may not be a face over which $A$ is a pyramid, and SW12's Theorem 5.1,
p. 7, needs that of the centre itself), and when $A$ does not have full rank, where SW12's standing
assumption fails for the matrix as given. It is never False: irreducibility needs the resonance
centre, a question about faces of every dimension.

## The Schwinger-representation system

$T$ = `lp_to_cayley(N, L)`, the matrix of section 4.6 of the mathematics reference taking the
Lee-Pomeransky rows to the Cayley rows, is an integer matrix of determinant $\pm 1$: with its
columns in the order (ones, $\alpha_N$, $\alpha_1, \ldots, \alpha_{N-1}$) it is block triangular
with diagonal blocks $\bigl(\begin{smallmatrix} L+1 & -1 \\ -L & 1 \end{smallmatrix}\bigr)$ and
the identity, so its determinant is 1 there and $(-1)^{N-1}$ in feynkit's order. It maps
the Lee-Pomeransky columns onto the Cayley columns and
$T\beta_{\mathrm{LP}} = (\nu - (L+1)D/2,\ LD/2 - \nu,\ -\nu_1, \ldots, -\nu_{N-1})$, the Cayley
parameter. So $T$ carries $\mathbb{Z}A$, spans, faces and pyramids across, the functionals go to
$l_F T^{-1}$, and $l_F(\beta)$ is the same form on both sides. `facet_resonance(system="schwinger")`
does not rely on this: it computes the facets of the Cayley configuration from its own columns,
with `classify_configuration`, and uses $T$ only to write $\beta_{\mathrm{Cayley}}$ as forms in
$D$ and $\nu$. The tests compare the two classifications facet by facet.

The $\tilde F$ block is the face of the Cayley configuration on which the first row vanishes, and
corresponds under $T$ to $F_{\mathcal U}$, the facet whose points are the monomials of $F$
(BGH26 p. 46). When its columns span the hyperplane $y_0 = 0$, admissibility is
$\nu = (L+1)D/2$. `restrict_to_f_block` now calls `admissible` on the Cayley matrix and parameter
and warns when it fails; with $D$ a symbol it fails unless $\nu - (L+1)D/2$ vanishes identically.
The report builds the restriction without the warning and states instead whether $\beta$ lies in
the span, with $\nu = (L+1)D/2$ only as the condition for a block that spans $y_0 = 0$. A block of
lower rank asks more: the massless bubble's $\tilde F$ block is the one column $(0, 1, 1)$, and at
$D = 3$, $\nu = (1, 2)$ the Cayley parameter $(0, -3/2, -1)$ is not in its span although
$\nu = (L+1)D/2$.

## Interfaces

`feynkit/resonance.py`, not a core module:

- `EpsilonSet(kind, offset=None, period=None)`, frozen, with `eps in s` and `s.window(low, high)`.
- `ParameterForm(dimension, nu)`, frozen: the form $\text{dimension}\cdot D + \sum_e \nu[e]\,\nu_e$,
  with `at(nu, d0)`, the pair $(c, s)$, and `expression(D, exponents)`. It keeps $l_F(\beta)$ for all
  $D$ and $\nu$ at once, so that symbolic powers can later be classified from the same data.
- `FacetResonance`, frozen: `facet`, `functional`, `form`, `resonant`, `admissible`,
  `columns_off`, `reducible`, with the properties `kind`, `lattice_index`, `resonant_at_zero` and
  `pyramid`.
- `classify_facets(data, nu, d0=4)`: the facets of a `PolytopeData` for the Lee-Pomeransky
  $\beta$, reusing the facets already computed.
- `classify_configuration(a_matrix, beta, nu, d0=4)`: any homogeneous configuration, with $\beta$
  one `ParameterForm` per row.
- `span_epsilons(data, nu, d0=4)` and `lee_pomeransky_beta(n)`.
- `admissible(a_matrix, face, beta)`.

`FeynmanIntegral.facet_resonance(d0=None, *, nu=None, system="gkz")` classifies the facets of
`fi.gkz` (point indices are its columns, the $z_j$) or of `fi.schwinger_gkz` (its columns, $w$
then $z$). `choose_d0(d0, dimension)` fixes $D_0$ for it and for the report: `d0` when given,
otherwise the number $D_0$ when the dimension of the integral is $D_0 - 2\varepsilon$ in the symbol
named `epsilon`, the regulator of the parametric prefactors, and otherwise 4. So an integral built
with $D = 6 - 2\varepsilon$ is classified at its own $\varepsilon$. The powers are the integral's, which must then be integers, or `nu`, a mapping from
edge index to integer. `lp_to_cayley(n_edges, loop_count)` becomes public in
`feynkit.systems.cayley`.

## Report and CLI

A report section `resonance`, built by default: it reuses the polytope data of the other sections
and costs milliseconds. It states $D_0$ and where it comes from (given, read from the dimension of
the integral, or the default 4), the powers (the integral's when they are integers, otherwise 1 on
every edge, which it says), gives for each facet its inequality, $l_F(\beta)$ in $D$
and the $\nu_e$, the resonant set, the resonant values with $-1 \le \varepsilon \le 1$, whether
$\varepsilon = 0$ is resonant, where the facet is admissible and whether the system is reducible,
and below full dimension the values of $\varepsilon$ at which $\beta$ lies in the span of $A$.
`AnalysisReport.from_integral`, `to_latex` and `to_text` take `d0`. `fk analyse` gains the section
flag `-r`/`--resonance` and `--d0 VALUE`, an integer or a fraction such as `7/2`.

## Tests

- BGH26's statements with their pages: the bubble with two masses (Eqs. 63, 67-68, pp. 21-22), with
  $m_2 = 0$ (Eq. 105, p. 29), the triangle (Eqs. 119-120, pp. 31-32), the box as a massive
  $n$-gon (section 6.2, p. 37), the sunrise with three masses (Eq. 152, p. 41), two (p. 43) and
  one (p. 44), the banana (pp. 44-45), and $F_{\mathcal F}$, $F_{\mathcal U}$ at one, two and three
  loops (p. 46).
- $g_F > 1$ and the "never" class on $(0,0), (2,0), (0,1)$.
- Lower-dimensional polytopes: the massless on-shell bubble, which is scaleless, and `1ee|1|:zn`,
  whose affine hull passes through the origin.
- $D_0 = 3$ and $D_0 = 4$ on the bubble, $D_0 = 7/2$ on the banana.
- `fi.gkz` against `fi.schwinger_gkz`, facet by facet, on ten graphs from the bubble to the
  banana.
- $F_{\mathcal U}$ admissibility against `restrict_to_f_block`, with and without the warning.
- The massless bubble, a pyramid over every facet, has `reducible` None; on nine other graphs
  every facet has at least two columns off it, as BGH26 say of theirs (p. 12).

Classifying the facets of the massless pentagon, the massless hexagon or the massive kite takes
about 2 ms given their polytope data, which takes 5 to 25 ms; the Cayley classification, facets
included, takes 10 to 30 ms.

## Decisions

- Resonance is classified for facets only. Faces of lower dimension need a lattice test on
  several forms at once; BGH26 mention lower-dimensional resonant faces, such as intersections of
  edge facets, in their outlook (p. 51).
- The powers are integers. The forms keep $\nu$ symbolic, so that non-integer powers can be added
  as a new output later.
- Reducibility is reported only as Proposition 2 allows, and never as False.
- `admissible` does not check that the columns form a face.
