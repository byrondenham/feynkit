# Graphs of the faces

Design note for a module `feynkit/face_identification.py` that names each face of the Newton
polytope of $\mathcal G = \mathcal U + \mathcal F$ by a product of Symanzik polynomials of minors of
the graph; for the minors it needs, `Graph.contract`, `Graph.delete` and
`feynkit.polynomials.minor_polynomials`; for `FeynmanIntegral.face_identification`; and for the
faces section of the analysis report and `fk analyse -f`.

References: Fevola, Mizera and Telen, Principal Landau determinants, Comput. Phys. Commun. 303
(2024) 109278, arXiv:2311.16219 (FMT24 below); Britto, Grimm and Hoefnagels, Resonance and
differential reduction of Feynman integrals, JHEP 09 (2026) 018, arXiv:2606.09978 (BGH26); and
Arkani-Hamed, Hillman and Mizera, Feynman polytopes and the tropical geometry of UV and IR
divergences, Phys. Rev. D 105 (2022) 125013, arXiv:2202.12296 (AHM22). Page numbers are those
printed on the pages of the arXiv versions.

## Purpose

`polytope_data` lists the faces of the Newton polytope $P$ as sets of monomials. The literature
names them by graphs: FMT24 label faces by subgraphs $\gamma$ with initial forms
$\mathcal U_\gamma\,\mathcal G_{\Gamma/\gamma}$ (Eq. 3.15, p. 28), BGH26 identify the face of the
terms free of $x_e$ with the contraction $\Gamma/e$ (Eq. 21, p. 11, for a one-particle-irreducible graph), and AHM22 describe the facets
of the Feynman polytope by ultraviolet and infrared subgraphs (Eqs. 7-8, p. 3). This note gives one
rule that names every face this way when it can, checks each name exactly, and says when it
cannot.

## Conventions

$\mathcal G$ is written in the Lee-Pomeransky parameters $u_e$, one per internal edge, with feynkit's
$\mathcal F$, which carries the factor $1/\mu^2$ (section 2 of the mathematics reference). A facet of
a full-dimensional $P$ is $m \cdot x \le b$ with $m$ primitive and outward (section 5.5 there).

**Minors.** For disjoint sets $C$ and $D$ of edges, $(\Gamma - D)/C$ is `graph.delete(D).contract(C)`.
Contraction identifies the two ends of each edge of $C$ and removes the edge, so a set with a cycle
contracts each of its components to one vertex, as in FMT24's $G/\gamma$ (p. 28); an edge outside
$C$ whose ends are identified becomes a self-loop. Legs move with their vertices, and a vertex left
without edges is removed with its legs. Edges keep their indices, so a minor's polynomials are
written in the parameters of $\Gamma$, and its legs keep their momenta. Deletion and contraction of
disjoint sets commute, since in either order the minor has the edges outside $C \cup D$ on the
vertices of $\Gamma/C$ they touch; a minor need not be connected.

For a minor $H$ with components $c$ we write $\mathcal U_H = \prod_c \mathcal U_c$ and
$\mathcal F_H = \sum_c \mathcal F_c \prod_{c' \ne c} \mathcal U_{c'}$, where $\mathcal U_c$ and
$\mathcal F_c$ are feynkit's polynomials of the component with the legs at its vertices. Expanding,
$\mathcal U_H$ sums the complement monomials of the maximal forests of $H$ (one tree per
component) and $\mathcal F_H$ those of the forests with one component more, with feynkit's
coefficients over $\mu^2$: the products $p_j \cdot p_k$ of the legs on the two sides of the split
component and the squared masses of its edges joining them, together with the terms
$m_e^2 u_e\,u^{E_H \setminus T}/\mu^2$ for a maximal forest $T$ and an edge $e \notin T$. For a
connected graph these are $\mathcal U$ and $\mathcal F$ of `FeynmanIntegral.symanzik`, which the
tests check on seven graphs.

## 1. Faces as initial forms

For $w \in \mathbb{Z}^N$, $\mathrm{in}_w(\mathcal G)$ is the sum of the terms $c_\alpha u^\alpha$
with $w \cdot \alpha$ least. Its Newton polytope is the face of $P$ on which $w$ is least (FMT24,
p. 28). So $\mathcal G|_F$, the sum of the terms whose exponents lie on $F$, is
$\mathrm{in}_w(\mathcal G)$ for every $w$ least exactly on $F$.

**Lemma 1.** Let $P$ be full-dimensional and $F$ a face of `polytope_data`. With
$w_F = -\sum m$ over the facets $m \cdot x \le b$ containing $F$ ($w_P = 0$), $F$ is the set of
points of $P$ on which $w_F$ is least.

*Proof.* Each facet gives $-m \cdot x \ge -b$ on $P$, with equality exactly on the facet. Summing
over the facets containing $F$, $w_F \cdot x \ge -\sum b$ on $P$, with equality exactly on the
intersection of those facets. `polytope_data` builds its faces as the non-empty intersections of
facets (section 5.5 of the mathematics reference), so that intersection is $F$. $\square$

## 2. The flag and the initial form of U

Let $t_1 > \dots > t_k$ be the distinct values of $w_e$, $\sigma_0 = \emptyset$,
$\sigma_j = \{e : w_e \ge t_j\}$ and $H_j = (\Gamma - (E \setminus \sigma_j))/\sigma_{j-1}$: the edges
of weight $t_j$, with those of greater weight contracted and the others deleted. For a set $S$ of
edges let $r(S)$ be the number of edges of a maximal forest of $(V, S)$, the vertex count minus the
number of components.

**Lemma 2.** Let $S \subseteq E$, $B$ a maximal forest of $(V, S)$ and $X \subseteq E \setminus S$.
Then $B \cup X$ is a forest if and only if $X$ is a forest of $\Gamma/S$, a self-loop counting as a
cycle. Hence the maximal forests of $H_j$ have $r(\sigma_j) - r(\sigma_{j-1})$ edges.

*Proof.* $\Gamma/S$ identifies the vertices of each component of $(V, S)$, which are those of
$(V, B)$. A cycle of $B \cup X$ uses an edge of $X$, since $B$ is a forest. Following it and
replacing each stretch in $B$ by the vertex of $\Gamma/S$ it lies in gives a closed walk of
$\Gamma/S$ that uses those edges of $X$ once each, and a closed walk using an edge once contains a
cycle through it. Conversely, a cycle of $X$ in $\Gamma/S$ passes through each identified vertex by
entering and leaving at two vertices of one component of $(V, B)$, which a path of $B$ joins; these
paths and the edges of the cycle form a closed walk of $B \cup X$ using each of those edges once, so
$B \cup X$ has a cycle. For the count, take $S = \sigma_{j-1}$: a forest $X$ of $H_j$ gives the
forest $B \cup X$ of $(V, \sigma_j)$, so $|X| \le r(\sigma_j) - r(\sigma_{j-1})$, and a maximal
forest of $(V, \sigma_j)$ containing $B$ gives an $X$ of that size. Deleting the edges outside
$\sigma_j$ and the vertices left without edges changes no forest. $\square$

**Proposition 2.** For a connected graph,
$\mathrm{in}_w(\mathcal U_\Gamma) = \prod_{j=1}^k \mathcal U_{H_j}$.

*Proof.* $\mathcal U_\Gamma = \sum_T u^{E \setminus T}$ over the spanning trees $T$, and the
$w$-degree of $u^{E \setminus T}$ is $w(E) - w(T)$, with $w(S) = \sum_{e \in S} w_e$. Summing by
parts,
$$w(T) = \sum_{j=1}^{k} t_j\,|T \cap (\sigma_j \setminus \sigma_{j-1})|
= \sum_{j=1}^{k-1} (t_j - t_{j+1})\,|T \cap \sigma_j| + t_k\,|T|.$$
Here $|T| = |V| - 1$, $t_j - t_{j+1} > 0$ and $|T \cap \sigma_j| \le r(\sigma_j)$, so $w(T)$ is
greatest exactly when $T \cap \sigma_j$ is a maximal forest of $(V, \sigma_j)$ for every $j$, and
such trees exist, by extending a maximal forest of each $\sigma_j$ to one of $\sigma_{j+1}$. By
Lemma 2 applied at each level, these trees are exactly the unions $T = T_1 \cup \dots \cup T_k$ with
$T_j$ a maximal forest of $H_j$, each such tuple giving one tree. Then
$u^{E \setminus T} = \prod_j u^{(\sigma_j \setminus \sigma_{j-1}) \setminus T_j}$, and the sum over
the tuples is $\prod_j \mathcal U_{H_j}$. $\square$

## 3. The prediction

Let $j^*$ be the last level with $\mathcal F_{H_j} \ne 0$. The prediction is
$$\mathcal G|_F \overset{?}{=} \prod_{j \ne j^*} \mathcal U_{H_j} \times \begin{cases}
\mathcal G_{H_{j^*}} & t_{j^*} = 0, \\ \mathcal F_{H_{j^*}} & t_{j^*} < 0, \\
\mathcal U_{H_{j^*}} & t_{j^*} > 0, \end{cases}$$
or $\prod_j \mathcal U_{H_j}$ when no level has $\mathcal F_{H_j} \ne 0$. It is motivated, not proved:
the terms of $\mathcal U$ of least degree have degree $w(E) - W$, with $W$ the greatest $w(T)$; a
spanning forest with one tree more that is maximal on every level but one misses one edge of a
greatest tree, and has degree $w(E) - W + t_j$ at the level $j$ of that edge; the lowest level
contributes the least degree, $\mathcal F$ and $\mathcal U$ tie when it is 0, and $\mathcal F$ alone
survives when it is negative. What the argument does not control is the coefficient: the momentum
between the two trees of a forest depends on the edges of every level, while $\mathcal F_{H_j}$
sees only the legs at the vertices of $H_j$. feynkit therefore compares the prediction with
$\mathcal G|_F$ term by term, coefficients expanded exactly, and identifies the face only when the
two are equal. Every identification is a verified identity of polynomials, and its factors are the
Symanzik polynomials of the named minors.

The classes follow the flag of a verified face:

| Class | Condition | Example |
|-------|-----------|---------|
| `whole` | $F = P$ | $\mathcal G_\Gamma$ |
| `contraction` | one $\mathcal G$ factor, on $\Gamma/S$, the others 1 | $\mathcal G_{\Gamma/\{3\}}$ |
| `product_uv` | the $\mathcal G$ factor on $\Gamma/S$, another factor not 1 | $\mathcal U_{\{3,4\}}\,\mathcal G_{\Gamma/\{3,4\}}$ |
| `product_ir` | the $\mathcal G$ factor on a minor with edges deleted | $\mathcal G_{\{1\}}\,\mathcal U_{\Gamma/\{1\}}$ |
| `u_layer` | no $\mathcal G$ or $\mathcal F$ factor | $\mathcal U_\Gamma$ |
| `f_layer` | an $\mathcal F$ factor | $\mathcal F_\Gamma$ |

A face that is not verified is a `support_product` or `unidentified`, as the next subsection
decides.

A factor is 1 exactly when it is $\mathcal U$ of a minor without loops, $|E_H| - |V_H| + c_H = 0$.
A lone $\mathcal G$ factor on a minor with deleted edges, the other factors 1, counts as
`product_ir`. The report names a factor by its minor $\sigma_j/\sigma_{j-1}$, writing $\Gamma$ for
all the edges and a set of edges for the subgraph they span.

### Support products

A face that is not verified can still have exactly the exponents of the predicted product. Let
$A_j$ be the support of the polynomial that level $j$ contributes to the prediction. The face is a
`support_product` when its set of exponents is
$$A_1 + \dots + A_k = \{a_1 + \dots + a_k : a_j \in A_j\},$$
and `unidentified` when it is not. The levels have disjoint sets of edges, so each point of the sum
arises from one choice of the $a_j$; the sum is therefore the support of the predicted product, with
no term cancelling, and $\mathrm{Newt}(\mathcal G|_F) = \prod_j \mathrm{Newt}(A_j)$ in the
coordinates of the levels. What fails is only the coefficients: $\mathcal G|_F$ is not the product
of the factors. feynkit compares the two point sets exactly, records the outcome in
`support_verified`, true for every verified face and every support product, and keeps both
polynomials. A face verified as a product is never a support product, so no class of section 3
changes; the new class takes faces that were unidentified.

The massless box with $p_2^2 = 0$ has four support products among its faces: the facet
$x_1 + x_3 \le 1$ of section 4 and three faces of codimension 2 inside it. On that facet the flag
predicts $\mathcal G_{\{2,4\}}\,\mathcal U_{\Gamma/\{2,4\}}$; $\mathcal G|_F$ has the six exponents
of that product, but its $\mathcal F$ coefficients form a matrix of rank 2, and $\mathcal G|_F$,
linear in $u_1, u_3$, factors only if that matrix has rank 1, so it is irreducible. The massless
double box with $p_i^2 = 0$ has 22 support products among the 932 faces the flag does not
identify, of codimension 5 to 7, four of them vertices whose one term has the predicted exponent
and another coefficient; the massless box with $p_i^2 = 0$ has none. The check costs one
set of sums per face.

## 4. The published cases

**FMT24.** For a connected subgraph $\gamma \ne E$ and every mass non-zero,
$\mathrm{in}_{w_\gamma}(\mathcal G_\Gamma) = \mathcal U_\gamma\,\mathcal G_{\Gamma/\gamma}$, with
$w_\gamma$ 1 on $\gamma$ and 0 elsewhere (Eqs. 3.13 and 3.15, pp. 27-28). When the face selected
by $w_\gamma$ is a facet, its primitive inward normal is $w_\gamma$, so $w_F = w_\gamma$; the flag is
$\gamma \subset E$ with $H_1 = \gamma$ and $H_2 = \Gamma/\gamma$, whose masses make
$\mathcal F_{\Gamma/\gamma} \ne 0$, and the prediction is $\mathcal U_\gamma\,\mathcal G_{\Gamma/\gamma}$.
On the parachute with four masses the nine facets have FMT24's nine rays as inward normals (p. 29),
and each of the eight with $w_\gamma \ge 0$ is $\mathcal U_\gamma\,\mathcal G_{\Gamma/\gamma}$: the
single edges as contractions, $\{3,4\}$, $\{1,2,3\}$ and $\{1,2,4\}$ as `product_uv`, and
$\gamma = E$ as the $\mathcal U$ layer $\mathcal U_\Gamma$, $\Gamma/\Gamma$ being a point.

**BGH26.** The face $F_e$ of the terms free of $u_e$ gives $\mathcal G_{\Gamma/e}$ (Eq. 21,
p. 11, for a one-particle-irreducible graph). If $F_e$ is a facet, its inequality is $u_e$'s exponent at least 0, so
$w_F = w_{\{e\}}$ and the flag is $\{e\} \subset E$. The edge $e$ is not a self-loop, since a
self-loop lies outside every forest and $F_e$ would be empty, so
$\mathcal U_{H_1} = 1$. If $\mathcal F_{\Gamma/e} \ne 0$ the prediction is
$\mathcal G_{\Gamma/e}$; otherwise it is $\mathcal U_{\Gamma/e} = \mathcal G_{\Gamma/e}$. The tests
check every edge facet of four graphs, the bubble with $m_2 = 0$, where $F_1$ is not a facet and the
new facet $F_{2,(1,2)}$ (Eq. 105, p. 29) is $\mathcal G_{\{1\}}\,\mathcal U_{\Gamma/\{1\}}$, and the
sunrise with $m_3 = 0$, where only $F_3$ remains a facet (p. 43).

**AHM22.** AHM22 describe the facets of the Feynman polytope
$\mathbf U_G \oplus c\,\mathbf F_G$, a Minkowski sum, by subgraphs $\gamma$, with
$\mathcal F_{G/\gamma} \ne 0$ for ultraviolet facets and $\mathcal F_{G/\gamma} = 0$ for infrared
ones (Eqs. 4, 7 and 8, pp. 2-3). feynkit's polytope is $\mathrm{Newt}(\mathcal U + \mathcal F)$, and it
uses their examples as checks rather than their statements. The massive sunrise has eight facets:
the two layers and six others, three of them the ultraviolet bubbles
$\mathcal U_{\{i,j\}}\,\mathcal G_{\Gamma/\{i,j\}}$, as in their App. B (Eq. B14, p. 9). On their
three-mass box, feynkit's box with $p_2^2 = 0$, the facet $x_1 + x_3 \le 1$ has the flag
$\{2,4\} \subset E$, their $\gamma_{14}$, and its $\mathcal G|_F$ is their restricted $\mathcal U$ and
$\mathcal F$ (Eqs. B24-B25, p. 10). Its $\mathcal F$ part has the four terms $u_iu_j$ with
$i \in \{2,4\}$, $j \in \{1,3\}$ and coefficients $-p_1^2$, $-s_{12}$, $-(p_1 + p_3)^2$ and $-p_4^2$
over $\mu^2$, a matrix of rank 2, so it is no product of a polynomial in $u_2, u_4$ and one in
$u_1, u_3$. The face is a support product (section 3).

## 5. Scope

- **Below full dimension.** A facet relative to the affine hull has a normal fixed only modulo the
  equations of the hull, so $w_F$, and the flag, depend on a choice. Every scaleless integral is in
  this case. feynkit identifies no face there and says so in `reason`.
- **One name per face.** Another weight least on $F$ can give another flag with the same product,
  so the name is canonical only through $w_F$.
- **Faces not identified.** Support products and unidentified faces are recorded with
  $\mathcal G|_F$ and the prediction side by side, so that the report can show both; no search for
  other factorisations is made.
- **Codimension.** `identify_faces` takes the faces up to codimension 2 by default and every face
  with `max_codimension=None`.

## 6. Interfaces

- `Graph.contract(edges)`, `Graph.delete(edges)`: the minors of the Conventions; an index that is
  not an internal edge raises `ValidationError`, and a minor without edges `GraphTopologyError`.
- `minor_polynomials(graph, momentum_products, *, contract=(), delete=(), parameters=None)`:
  $\mathcal U_H$ and $\mathcal F_H$ in the parameters of `graph`, the Schwinger $a_e$ by default.
- `identify_faces(fi, *, max_codimension=2, data=None)`: one frozen `FaceIdentification` per face,
  by codimension, the facets in the order of `PolytopeData.facets`, with `point_indices`,
  `dimension`, `codimension`, `facet`, `weight`, `levels` (frozen `FlagLevel`s with `edges`,
  `contracted`, `deleted`, `weight`, `kind` and `loops`), `kind`, `contracted`, `verified`,
  `polynomial`, `prediction`, `reason`, `support_verified` and `name()`.
- `FeynmanIntegral.face_identification(max_codimension=2)`, cached for each argument.
- The report section `faces`, built by default: the facets with their inequalities and names, a
  support product marked, the number of faces of each class and codimension, and up to ten faces
  that are not identified with $\mathcal G|_F$ and the prediction. `fk analyse` prints the facets and
  counts by default, and `-f` alone; `-f` also adds the section to a report whose `--sections`
  leave it out.

## 7. Cost

The minors of a flag are cached by their edge sets, and polynomials are kept as dictionaries of
exponent vectors, so that a prediction is a product of dictionaries in disjoint variables and the
comparison expands one coefficient difference per term. On one core, with the polytope data given:

| Graph | Faces up to codim. 2 | Time | All faces | Time |
|-------|---------------------:|-----:|----------:|-----:|
| massless pentagon `12e|3e|4e|4e|e|:zzzzz` | 58 | 0.08 s | 213 | 0.12 s |
| massless hexagon `12e|3e|4e|5e|5e|e|:zzzzzz` | 78 | 0.14 s | 519 | 0.37 s |
| massive kite `12e|23|3|e|:nnnnn` | 50 | 0.08 s | 213 | 0.10 s |
| massless double box `15e|24|3e|4e|5|e|:zzzzzzz` | 154 | 0.28 s | 2245 | 1.5 s |

Computing the polytope data takes 0.07 to 0.3 s on these graphs, so the default costs about as
much again.

## 8. Tests

The minors: deletion-contraction for $\mathcal U$, Kirchhoff's spanning-tree counts and loop
numbers over nine graphs with parallel edges, self-loops and bridges; contraction and deletion
commute (a property test). The polynomials: $\mathcal U$ and $\mathcal F$ of the whole graph against
`FeynmanIntegral.symanzik`, BGH26's Eq. 21 on every edge of seven graphs, FMT24's Eq. 3.13 on the
massive parachute, and the forest convention. The identifier: the published cases of section 4;
every verified face equals the product of its level polynomials computed afresh; the class counts
do not change when the edges are relabelled (a property test); and, as a regression, the massless
box with $p_i^2 = 0$ keeps its 20 unidentified faces of dimension at least 1. The support products:
the four of the box with $p_2^2 = 0$, whose exponents are checked against sums computed afresh
from the factors, the 22 of the double box, none on the box with $p_i^2 = 0$, and, on seven graphs,
the class of every face identified as a product unchanged from before the class existed.
