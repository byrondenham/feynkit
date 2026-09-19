# feynkit: Mathematics Reference

This document gives a self-contained account of every mathematical object and algorithm implemented in
the feynkit library. It is intended as a reference to be read alongside the primary
source papers, and as a dissertation reference. Each section states the definition, the precise formula
used in the code, the variable-naming conventions (symbol ↔ Python identifier), and the paper(s) to
consult for derivation or proof.

---

## 1. Graph-Theoretic Foundations

### 1.1 Feynman Graph

A **Feynman graph** $\Gamma$ consists of:

- $V$ **internal vertices**, indexed $1, \ldots, V$.
- $n$ **external legs** (half-edges), attached at external vertices $V+1, \ldots, V+n$.
- $E$ **internal edges**, each connecting two internal vertices or forming a self-loop.
  Each edge $e$ carries:
  - an index $e.\mathrm{idx} \in \mathbb{Z}_{>0}$ (unique);
  - a mass $m_e \geq 0$ (possibly zero — "massless");
  - a propagator exponent $\nu_e$ (default $1$; can be a symbol for IBP analysis).
- Each external leg $j$ carries momentum $p_j$; the energy scale $\mu$ normalises kinematic invariants.

**Loop count** (Euler characteristic of the graph):

$$L \;=\; E - V + 1.$$

This is the number of independent loop momenta. *Ref:* Weinzierl (2022), §2.

### 1.2 Schwinger Parameters

To each internal edge $e$ one assigns a **Schwinger parameter** $a_e \geq 0$.  In the code these are
SymPy symbols with names `a_{e.idx}` (prefix `"a"`, non-negative, real); the full dictionary is
`graph.schwinger_parameters`.

### 1.3 Vertex Indexing Convention

Internal vertices: $1, \ldots, V$ (stored in `edge.v1`, `edge.v2` for internal edges).
External vertices: $V+1, \ldots, V+n$ (stored in `edge.v2` for external legs; $\text{leg number} = v_2 - V$).

### 1.4 CNickel Notation

The **CNickel string** is a compact canonical encoding of a Feynman graph.

**Format:** `topology_string:mass_colors`

- **Topology string:** vertices are listed in order; for each vertex the higher-numbered neighbours
  and external legs (coded `e`) are concatenated, separated by `|`.
- **Mass color string:** one character per internal edge in the order they appear in the topology
  string: `z` or `0` = massless; `n` = individual symbolic mass $m_{e.\mathrm{idx}}$;
  digits `1`–`9` or letters `a`–`y` (except `n`, `s`) = shared labeled mass $m_c$; `s` = shared
  special mass $m_s$.

Examples: `"11e|e|:zz"` (massless bubble), `"12e|2e|e|:zzz"` (massless triangle),
`"12e|2e|e|:nzz"` (one-mass triangle).

---

## 2. Symanzik Polynomials

*Primary references:* Symanzik (1971); Weinzierl (2022) §2.2.

### 2.1 First Symanzik Polynomial $U$

Let $\mathcal{T}$ denote the set of **spanning trees** of the internal graph (subsets of edges that
connect all $V$ vertices with no cycles; each spanning tree has exactly $V-1$ edges).

The **spanning-tree polynomial** is
$$C(a) \;=\; \sum_{T \in \mathcal{T}} \prod_{e \in T} a_e.$$

The first Symanzik polynomial is the **complement** monomial-reversal:

$$U(a) \;=\; \Bigl(\prod_e a_e\Bigr)\cdot C\!\left(\tfrac{1}{a_1},\ldots,\tfrac{1}{a_E}\right)
\;=\; \sum_{T \in \mathcal{T}} \prod_{e \notin T} a_e.$$

Each monomial of $U$ is square-free of degree $L = E - V + 1$.

**Algorithm:** Enumerate all $\binom{E}{V-1}$ edge subsets, check acyclicity and connectivity via
union-find, multiply the complement parameters.  The reversal $e_i \mapsto 1 - e_i$ on exponent
vectors is performed without rational arithmetic by `_reverse_monomials` in `polynomials/symanzik.py`.

### 2.2 Second Symanzik Polynomial $F$

The second Symanzik polynomial splits into a kinematic part and a mass part:

$$F(a) \;=\; F_0(a) + U(a)\cdot\sum_e \frac{m_e^2}{\mu^2}\,a_e,$$

where the **kinematic part** sums over pairs of external legs:

$$F_0(a) \;=\; \sum_{j < k} \frac{p_j \cdot p_k}{\mu^2}
\cdot \Bigl(\prod_e a_e\Bigr)\cdot Q_{jk}\!\left(\tfrac{1}{a}\right).$$

Here $Q_{jk}$ is the **separating 2-forest polynomial**:

$$Q_{jk}(a) \;=\; \sum_{\substack{F \in \mathcal{F}: \\ v_j,v_k\,\text{separated}}} \prod_{e \in F} a_e,$$

where $\mathcal{F}$ is the set of spanning 2-forests (subsets of $V-2$ edges that split the graph into
exactly 2 connected components), restricted to those separating the internal vertex attached to leg $j$
from the vertex attached to leg $k$.

If two external legs are attached to the same internal vertex their contribution to $F_0$ vanishes.

Each monomial of $F_0$ is square-free of degree $L+1$.  Mass monomials $a_e \cdot (\text{U-monomial})$
have degree $L+1$ but are not generally square-free (degree 2 in $a_e$ possible when $e$ is a chord).

**Momentum products** in code: `graph.momentum_products` is a dict `{(j,k): p_j·p_k}`.  Default
symbols: `p{j}p{k}` (generic) or `s_{jk}/2` (Mandelstam). Energy scale `mu`; kinematic invariant
stored as `p_jk / mu**2`.

### 2.3 Lee-Pomeransky Polynomial $G$

The **Lee-Pomeransky polynomial** is simply

$$G(u) \;=\; U(u) + F(u),$$

where the Schwinger parameters have been renamed to **Lee-Pomeransky parameters** $u_e$ via the
substitution $a_e \to u_e$.  The code uses the prefix `"u"`: symbol name `u_{e.\mathrm{idx}}`,
non-negative, real; the list is `SymanzikPolynomials.lp_parameters`.

$G$ is a polynomial in $n = E$ variables of degrees $L$ (from $U$) and $L+1$ (from $F$), with
positive integer coefficients (for massless graphs with positive kinematic invariants).

*Ref:* Lee & Pomeransky (2013).

---

## 3. Parametric Representations

### 3.1 Schwinger Parametrisation

*Ref:* Schwinger (1951); Weinzierl (2022) §2.3.

$$\boxed{I_\Gamma \;=\; \frac{e^{L\varepsilon\gamma_E}}{\prod_e \Gamma(\nu_e)}
\int_0^\infty \prod_e \left(d\alpha_e\,\alpha_e^{\nu_e-1}\right)
U(\alpha)^{-D/2}\,\exp\!\left(-\frac{F(\alpha)}{U(\alpha)}\right).}$$

**Prefactor:** $e^{L\varepsilon\gamma_E}/\prod_e \Gamma(\nu_e)$.
**Measure:** $\prod_e \alpha_e^{\nu_e-1}$.
**Integrand:** $U^{-D/2}\exp(-F/U)$.
**Domain:** $\alpha_e \in (0,\infty)$ for all $e$.

In the code: `ParametrisationResult.prefactor`, `.measure`, `.integrand`; parameter symbols `a_{e.idx}`.

### 3.2 Feynman Parametrisation

*Ref:* Feynman (1949); Weinzierl (2022) §2.4.

$$\boxed{I_\Gamma \;=\; \frac{e^{L\varepsilon\gamma_E}\,\Gamma\!\left(\Sigma\nu - \tfrac{L\,D}{2}\right)}{\prod_e \Gamma(\nu_e)}
\int_\Sigma \prod_e \left(d x_e\,x_e^{\nu_e-1}\right)\delta\!\left(\sum x_e - 1\right)
\frac{U(x)^{\Sigma\nu-(L+1)D/2}}{F(x)^{\Sigma\nu-LD/2}}.}$$

where $\Sigma\nu = \sum_e \nu_e$.

**Domain:** Standard simplex $\Sigma = \{x_e \geq 0,\;\sum x_e = 1\}$.
**Prefactor:** $e^{L\varepsilon\gamma_E}\,\Gamma(\Sigma\nu - LD/2) / \prod_e\Gamma(\nu_e)$.
**Exponents:** $U^{\Sigma\nu - (L+1)D/2}$, $F^{-(\Sigma\nu - LD/2)}$.

### 3.3 Lee-Pomeransky Parametrisation

*Ref:* Lee & Pomeransky (2013); Weinzierl (2022) §2.5.

$$\boxed{I_\Gamma \;=\; \frac{e^{L\varepsilon\gamma_E}\,\Gamma(D/2)}{\Gamma\!\left((L+1)\tfrac{D}{2} - \Sigma\nu\right)\prod_e \Gamma(\nu_e)}
\int_0^\infty \prod_e \left(d u_e\,u_e^{\nu_e-1}\right) G(u)^{-D/2}.}$$

**Prefactor:** $e^{L\varepsilon\gamma_E}\,\Gamma(D/2) / \bigl[\Gamma((L+1)D/2 - \Sigma\nu)\prod_e\Gamma(\nu_e)\bigr]$.
**Measure:** $\prod_e u_e^{\nu_e-1}$.
**Integrand:** $G(u)^{-D/2}$.
**Domain:** $u_e \in (0,\infty)$.

The simplification to a single polynomial $G$ is the key advantage: it packages all analytic structure
into one object, making the GKZ structure transparent.

### 3.4 Symbol Conventions for Parametrisations

| Symbol | Name | Python identifier | Notes |
|--------|------|-------------------|-------|
| $\alpha_e, a_e$ | Schwinger parameter | `a_{e.idx}` | nonneg, real |
| $x_e$ | Feynman parameter | `alpha_{e.idx}` | nonneg, real, simplex |
| $u_e$ | Lee-Pomeransky parameter | `u_{e.idx}` | nonneg, real |
| $\nu_e$ | propagator exponent | `nu_{e.idx}` | positive, real |
| $D$ | spacetime dimension | `D` | positive, real |
| $\varepsilon$ | reg. parameter ($D = 4-2\varepsilon$) | `epsilon` | real |
| $\gamma_E$ | Euler-Mascheroni constant | `gamma_E` | real |
| $\mu$ | energy scale | `mu` | positive, real |
| $L$ | loop count | `fi.loop_count` | positive int |

---

## 4. GKZ A-Hypergeometric System

*Primary references:* GKZ (1989, 1994); de la Cruz (2019); Klausen (2020); Weinzierl (2022) §9.

### 4.1 The A-Matrix

Write the Lee-Pomeransky polynomial as a sum of $N$ monomials:

$$G(u) \;=\; \sum_{j=1}^{N} z_j\, u^{\alpha_j}, \qquad \alpha_j \in \mathbb{N}_0^n,$$

where $n = E$ (number of Lee-Pomeransky variables) and the $z_j$ are the (kinematic) coefficients.

The **GKZ A-matrix** is the $(n+1)\times N$ integer matrix

$$A \;=\; \begin{pmatrix} 1 & 1 & \cdots & 1 \\ \alpha_1 & \alpha_2 & \cdots & \alpha_N \end{pmatrix},$$

i.e.\ column $j$ is the vector $(1, \alpha_j^{(1)}, \ldots, \alpha_j^{(n)})^T \in \mathbb{Z}^{n+1}$.

The first (homogenising) row of all-ones encodes the grading structure.

**Column ordering** in feynkit: columns are sorted in descending graded-reverse-lex order —
first by total degree $\sum_i \alpha_j^{(i)}$ (descending), then lex on $(-\alpha_j^{(1)}, \ldots,
-\alpha_j^{(n)})$ — so the ordering is deterministic and reproducible.

The A-matrix is accessed as `fi.gkz.a_matrix` (a SymPy `ImmutableMatrix`).

### 4.2 Differential Variables and Euler Operators

Introduce formal differentiation variables $z_j$ (one per monomial of $G$), accessed as
`fi.gkz.z_variables`.  For each row $r = 0, \ldots, n$ of $A$, the **Euler (homogeneity) operator** is

$$\hat{E}_r \;=\; \sum_{j=1}^N A_{r,j}\, z_j\,\partial_j,$$

and the GKZ Euler equation is $\hat{E}_r \cdot \Phi = \beta_r\,\Phi$.

**Beta parameter vector:**

$$\beta \;=\; \bigl(-D/2,\;-\nu_1,\;-\nu_2,\ldots,-\nu_n\bigr)^T \in \mathbb{C}^{n+1}.$$

Each component is the homogeneity weight of the integral $\int u^{\nu-1} G^{-D/2}\,du$ under the
rescaling encoded by the corresponding row of $A$: scaling every coefficient $z_j \to \lambda z_j$
multiplies $G^{-D/2}$ by $\lambda^{-D/2}$, and scaling $z_j \to \lambda^{\alpha_j^{(i)}} z_j$ is
undone by $u_i \to u_i/\lambda$, which costs $\lambda^{-\nu_i}$ from the measure.  This is the
vector $\kappa = (-d/2, -\alpha)$ of de la Cruz (2019), eq. (fintegral-A-hypergeometric), and the
system $\langle A\theta + \underline{\nu}\rangle$ with $\underline{\nu} = (d/2, \nu)$ of Klausen
(2020), Thm 3.1.  Versions of feynkit before 0.3.0 used $(\Sigma\nu - D/2, \nu_1, \ldots, \nu_n)$,
which does not satisfy the Euler equations.

Accessed as `fi.gkz.beta_parameters` and `fi.gkz.euler_equations`.

### 4.3 Toric Operators

The GKZ ideal $H_A(\beta)$ also contains the **toric operators**

$$\partial^u - \partial^v \quad \text{for all } u,v \in \mathbb{N}_0^N \text{ with } Au = Av.$$

These are the generators of the toric ideal $I_A$ (see §6); together with the Euler operators they
generate the full GKZ D-module. *Ref:* GKZ (1994) §3; SST (2000) §3.

### 4.4 Proof That Feynman Integrals Are A-Hypergeometric

**Theorem** (de la Cruz 2019, Thm 1; Klausen 2020, Thm 3.1):
*The generalised Feynman integral $I_A(\nu, z)$ — with $z_j$ promoted to formal indeterminates — is
annihilated by the GKZ system $H_A(\beta)$ with $\beta$ as above.*

**Proof sketch:** Toric relations follow from homogeneity of the integrand under simultaneous rescaling
$z_j \to s^{a_j^{(r)}} z_j$; Euler relations follow from differentiation in $s$.

### 4.5 Holonomic Rank and Master Integral Count

$$\operatorname{rank} H_A(\beta) \;=\; \mathrm{vol}_0\!\bigl(\Delta_G\bigr) \quad\text{for very generic } \beta,$$

where $\mathrm{vol}_0$ is the **normalised (lattice) volume** of the Newton polytope $\Delta_G$
(see §5.3).  For physical (integer or half-integer) $D$ the rank can be lower — a "rank jump" —
corresponding to linear relations among master integrals.

The Lee-Pomeransky count of master integrals (Euler characteristic) satisfies
$\chi \leq \mathrm{vol}_0(\Delta_G)$, with equality for generic kinematics.

*Ref:* GKZ (1994) Thm 3.11; Klausen (2020) Thm 2.2; de la Cruz (2019) §2.

---

## 5. Newton Polytope

*Ref:* GKZ (1994) §5–6; Klausen (2020) §2.

### 5.1 Definition

The **Newton polytope** of $G$ is the convex hull of the exponent vectors of its monomials:

$$\Delta_G \;=\; \operatorname{Conv}\!\bigl(\{\alpha_j : j=1,\ldots,N\}\bigr) \;\subset\; \mathbb{R}^n.$$

Equivalently: the columns of the A-matrix (minus the homogenising row) are the lattice points; the
Newton polytope is their convex hull.

Accessed as `fi.newton_polytope`; the full monomial support is
`fi.newton_polytope.support` (list of `(exponent_vector, coefficient)` pairs);
hull vertices are `fi.newton_polytope.points`.

### 5.2 Monomial Support and Hull

The **monomial support** of $G$ is the finite set $\mathcal{A} = \{\alpha_1, \ldots, \alpha_N\} \subset \mathbb{N}_0^n$.
Not all support points need be vertices of $\Delta_G$; interior lattice points also occur (e.g.\ for
massive banana graphs).

`fi.newton_polytope.support` preserves all $N$ points including interior ones;
`fi.newton_polytope.points` returns only the hull vertices.

### 5.3 Normalised Volume

The **normalised (lattice) volume** of a full-dimensional polytope $P \subset \mathbb{R}^n$ is

$$\mathrm{vol}_0(P) \;=\; n!\,\mathrm{Vol}(P),$$

where $\mathrm{Vol}$ is ordinary Euclidean volume.  For a lattice simplex $\sigma$ with vertices
$v_0,\ldots,v_n$, $\mathrm{vol}_0(\sigma) = |\det(v_1-v_0, \ldots, v_n-v_0)|$.

For a general convex lattice polytope, $\mathrm{vol}_0$ equals the sum of $\mathrm{vol}_0$ over any
triangulation into primitive simplices (each of unit normalised volume).

$\mathrm{vol}_0(\Delta_G)$ equals the GKZ holonomic rank for generic $\beta$.

Accessed as `fi.newton_polytope.normalised_volume` (via `AConfiguration.normalized_volume`).

### 5.4 Smith Normal Form and Intrinsic Lattice Model

Let $\mathrm{diffs} = (\alpha_2 - \alpha_1, \ldots, \alpha_N - \alpha_1)^T$ be the $(N-1)\times n$
difference matrix.  Its **Smith normal form** is

$$D \;=\; U \cdot \mathrm{diffs} \cdot V, \qquad U \in GL_{N-1}(\mathbb{Z}),\; V \in GL_n(\mathbb{Z}),$$

where $D$ is diagonal with non-negative entries $d_1 | d_2 | \cdots | d_r$ (the **Smith invariants**).

The Smith invariants classify the sublattice spanned by the configuration inside $\mathbb{Z}^n$:
the index is $\prod_i d_i$ and is 1 if and only if the configuration spans $\mathbb{Z}^n$.

**Intrinsic lattice model:** Choose an affine basis $W$ consisting of $r$ linearly independent rows of
$\mathrm{diffs}$ (where $r = $ affine dimension).  Express every support point in this basis:

$$\mathrm{intrinsic\_coords}_j \;=\; W^{-1}(\alpha_j - \alpha_1) \;\in\; \mathbb{Z}^r.$$

This embeds the configuration canonically in $\mathbb{Z}^r$, stripping away the ambient $\mathbb{Z}^n$
embedding.

Accessed via `AConfiguration.smith_invariants`, `AConfiguration.intrinsic_model`.

---

## 6. Toric Ideal

*Ref:* Cox–Little–O'Shea (2015) Ch.\ 11; Sturmfels (1996); de la Cruz (2019) §2.3.

### 6.1 Definition

With $z$-variables $z_1, \ldots, z_N$ (one per monomial of $G$), the **toric ideal** is

$$I_A \;=\; \ker(\phi) \;\subset\; \mathbb{C}[z_1,\ldots,z_N],$$

where $\phi: z_j \mapsto t^{A_{:,j}} = t_1^{A_{1,j}}\cdots t_{n+1}^{A_{n+1,j}}$ is the toric
parametrisation.

Equivalently:

$$I_A \;=\; \bigl\langle z^u - z^v : u,v \in \mathbb{N}_0^N,\; Au = Av \bigr\rangle,$$

i.e.\ $I_A$ is generated by **binomials** $z^u - z^v$ for all integer vectors in the lattice
$L = \ker_{\mathbb{Z}} A$.

Accessed as `fi.toric_ideal`; generators listed in `fi.toric_ideal.generators` (SymPy polynomials in
the $z$-variables `fi.toric_ideal.z_variables`).

### 6.2 Generators: Markov Basis vs.\ Gröbner Basis

A **Markov basis** is a minimal generating set of $I_A$.  It can be smaller than a Gröbner basis and
is computed by the external tool `4ti2` (binary `markov`).  A **Gröbner basis** is computed by
Buchberger's algorithm (SymPy); for large examples it can have more generators than a Markov basis.

feynkit chooses automatically between the two backends (`backend="auto"` in
`compute_toric_ideal_generators`): `4ti2` if available, else SymPy.

### 6.3 Physical Interpretation (IBP Relations)

Each generator $z^u - z^v \in I_A$ encodes an **integration-by-parts (IBP) identity** among the
monomials of $G$: the toric relation $Au = Av$ means the two monomials $z^u$ and $z^v$ have the same
image under $\phi$, so they represent the same combination of propagator factors after the change of
variables, and their difference vanishes in the Feynman integral.  The number of independent generators
(modulo the obvious degree-one relations) equals $N - \operatorname{rank}(A)$, the codimension of the
toric variety.

---

## 7. Polytope Automorphisms

*Ref:* Liu & Cai (2025) arXiv:2506.23846; Grinis & Kasprzyk (2013) arXiv:1301.6641.

### 7.1 Unimodular Automorphisms

A **unimodular automorphism** of the Newton polytope $\Delta_G$ is a pair $(U, t)$ with

$$U \in GL_n(\mathbb{Z}),\quad |\det U| = 1,\quad t \in \mathbb{Z}^n,$$

such that $\{U\alpha + t : \alpha \in \mathcal{A}\} = \mathcal{A}$ (the map is a bijection on the
support points).

The set of all such pairs forms the **unimodular automorphism group** $\mathrm{Aut}(\Delta_G)$.

Accessed as `fi.polytope_automorphisms`; order `fi.polytope_automorphisms.order`;
vertex permutations `fi.polytope_automorphisms.vertex_permutations`;
vertex orbits `fi.polytope_automorphisms.vertex_orbits`.

### 7.2 Liu-Cai Vertex Labels

For each hull vertex $v$ of the support, define the **Liu-Cai label**

$$\ell(v) \;=\; \det\!\left(\sum_{w \sim v} (w - v)(w - v)^T\right) \;\in\; \mathbb{Z},$$

where the sum runs over all hull vertices $w$ adjacent to $v$ in the 1-skeleton of $\Delta_G$.

**Invariance:** For any unimodular map $(U,t)$, $\ell(Uv + t) = \ell(v)$.  This follows because the
outer-product matrix transforms as $A_{Uv+t} = U A_v U^T$, so $\det A_{Uv+t} = (\det U)^2 \det A_v = \det A_v$.

Labels are used to filter candidate basis points, reducing the search from $O(N^n)$ brute force to
$O(|\mathrm{Aut}| \cdot n!)$ in typical cases.

### 7.3 Algorithm

1. Compute hull vertices and their Liu-Cai labels.
2. Select a **label-diverse basis**: $r = \mathrm{affine\_dim}$ vertices from the hull, choosing rarest
   labels first, such that the difference matrix is invertible.
3. For each hull vertex $w$ with the same label as the base anchor $\alpha_0$, attempt it as the image
   anchor.
4. For each label-preserving ordered $r$-tuple of remaining vertices (generated via
   `_label_preserving_orderings`), solve for $U = \Delta_{\mathrm{target}}\Delta_{\mathrm{src}}^{-1}$
   and check integrality.
5. Verify that $U$ is a bijection on the full support (not just hull vertices).
6. Deduplicate by $(U, t)$.

Implemented in `feynkit/normal_forms/polytope_automorphisms.py`.

### 7.4 Known Examples

| Graph | $|\mathrm{Aut}(\Delta_G)|$ | Group |
|-------|--------------------------|-------|
| Massless bubble | 6 | $S_3$ |
| One-mass bubble | 8 | Kummer (eightfold $\,_2F_1$) |
| Massless triangle | 48 | $B_3$ (hyperoctahedral) |
| $L$-loop massless banana | $(L+1)!$ | $S_{L+1}$ |
| Massless box | 120 | $S_5$ |

---

## 8. Symmetry Pairs

*Ref:* Forsgård–Matusevich–Sobieska (FMS, 2019) arXiv:1703.03036; de la Cruz (2024).

### 8.1 Definition

A **symmetry pair** (or A-configuration self-map) of the GKZ configuration $\mathcal{A}$ is a triple
$(M, t, P)$ where

- $M \in GL_n(\mathbb{Z})$ is an invertible integer linear map,
- $t \in \mathbb{Z}^n$ is a translation vector,
- $P$ is a permutation of $\{1, \ldots, N\}$,

such that $M \alpha_j + t = \alpha_{P(j)}$ for every $j$.

In **homogenised form**, define

$$T \;=\; \begin{pmatrix} 1 & \mathbf{0}^T \\ t & M \end{pmatrix} \;\in\; GL_{n+1}(\mathbb{Z}).$$

The defining condition becomes $T A = A \Pi_P$, where $\Pi_P$ is the $N\times N$ column permutation
matrix of $P$.

Accessed as `fi.symmetry_pairs` (list of `SymmetryPair` objects); each has `.linear_map` ($M$),
`.translation` ($t$), `.column_permutation` ($P$), `.determinant` ($|\det M|$),
`.is_unimodular`, `.homogenized_map` ($T$).

### 8.2 Transformation Identity for Feynman Integrals

**Theorem** (FMS 2019, Thm 4.2; de la Cruz 2024):
*For each symmetry pair $(M,t,P)$ of $\mathcal{A}$, the generalised Feynman integral satisfies*

$$I_A(\beta, z) \;=\; I_A(T\beta,\; z_P),$$

*where $z_P = (z_{P(1)},\ldots,z_{P(N)})$ is the permuted kinematic vector and the prefactor $R(\beta) = 1$.*

Specialising $z$ to physical kinematic values gives a concrete functional identity between two
(possibly equal) Feynman integrals at different values of $\beta$.  The transformation
$\beta \mapsto T\beta$ is computed by `SymmetryPair.transform_beta(beta)`.

### 8.3 Unimodularity of All Self-Maps

For a Feynman configuration (non-degenerate: affine span = $\mathbb{Z}^n$), every integer affine
self-map is automatically **unimodular** ($|\det M| = 1$).

**Proof:** $M$ maps the configuration $\mathcal{A}$ bijectively to itself, so it maps
$\operatorname{Conv}(\mathcal{A})$ to itself with the same volume.  Therefore
$|\det M| = \mathrm{Vol}(M\cdot\Delta)/\mathrm{Vol}(\Delta) = 1$.

Configurations with non-trivial Smith invariants (e.g.\ BMS simplex with $d_n = 2$) can admit
non-unimodular **finite-index** maps between two *different* configurations (see §9.4).

### 8.4 Connection to Classical Hypergeometric Identities

The symmetry pairs of the one-mass bubble ($N=4$ monomials, $n=2$ variables) are exactly the 8
elements of the Kummer group of $\,_2F_1$ (quadratic transformations of Gauss's hypergeometric
function).  The 48 pairs of the massless triangle are the hyperoctahedral group $B_3$, reflecting
the cross-polytope structure of the Newton polytope.

*Ref:* de la Cruz (2024) §3.1, §4.

---

## 9. Unimodular and Affine Equivalence

*Ref:* Liu & Cai (2025); GKZ (1994) §6.

### 9.1 Unimodular Equivalence (Liu-Cai)

Two configurations $\mathcal{A} \subset \mathbb{Z}^n$ and $\mathcal{B} \subset \mathbb{Z}^n$ are
**unimodularly equivalent** if there exists $(U, t)$ with $U \in GL_n(\mathbb{Z})$, $|\det U|=1$,
$t \in \mathbb{Z}^n$, such that $\{U\alpha + t : \alpha \in \mathcal{A}\} = \mathcal{B}$ (as sets).

This is strictly stronger than affine isomorphism over $\mathbb{Q}$: the map must preserve the integer
lattice.

**Physical meaning:** Two Feynman integrals with unimodularly equivalent Newton polytopes have the same
GKZ system up to a coordinate change in $z$-space (relabelling of monomials) and a reparametrisation
$u \mapsto Mu + t$ of the integration variables — they are the *same* A-hypergeometric function.

Accessed via `fi.is_unimodular_equivalent_to(other)`, which returns a `PolytopeEquivalence` with
fields `.equivalent` (bool), `.witness_map` ($U$), `.translation` ($t$), `.determinant`, `.vertex_correspondence`.

The algorithm is the Liu-Cai basis-search (same as automorphism computation, but between two
configurations): build labelled polytope graphs, enumerate MST isomorphisms, solve for $U$.

### 9.2 Affine Equivalence

Two configurations are **affinely equivalent** (over $\mathbb{Q}$) if there exists $M \in GL_n(\mathbb{Q})$
and $t \in \mathbb{Q}^n$ mapping one to the other.  This is a weaker relation; some GKZ structural
properties (holonomic rank, series solutions) are invariant only under unimodular equivalence.

Accessed via `fi.is_affinely_equivalent_to(other)`.  The algorithm checks equivalence of the hull
vertices only (not the full monomial support).

### 9.3 Point-Configuration Equivalence

Checks unimodular equivalence of the full monomial support (all $N$ points including interior ones),
not only the hull.  Two integrals can have the same Newton polytope but different monomial supports
(different sets of interior points), giving different GKZ systems with the same Feynman polytope.

Accessed via `AConfiguration.is_point_config_equivalent_to(other)`.

### 9.4 Finite-Index Maps

A **finite-index map** $(M, t)$ with $M \in GL_n(\mathbb{Z})$, $|\det M| = k > 1$, maps source
configuration $\mathcal{A}$ bijectively to target $\mathcal{B}$.  The integer $k = |\det M|$ is the
**index** of the sublattice.

Physical meaning: $\mathcal{B}$ lies in a sublattice of index $k$ in $\mathbb{Z}^n$; the map bridges
two configurations with different Smith invariants.

Example: The massless triangle (Smith invariants $[1,1,1]$) maps to the triple-K integral (Smith
invariants $[1,1,2]$) via $M = \bigl(\begin{smallmatrix}0&1&1\\1&0&1\\1&1&0\end{smallmatrix}\bigr)$
with $\det M = 2$.  Both have 6 monomials and $B_3$ automorphism group, but they are NOT unimodularly
equivalent.

Accessed via `AConfiguration.finite_index_map_to(other)`, returning a `FiniteIndexResult`.

---

## 10. Landau Singularities via Principal A-Determinant

*Ref:* GKZ (1994) §10; Klausen (2022).

### 10.1 Principal A-Determinant and Its Edge Part

The **principal A-determinant** $E_A(G)$ is a polynomial in the coefficients of $G$ whose zero locus
encodes all singularities of the GKZ system (including Landau singularities of the Feynman integral).

The **edge part** $E_A^{(1)}(G)$ — the physically relevant piece — is the product of discriminants
over the **1-faces (edges)** of the Newton polytope:

$$E_A^{(1)}(G) \;=\; \prod_{\tau \;\text{edge of}\; \Delta_G} \Delta_{A_\tau}\!\left(G\big|_\tau\right).$$

Here $G|_\tau$ is the restriction of $G$ to the edge $\tau$: keep only the monomials $\alpha_j \in \tau$,
viewed as a univariate polynomial in the primitive edge direction.

### 10.2 Edge Discriminant

For a univariate polynomial $P(t) = c_0 t^{e_0} + c_1 t^{e_1} + \cdots + c_s t^{e_s}$ (not
necessarily with consecutive exponents), the **discriminant** is

$$\Delta(P) \;=\; \frac{\operatorname{Res}(P, P')}{\operatorname{lc}(P)^{\deg P - 1}},$$

where $P' = dP/dt$, $\operatorname{lc}(P)$ is the leading coefficient, and $\operatorname{Res}$ is the
resultant.  The zero locus of $\Delta(P)$ (viewed as a polynomial in the kinematic variables $z$) is
the set of kinematics at which the restricted polynomial $G|_\tau$ has a repeated root — the
**Landau surface** associated with edge $\tau$.

### 10.3 Algorithm

1. Compute the monomial support of $G(u;\,\text{kinematics})$.
2. Find all 1-faces (edges) of $\Delta_G$ using `ConvexHull`: two hull vertices form an edge iff they
   share at least $\dim\Delta_G - 1$ facets.  Include all lattice points on each closed segment.
3. For each edge $\tau$ with $\geq 2$ distinct lattice points:
   a. Compute the primitive direction $v_\tau \in \mathbb{Z}^n$ of the edge.
   b. Express the restriction $G|_\tau(t) = \sum_{j:\,\alpha_j\in\tau} c_j \cdot t^{\langle\alpha_j, v_\tau\rangle}$
      as a polynomial in the scalar $t$ (with kinematic coefficients $c_j$).
   c. Compute the discriminant $\Delta(G|_\tau)$.
   d. Retain it if it depends on the kinematic symbols (discard purely numerical factors).
4. Factor the product of retained discriminants into irreducible kinematic factors.
5. Return these irreducible factors as the **Landau surfaces**.

### 10.4 Known Results for Standard Topologies

- **Massive bubble:** $\Delta_G$ has one edge; $G|_\tau = c_1 t + c_2$, degenerate; discriminant is
  the threshold $(s - (m_1 + m_2)^2)$ (normal threshold).
- **Massless triangle:** Three external legs at zero mass; $G|_\tau = p_i^2 \cdot t^{e_1} + \ldots$
  for each edge; discriminant factors give $p_i^2 = 0$ (IR singularities).
- **BMS simplex ($n$ points):** Edge restrictions give $p_i^2 = 0$ for each external momentum — a
  novel result for the CFT correlation function integral, not previously catalogued.

Accessed via `fi.landau` (module `feynkit/landau.py`); returns `LandauAnalysis` with
`.edge_discriminants`, `.landau_polynomial`, `.landau_surfaces`.

---

## 11. Conformal and BMS Configurations

*Ref:* Bzowski–McFadden–Skenderis (2021) arXiv:2008.07543; Caloro (2024); feynkit `artifacts/conformal.py`.

### 11.1 Massless $n$-gon $C_n$

The **massless $n$-gon** is the 1-loop graph with $n$ internal edges and $n$ external legs (a cycle).
It has $n$ Lee-Pomeransky variables $u_1,\ldots,u_n$, loop count $L=1$, and

$$G_{C_n}(u) \;=\; U(u) + F_0(u),$$

with $U$ and $F_0$ each having $n$ monomials.  The A-matrix is $(n+1)\times 2n$.

For $n = 3$ (massless triangle): $2n = 6$ monomials, $\Delta_G$ is a cross-polytope in $\mathbb{R}^3$
with automorphism group $B_3$.  For $n \geq 4$: $|\mathrm{Aut}| = 2$ (only trivial symmetries remain).

### 11.2 BMS $n$-Point Simplex

The **BMS simplex** integral arises from the $n$-point scalar contact Witten diagram in a holographic
CFT.  Starting from

$$I_n \;\propto\; \int_0^\infty r^{\beta_0-1}\prod_i K_{\nu_i}(p_i r)\,dr,$$

using the Bessel-function integral $K_\nu(pr) = (p/2)^\nu/(2) \int_0^\infty u^{-\nu-1}e^{-r(u + p^2/(4u))}du$
and integrating over $r$, then multiplying through by $4\prod_j u_j$, gives the Lee-Pomeransky form
with

$$G_{\mathrm{BMS}_n}(u) \;=\; \sum_{i=1}^n p_i^2 \prod_{j\neq i} u_j \;+\; 4\sum_{i=1}^n u_i^2 \prod_{j\neq i} u_j.$$

This has $2n$ monomials: $n$ **lower** monomials of degree $n-1$ (exponent vectors: 0 in position $i$,
1 elsewhere) and $n$ **upper** monomials of degree $n+1$ (exponent vectors: 2 in position $i$, 1
elsewhere).

Key properties:
- Smith invariants: $[1, 1, \ldots, 1, 2]$ (index-2 sublattice in $\mathbb{Z}^n$).
- Holonomic rank: $\mathrm{vol}_0(\Delta_{\mathrm{BMS}}) = 2^{n-1}$.
- For $n=3$: reproduces the triple-K integral; same Newton polytope as massless triangle up to the
  sublattice index.

Accessed via `bms_simplex_a_config(n)` in `feynkit/artifacts/conformal.py`.

### 11.3 Conformal Companion

The **conformal companion** $\mathcal{A}_n^{\mathrm{comp}}$ is a configuration designed to admit a
finite-index map to BMS$_n$.  Its $G$-polynomial is

$$G_{\mathrm{comp}}(u) \;=\; \sum_{i=1}^n \prod_{j\neq i} u_j \;+\; \sum_{i=1}^n p_i^2\, u_i.$$

Lower monomials have degree $n-1$ (same as BMS lower); upper monomials have degree 1.  The A-matrix is
$(n+1) \times 2n$.

The finite-index map $M: \mathcal{A}^{\mathrm{comp}} \to \mathcal{A}^{\mathrm{BMS}}$ has $|\det M| = 2$
for all $n$ and $|\det M| = 1$ (unimodular) only for $n=3$.

For $n=3$: the companion coincides with the massless triangle (same A-configuration).

Accessed via `conformal_companion_a_config(n)` in `feynkit/artifacts/conformal.py`.

### 11.4 Complete Graph $K_n$

The complete graph $K_n$ has $V=n$ internal vertices, $\binom{n}{2}$ internal edges, and $n$ external
legs (one per vertex).  Loop count $L = \binom{n}{2} - n + 1$.  Key cases:
- $K_3$ = massless triangle ($L=1$, 3 edges, 6 monomials).
- $K_4$ = massless tetrahedron graph ($L=4$, 6 edges, 31 monomials; studied extensively in BMS context).

Accessed via `complete_graph_a_config(n)`.

---

## 12. Variable Naming Conventions

This table maps every mathematical symbol to the corresponding Python identifier in feynkit.

### 12.1 Graph and Parameters

| Math symbol | Meaning | Python / feynkit identifier |
|-------------|---------|----------------------------|
| $V$ | number of internal vertices | `graph.internal_vertices` |
| $E$ | number of internal edges | `len(graph.get_internal_edges())` |
| $n$ (= $E$) | number of integration variables | `len(fi.symanzik.lp_parameters)` |
| $L$ | loop count | `fi.loop_count` |
| $e.\mathrm{idx}$ | edge index | `edge.idx` |
| $m_e$ | edge mass | `edge.get_mass()` (SymPy expr) |
| $\nu_e$ | propagator exponent | `fi.propagator_exponents[e.idx]` |
| $a_e$ | Schwinger parameter | `a_{e.idx}` (SymPy `Symbol`) |
| $u_e$ | Lee-Pomeransky parameter | `u_{e.idx}` (SymPy `Symbol`) |
| $D$ | spacetime dimension | `fi.dimension` (symbol `D`) |
| $\varepsilon$ | reg.\ parameter | symbol `epsilon` |
| $\gamma_E$ | Euler-Mascheroni | symbol `gamma_E` |
| $\mu$ | energy scale | `graph.energy_scale` (symbol `mu`) |
| $p_j \cdot p_k$ | momentum product | `fi.momentum_products[(j,k)]` |

### 12.2 Symanzik Polynomials

| Math symbol | Python / feynkit identifier |
|-------------|----------------------------|
| $U(a)$ | `fi.symanzik.u` |
| $F(a)$ | `fi.symanzik.f` |
| $G(u) = U(u)+F(u)$ | `fi.symanzik.g` |
| $U(u)$ (LP form) | `fi.symanzik.u_lp` |
| $F(u)$ (LP form) | `fi.symanzik.f_lp` |

### 12.3 GKZ System

| Math symbol | Python / feynkit identifier |
|-------------|----------------------------|
| $A$ | `fi.gkz.a_matrix` (SymPy `ImmutableMatrix`) |
| $N$ | `fi.gkz.a_matrix.cols` |
| $n+1$ | `fi.gkz.a_matrix.rows` |
| $z_j$ | `fi.gkz.z_variables[j-1]` |
| $\beta$ | `fi.gkz.beta_parameters` (list) |
| $\beta_0$ | `fi.gkz.beta_parameters[0]` |
| $\beta_i$ ($i\geq 1$) | `fi.gkz.beta_parameters[i]` |
| Euler equations | `fi.gkz.euler_equations` |

### 12.4 Newton Polytope

| Math symbol | Python / feynkit identifier |
|-------------|----------------------------|
| $\Delta_G$ | `fi.newton_polytope` |
| $\{\alpha_j\}$ (support) | `fi.newton_polytope.support` (list of `(tuple, coeff)`) |
| Hull vertices | `fi.newton_polytope.points` (numpy array) |
| $\mathrm{vol}_0(\Delta_G)$ | `AConfiguration(A).normalized_volume` |
| Smith invariants | `AConfiguration(A).smith_invariants` |
| Affine dimension | `AConfiguration(A).affine_dim` |

### 12.5 Automorphisms, Symmetry Pairs, Equivalence

| Math symbol | Python / feynkit identifier |
|-------------|----------------------------|
| $\mathrm{Aut}(\Delta_G)$ | `fi.polytope_automorphisms` |
| $|\mathrm{Aut}|$ | `fi.polytope_automorphisms.order` |
| Symmetry pair $(M,t,P)$ | `fi.symmetry_pairs[k]` (a `SymmetryPair`) |
| $M$ (linear part) | `pair.linear_map` (SymPy matrix) |
| $t$ (translation) | `pair.translation` (SymPy column vector) |
| $P$ (permutation) | `pair.column_permutation` (tuple of ints) |
| $T$ (homogenised) | `pair.homogenized_map` |
| $|\det M|$ | `pair.determinant` |
| Unimodular? | `pair.is_unimodular` |
| $T\beta$ | `pair.transform_beta(beta)` |
| Unimodular equivalence | `fi.is_unimodular_equivalent_to(other)` |
| Affine equivalence | `fi.is_affinely_equivalent_to(other)` |
| Finite-index map | `AConfiguration(A).finite_index_map_to(other)` |

### 12.6 Toric Ideal

| Math symbol | Python / feynkit identifier |
|-------------|----------------------------|
| $I_A$ | `fi.toric_ideal` |
| Generators | `fi.toric_ideal.generators` |
| $z_j$ (toric vars) | `fi.toric_ideal.z_variables` |

### 12.7 Landau Analysis

| Math symbol | Python / feynkit identifier |
|-------------|----------------------------|
| $E_A^{(1)}(G)$ | `fi.landau.landau_polynomial` |
| Per-edge discriminants | `fi.landau.edge_discriminants` (list of `EdgeDiscriminant`) |
| Landau surfaces | `fi.landau.landau_surfaces` |
| Edge exponent vectors | `edge_disc.edge_exponents` |
| Edge kinematic coefficients | `edge_disc.edge_coefficients` |

---

## Bibliography

All papers cited in the feynkit source and directly relevant to the implemented analyses.

1. **GKZ (1989).** I.M. Gelfand, M.M. Kapranov, A.V. Zelevinsky.
   *Hypergeometric functions and toric varieties.*
   Funct.\ Anal.\ Appl.\ **23** (1989) 94–106.

2. **GKZ (1994).** I.M. Gelfand, M.M. Kapranov, A.V. Zelevinsky.
   *Discriminants, Resultants, and Multidimensional Determinants.*
   Birkhäuser, 1994.

3. **SST (2000).** M. Saito, B. Sturmfels, N. Takayama.
   *Gröbner Deformations of Hypergeometric Differential Equations.*
   Springer, 2000.

4. **Schwinger (1951).** J. Schwinger.
   *On gauge invariance and vacuum polarization.*
   Phys.\ Rev.\ **82** (1951) 664.

5. **Feynman (1949).** R.P. Feynman.
   *Space-time approach to quantum electrodynamics.*
   Phys.\ Rev.\ **76** (1949) 769.

6. **Symanzik (1971).** K. Symanzik.
   Commun.\ Math.\ Phys.\ **18** (1971) 227–246.

7. **Lee–Pomeransky (2013).** R.N. Lee, A.A. Pomeransky.
   *Critical points and number of master integrals.*
   JHEP **11** (2013) 165.

8. **de la Cruz (2019).** L. de la Cruz.
   *Feynman integrals as A-hypergeometric functions.*
   JHEP **12** (2019) 123.  arXiv:1907.00507.

9. **Klausen (2020).** R.P. Klausen.
   *Hypergeometric series representations of Feynman integrals by GKZ hypergeometric systems.*
   JHEP **04** (2020) 121.  arXiv:1910.08651.

10. **Klausen (2022).** R.P. Klausen.
    *Kinematic singularities of Feynman integrals and principal A-determinants.*
    JHEP **02** (2022) 004.

11. **FMS (2019).** J. Forsgård, L.F. Matusevich, A. Sobieska.
    *On transformations of A-hypergeometric functions.*
    Funkcialaj Ekvacioj **62** (2019) 319.  arXiv:1703.03036.

12. **de la Cruz (2024).** L. de la Cruz.
    *Polytope symmetries of Feynman integrals.*  arXiv:2406.xxxxx (2024).

13. **Grinis–Kasprzyk (2013).** R. Grinis, A.M. Kasprzyk.
    *Normal forms of convex lattice polytopes.*  arXiv:1301.6641.

14. **Liu–Cai (2025).** Q. Liu, Z. Cai.
    *On the Unimodular Isomorphism Problem of Convex Lattice Polytopes.*  arXiv:2506.23846.

15. **BMS (2021).** A. Bzowski, P. McFadden, K. Skenderis.
    *Conformal correlators as simplex integrals in momentum space.*
    JHEP **01** (2021) 192.  arXiv:2008.07543.

16. **Caloro (2024).** D. Caloro.
    *Shift operators and momentum-space conformal field theory.*
    PhD thesis (2024).

17. **ABP (2017).** N. Arkani-Hamed, P. Benincasa, A. Postnikov.
    *Cosmological Polytopes and the Wavefunction of the Universe.*  arXiv:1709.02813.

18. **Weinzierl (2022).** S. Weinzierl.
    *Feynman Integrals: A Comprehensive Treatment for Students and Researchers.*
    Springer, 2022.  arXiv:2201.03593.

19. **Cox–Little–O'Shea (2015).** D. Cox, J. Little, D. O'Shea.
    *Ideals, Varieties, and Algorithms.* 4th ed., Springer, 2015.

20. **Sturmfels (1996).** B. Sturmfels.
    *Gröbner Bases and Convex Polytopes.* AMS, 1996.

21. **Bitoun et al.\ (2019).** T. Bitoun, C. Bogner, R.P. Klausen, E. Panzer.
    *Feynman integral relations from parametric annihilators.*
    Lett.\ Math.\ Phys.\ **109** (2019) 497–564.
