# Automorphism Groups of Newton Polytopes of Feynman Integrals

## 1. Introduction

The Lee–Pomeransky polynomial $G(u; z)$ encodes a Feynman integral as a GKZ
hypergeometric function: up to a Gamma-function prefactor,

$$
I(z) = \int_{\mathbb{R}^n_{>0}} \prod_i u_i^{\nu_i - 1} \, G(u; z)^{-\nu_0} \, du,
$$

where $z = (z_1, \ldots, z_m)$ are the kinematic parameters (masses and
momentum invariants) appearing as coefficients of the monomials of $G$, and
$\nu_0$ is a combination of propagator exponents and the spacetime dimension.
The Newton polytope $P = \mathrm{Newt}(G) \subset \mathbb{R}^n$ controls
the combinatorics of the GKZ $D$-module associated with $I(z)$: the holonomic
rank — the number of master integrals in the IBP sense — equals the normalised
volume $\mathrm{vol}(P)$, and the set of GKZ exponents is read off from
the face structure of $P$.

A unimodular automorphism of $P$ is a lattice-preserving bijection $P \to P$.
Any such map $\sigma$ permutes the monomials of $G$ and, if it also permutes
them in a coefficient-preserving way, yields a functional equation

$$
I(z) = I(\sigma \cdot z),
$$

where $\sigma \cdot z$ is the rearrangement of kinematic parameters induced by
the monomial permutation. These symmetry relations are the polytope-automorphism
analogues of Ward identities: they collapse the parameter space and can reduce
the number of independent kinematic configurations that must be evaluated
numerically.

feynkit computes these automorphism groups exactly, via the Liu–Cai algorithm
applied to the Newton polytope of $G$.

---

## 2. Mathematical Background

### The Lee–Pomeransky polynomial

Let $\Gamma$ be a connected Feynman graph with $n$ internal edges, $L$ loops,
and $E$ external legs. Assign Lee–Pomeransky parameters $u_1, \ldots, u_n$ to
the internal edges. The Symanzik polynomials $\mathcal{U}$ (first kind) and
$\mathcal{F}$ (second kind) are given by sums over spanning trees and
2-trees of $\Gamma$, respectively. The Lee–Pomeransky polynomial is

$$
G(u) = \mathcal{U}(u) + \mathcal{F}(u),
$$

a polynomial in $u_1, \ldots, u_n$ whose coefficients are rational functions of
the kinematic variables (masses $m_i^2$ and Mandelstam invariants $s_{ij}$).

### The Newton polytope

The Newton polytope is the convex hull of the exponent vectors of all monomials
in $G$:

$$
P = \mathrm{Newt}(G) = \mathrm{conv}\bigl\{ \alpha \in \mathbb{Z}^n : c_\alpha \neq 0 \bigr\} \subset \mathbb{R}^n,
$$

where $G = \sum_\alpha c_\alpha \, u^\alpha$. The GKZ A-matrix is the matrix
whose columns are the homogenised exponent vectors $(1, \alpha)^T$; it encodes
all the operator equations (box operators, Euler operators) satisfied by $I(z)$.

### Unimodular automorphisms

A unimodular automorphism of $P$ is a pair $(U, t)$ with $U \in \mathrm{GL}_n(\mathbb{Z})$,
$|{\det U}| = 1$, $t \in \mathbb{Z}^n$, such that

$$
\{U p + t : p \in P\} = P.
$$

The collection of all such pairs under composition forms the unimodular
automorphism group $\mathrm{Aut}(P)$.

The unimodularity condition $|\det U| = 1$ is essential, not merely a
technical convenience. An affine map over $\mathbb{Q}$ that sends $P$ to itself
need not map $\mathbb{Z}^n$ to itself. A unimodular map preserves the lattice
$\mathbb{Z}^n$ and therefore preserves the full combinatorial structure of the
GKZ $D$-module: the A-matrix is sent to a column-permutation of itself, the
toric ideal is preserved, and the holonomic rank is unchanged. Rational affine
maps outside $\mathrm{GL}_n(\mathbb{Z})$ do not have this property.

### The Liu–Cai algorithm

feynkit locates $\mathrm{Aut}(P)$ using the basis-search strategy of
Liu and Cai (arXiv:2506.23846), originally introduced for deciding unimodular
*equivalence* between two distinct polytopes. The key idea is that any
automorphism $(U, t)$ is determined by the image of a single basis: fix a
vertex $v_0 \in P$ and an affinely independent set of $n$ neighbouring vertices
$\{v_1, \ldots, v_n\}$ forming a basis for the translations. The image of $v_0$
can be any vertex $v_0'$ with the same combinatorial label (degree sequence in
the polytope graph), and the images of $v_1, \ldots, v_n$ are constrained by
the requirement that $U$ is integer-valued and unimodular. The algorithm
enumerates all such candidate bases, filters by integrality and determinant, and
verifies each candidate against all vertices. Unlike the equivalence test, every
valid witness is retained rather than stopping at the first.

---

## 3. Three Levels of Symmetry

feynkit distinguishes three nested levels of symmetry, each carrying different
physical information.

### Level 1: Graph automorphisms

A graph automorphism of $\Gamma$ is a permutation of its internal vertices that
preserves both the edge topology and the mass assignment (massless edges mapped
to massless, massive to massive). For a graph with $V$ internal vertices,
`compute_graph_automorphisms` exhausts all $V!$ vertex permutations and retains
those for which the relabelled adjacency structure coincides with the original.

Each such vertex permutation $\sigma \in S_V$ induces a permutation matrix
$P_\sigma \in \mathrm{GL}_n(\mathbb{Z})$ acting on the $n$-dimensional parameter
space by permuting the rows of $G$. Consequently, $\mathrm{Aut}(\Gamma)$
embeds as a subgroup of $\mathrm{Aut}(P)$.

For single-edge-type graphs (all edges of the same mass character), graph
automorphisms are the familiar permutation symmetries of the propagators: the
massless bubble has $\mathbb{Z}/2$ (swap the two edges) and the massless
triangle has $S_3$ (permute the three edges). For multi-edge graphs such as
banana diagrams, however, vertex permutations alone miss the symmetry that
permutes the multiple parallel edges between the same pair of vertices; the
banana has only two internal vertices, giving $|\mathrm{Aut}(\Gamma)| = 2$
regardless of the number of propagators.

### Level 2: Newton polytope automorphisms

The full group $\mathrm{Aut}(P)$ always contains $\mathrm{Aut}(\Gamma)$
as a subgroup (via the embedding above) and can be strictly larger. The extra
elements are unimodular maps that are not induced by vertex permutations of the
graph. They arise when the lattice geometry of $P$ has more symmetry than the
graph-theoretic structure of $\Gamma$.

The most striking example is the banana family. The 3-propagator banana has two
internal vertices and hence $|\mathrm{Aut}(\Gamma)| = 2$, yet its Newton
polytope is the standard 3-simplex in $\mathbb{R}^3$, which carries the full
symmetric group $S_4$ acting by permutation of the four vertices — so
$|\mathrm{Aut}(P)| = 24$.

### Level 3: Coefficient-preserving automorphisms

A map $(U, t) \in \mathrm{Aut}(P)$ is coefficient-preserving if it sends
each monomial $c_\alpha u^\alpha$ of $G$ to another monomial with the same
coefficient:

$$
c_\alpha = c_{U\alpha + t} \quad \text{for all } \alpha \in \mathrm{supp}(G).
$$

This is the physically significant subgroup: when $(U, t)$ is
coefficient-preserving, the rearrangement $z \mapsto \sigma \cdot z$ of
kinematic parameters leaves $G$ invariant and therefore gives the exact
functional equation $I(z) = I(\sigma \cdot z)$.

With generic symbolic kinematics (all masses and Mandelstam invariants
independent symbols), each monomial of $G$ typically has a distinct coefficient,
so only the identity is coefficient-preserving. At special kinematic points —
most notably when all masses are zero — many coefficients coincide and the
coefficient-preserving subgroup can be as large as the full $\mathrm{Aut}(P)$.
For massless diagrams, all monomials of $\mathcal{U}$ and all monomials of
$\mathcal{F}$ carry integer coefficients determined purely by spanning-tree
counts; when these counts happen to be equal across all monomials (as they are
for bananas), the coefficient-preserving group is the full $\mathrm{Aut}(P)$.

---

## 4. Results for Standard Diagrams

The following table records feynkit-computed values for a selection of standard
one- and two-loop diagrams. CNickel strings use the convention
`topology:mass-coloring` where `z` denotes a massless edge and `n` a massive edge.

| Diagram | CNickel | $|\mathrm{Aut}(P)|$ | $|\mathrm{Aut}(\Gamma)|$ | Coeff-pres. |
|---|---|---|---|---|
| Massless bubble | `11e\|e\|:zz` | 6 ($S_3$) | 2 | 2 |
| Massive bubble | `11e\|e\|:nn` | 1 | 2 | 1 |
| One-mass bubble | `11e\|e\|:nz` | 8 | 2 | 1 |
| Massless triangle | `12e\|2e\|e\|:zzz` | 48 ($B_3$) | 6 | 1 |
| One-mass triangle | `12e\|2e\|e\|:nzz` | 6 ($S_3$) | 2 | 1 |
| Two-mass triangle | `12e\|2e\|e\|:nnz` | 1 | 2 | 1 |
| All-mass triangle | `12e\|2e\|e\|:nnn` | 1 | 6 | 1 |
| Massless box | `12e\|3e\|3e\|e\|:zzzz` | 120 ($B_4$) | 8 | 1 |
| One-mass box | `12e\|3e\|3e\|e\|:nzzz` | 24 ($S_4$) | 2 | 1 |
| Massless 3-banana | `111e\|e\|:zzz` | 24 ($S_4$) | 2 | 6 |
| Massive 3-banana | `111e\|e\|:nnn` | 6 ($S_3$) | 2 | 1 |
| Massless 4-banana | `1111e\|e\|:zzzz` | 120 ($B_4$) | 2 | 24 |

Several patterns are immediately visible. First, introducing masses breaks
symmetry: the massless triangle has $|\mathrm{Aut}(P)| = 48$ whereas
adding one mass reduces it to 6 and adding a second to 1. Second, the
coefficient-preserving order for the massless bananas equals the factorial of
the number of propagators: $6 = 3!$ for the 3-banana and $24 = 4!$ for the
4-banana. This reflects the fact that in the massless case every monomial of
$G$ has coefficient 1, so the full $\mathrm{Aut}(P)$ that permutes the
simplex vertices preserves coefficients — and the edge-permutation symmetry
missing from $\mathrm{Aut}(\Gamma)$ is recovered at Level 3.

---

## 5. The Banana Family

The $n$-propagator banana graph has $n$ parallel internal edges connecting two
external vertices and contributes one loop ($L = n - 1$ for a connected banana
with $n$ edges through a single loop; for the standard banana with 2 external
vertices and $n$ internal edges, $L = n - 1$). Its Lee–Pomeransky polynomial in
the massless case is

$$
G(u_1, \ldots, u_n) = (u_1 + u_2 + \cdots + u_n) + p^2 \, u_1 u_2 \cdots u_n / (\cdots),
$$

and the Newton polytope is the standard $(n-1)$-simplex
$\Delta_{n-1} = \mathrm{conv}(e_1, \ldots, e_n) \subset \mathbb{R}^n$
together with the vertex corresponding to $\prod_i u_i$ — i.e. a simplex in
$n$ dimensions with $n+1$ vertices.

The automorphism group of the standard $n$-simplex (as a lattice polytope) is
the symmetric group $S_{n+1}$ permuting its $n+1$ vertices, giving
$|\mathrm{Aut}(P)| = (n+1)!$. This matches the computed values:
$|\mathrm{Aut}(P)| = 24 = 4! = |S_4|$ for the 3-banana and
$|\mathrm{Aut}(P)| = 120 = 5! = |S_5|$ for the 4-banana.
For massive bananas the coefficients of $G$ are no longer uniform and the
symmetry is reduced, in agreement with the table above.

---

## 6. Physical Interpretation

The holonomic rank of the GKZ $D$-module equals the normalised volume
$\mathrm{vol}(P)$, which is a unimodular invariant. Two Feynman integrals
whose Newton polytopes are unimodularly equivalent therefore have the same
number of master integrals; a polytope automorphism provides a concrete
isomorphism between their GKZ systems.

Coefficient-preserving automorphisms go further: they produce exact functional
equations $I(z) = I(\sigma \cdot z)$ that hold as identities of analytic
functions, not merely as equalities of dimension counts. At a special kinematic
point where the coefficient-preserving group has order $k$, the kinematic
parameter space is reduced by a factor of $k$: any numerical evaluation of $I$
at $k$ related kinematic configurations can be recycled from a single evaluation.
This is directly useful in multi-loop computations where $I$ is numerically
expensive.

The vertex orbits of $\mathrm{Aut}(P)$ partition the monomials of $G$ into
equivalence classes under the full symmetry group. When all monomials in an
orbit share the same coefficient, the orbit contributes a single kinematic
parameter to the coefficient-preserving subgroup. One can therefore read off the
coefficient-preserving order directly from the orbit structure together with the
coefficient multiset, without rerunning the search.

---

## 7. feynkit API

```python
from feynkit import FeynmanIntegral
from feynkit.normal_forms.polytope_automorphisms import (
    compute_polytope_automorphisms,
    compute_graph_automorphisms,
    coefficient_preserving_indices,
)

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")

# Full polytope automorphism group (cached property on FeynmanIntegral)
auts = fi.polytope_automorphisms       # PolytopeAutomorphisms
print(auts.order)                      # 48
print(auts.vertex_orbits)             # [[0, 1, 2, 3, 4, 5]] — single orbit

# Graph automorphisms (vertex permutations only)
gauts = fi.graph_automorphisms         # list of vertex-permutation lists
print(len(gauts))                      # 6

# Which polytope automorphisms preserve G's coefficients?
cp = coefficient_preserving_indices(fi, auts)
print(len(cp))                         # 1 (identity only — generic kinematics)

# Banana: coefficient-preserving = full group in massless case
banana = FeynmanIntegral.from_cnickel("111e|e|:zzz")
auts_b = banana.polytope_automorphisms
cp_b = coefficient_preserving_indices(banana, auts_b)
print(auts_b.order)                    # 24
print(len(cp_b))                       # 6 (= 3! edge permutations)
```

Each automorphism is stored as an `(U, t)` pair where `U` is a SymPy
`ImmutableMatrix` in $\mathrm{GL}_n(\mathbb{Z})$ and `t` is an integer column
vector. The `vertex_permutations` list records the induced action on the Newton
polytope vertices in the same order as `maps`, enabling the orbit computation
that drives the `vertex_orbits` attribute.

---

## 8. References

1. **Liu, C. & Cai, Y.** (2025). *Unimodular isomorphism of lattice polytopes*. arXiv:2506.23846.

2. **de la Cruz, L.** (2019). Feynman integrals as A-hypergeometric functions. *JHEP* 12, 123. doi:10.1007/JHEP12(2019)123.

3. **Lee, R. N. & Pomeransky, A. A.** (2013). Critical points and number of master integrals. *JHEP* 11, 165. doi:10.1007/JHEP11(2013)165.

4. **Gelfand, I. M., Kapranov, M. M. & Zelevinsky, A. V.** (1994). *Discriminants, Resultants and Multidimensional Determinants*. Birkhäuser, Boston.

5. **Bogner, C. et al.** (2017). Loopedia, a database for loop integrals. arXiv:1709.01266.
