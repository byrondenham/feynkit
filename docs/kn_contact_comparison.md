# K_n LP vs. contact star: Newton polytope comparison

**Date:** 2026-05-04  
**Script:** `examples/kn_contact_comparison.py`

## Setup

Two families of A-configurations are compared for $n = 3, 4, 5$:

**Object 1 — $K_n$ LP integral.** The massless complete-graph Feynman integral on $K_n$ (one external leg per vertex, all internal masses zero). Built via feynkit's Lee–Pomeransky pipeline: $\mathcal{G} = \mathcal{U} + \mathcal{F}$ in the $C(n,2)$ Schwinger parameters. The A-matrix has $C(n,2)+1$ rows (one homogenisation row) and one column per monomial of $\mathcal{G}$.

**Object 2 — $n$-point contact star.** The A-matrix
$$A_\star = \begin{pmatrix} \mathbf{1} & \mathbf{1} \\ -I_n & I_n \end{pmatrix} \in \mathbb{Z}^{(n+1)\times 2n}$$
where $\mathbf{1}$ is a row of ones and $I_n$ is the $n\times n$ identity. The $2n$ affine points are $\{-e_i\} \cup \{+e_i\}$ — the standard cross-polytope in $\mathbb{R}^n$.

---

## Step 1 — n = 3 baseline: triangle LP vs. triple-K star

The $K_3$ LP A-matrix from feynkit:
$$A_\triangle = \begin{pmatrix}1&1&1&1&1&1\\1&1&1&0&0&0\\1&0&0&1&1&0\\0&1&0&1&0&1\end{pmatrix}$$

This is the same configuration as the task specification up to a column permutation (different edge-labelling convention). The column sets are identical as multisets; the Newton polytope, GKZ system, and all invariants are unchanged.

The triple-K (contact star, $n=3$) A-matrix:
$$A_\star = \begin{pmatrix}1&1&1&1&1&1\\-1&0&0&1&0&0\\0&-1&0&0&1&0\\0&0&-1&0&0&1\end{pmatrix}$$

### Invariants

| Object | dim | verts | norm-vol | Smith |
|---|---|---|---|---|
| Triangle LP | $\mathbb{R}^3$ | 6 | 4 | $[1,1,1]$ |
| Triple-K star | $\mathbb{R}^3$ | 6 | 8 | $[1,1,2]$ |

The triangle LP Newton polytope (norm-vol 4) and the triple-K star Newton polytope (the standard octahedron, norm-vol 8) are affinely equivalent over $\mathbb{Q}$ with:

$$M = \begin{pmatrix}0&1&1\\-1&-1&0\\-1&0&-1\end{pmatrix}, \quad t = (-1,\,1,\,1)^T, \quad \det M = -2$$

The map extends to the full point configuration (all 6 A-columns), also with $|\det M| = 2$.

**Sanity check passed.** The factor of 2 reflects the index-2 sublattice discrepancy in the Smith invariants: the contact star spans the even-sum sublattice $\{x \in \mathbb{Z}^3 : x_1+x_2+x_3 \equiv 0 \pmod{2}\}$ (Smith $[1,1,2]$), while the triangle LP spans the full $\mathbb{Z}^3$ (Smith $[1,1,1]$).

---

## Step 2 — K_4 and K_5 LP A-matrices

The LP Schwinger parameter count grows as $C(n,2)$:

| $n$ | edges $C(n,2)$ | A-shape | ambient $\mathbb{R}^d$ | monomials | verts | norm-vol | Smith |
|---|---|---|---|---|---|---|---|
| 3 | 3 | $4\times 6$ | $\mathbb{R}^3$ | 6 | 6 | 4 | $[1,1,1]$ |
| 4 | 6 | $7\times 31$ | $\mathbb{R}^6$ | 31 | 31 | 262 | $[1,1,1,1,1,1]$ |
| 5 | 10 | $11\times 235$ | $\mathbb{R}^{10}$ | 235 | 235 | 347112 | $[1,\ldots,1]$ (10 ones) |

**All monomials are hull vertices** for $K_4$ and $K_5$. This is consistent with the 0/1 structure of the spanning-tree polynomial: every monomial exponent vector is $\{0,1\}^{C(n,2)}$-valued, so the Newton polytope of $\mathcal{G}$ is a 0/1 polytope, and all vertices of a 0/1 polytope are extreme points.

The Smith invariants are all 1 for $n \geq 4$, meaning the monomial support spans the full integer lattice $\mathbb{Z}^{C(n,2)}$ — no sublattice constraint, unlike the contact star ($[1,\ldots,1,2]$) or the BMS simplex.

The K_5 hull computation (235 points in $\mathbb{R}^{10}$) requires several minutes and was pre-established in a background run; all other values are computed interactively.

---

## Step 3 — Dimensional obstruction for n ≥ 4

The contact star for $n$ external legs always lives in $\mathbb{R}^n$. The $K_n$ LP integral lives in $\mathbb{R}^{C(n,2)}$. For $n \geq 4$, $C(n,2) > n$, so no affine map between their ambient spaces exists:

| $n$ | $K_n$ LP dim | contact star dim | result |
|---|---|---|---|
| 3 | 3 | 3 | **EQUIVALENT**, $\det M = -2$ |
| 4 | 6 | 4 | **OBSTRUCTION**: $\mathbb{R}^6 \neq \mathbb{R}^4$ |
| 5 | 10 | 5 | **OBSTRUCTION**: $\mathbb{R}^{10} \neq \mathbb{R}^5$ |

The $n=3$ equivalence is the only case where direct comparison is possible, and it yields the expected $|\det| = 2$ map. The $n \geq 4$ obstruction is not a failure of a specific map search — it is a categorical impossibility: there is no affine injective map from $\mathbb{R}^4$ (or $\mathbb{R}^5$) to $\mathbb{R}^6$ (or $\mathbb{R}^{10}$) that could serve as an equivalence between Newton polytopes living in those spaces.

---

## Step 4 — Cross-ratio reduction

**Setup.** Setting $\bar{x}_i = 1$ (restricting to the physical subspace) corresponds to deleting the $n$ columns of $I_n$ from the contact star A-matrix. What remains is the $(n+1)\times n$ sub-matrix with affine points $\{-e_i : i = 1,\ldots,n\} \subset \mathbb{R}^n$ — the $n$ vertices of an $(n-1)$-simplex.

**Comparison against the triangle LP Newton polytope** ($\mathbb{R}^3$, 6 vertices, norm-vol = 4, a triangular prism with faces at $\sum x_i = 1$ and $\sum x_i = 2$):

| $n$ | restricted polytope | first obstruction |
|---|---|---|
| 3 | $\mathbb{R}^3$, $(n-1)$-simplex, 3 verts, vol = 2 | vertex count: $3 \neq 6$ |
| 4 | $\mathbb{R}^4$, $(n-1)$-simplex, 4 verts | ambient dim: $\mathbb{R}^4 \neq \mathbb{R}^3$ |
| 5 | $\mathbb{R}^5$, $(n-1)$-simplex, 5 verts | ambient dim: $\mathbb{R}^5 \neq \mathbb{R}^3$ |

**The triangle LP Newton polytope does not reappear at any $n \in \{3,4,5\}$ after the cross-ratio restriction.**

- For $n = 3$: same ambient space, but the restricted set $\{-e_1,-e_2,-e_3\}$ is a 2-simplex (3 vertices), while the triangle LP polytope is a triangular prism (6 vertices: the three $\mathcal{U}$-monomials at total degree 2 and the three $\mathcal{F}$-monomials at total degree 1). The cross-ratio restriction recovers only half of the polytope.
- For $n \geq 4$: the ambient dimension is the first obstruction, before any vertex-count or volume comparison.

---

## Summary

The $K_n$ LP and $n$-point contact star families coincide in their Newton polytopes only at $n = 3$ (up to a det = $-2$ rational-affine map). For $n \geq 4$, the quadratic growth of the Schwinger parameter space ($C(n,2)$ vs. $n$ variables) makes direct comparison impossible: the two families live in genuinely different spaces and there is no affine equivalence between their Newton polytopes.

The cross-ratio restriction (retaining only the $-I_n$ block) likewise does not recover the triangle LP polytope at any $n$: it yields only an $(n-1)$-simplex, which is structurally simpler (half the vertex count for $n=3$, lower-dimensional ambient space for $n \geq 4$).
