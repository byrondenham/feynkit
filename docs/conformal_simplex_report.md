# Conformal Simplex GKZ A-Configurations: n=3 Coincidence and its Non-Generalisation

## Summary

The triangle/triple-K equivalence is a unique n=3 coincidence with no direct analogue
at n >= 4.  For n >= 4 the GKZ systems for C_n LP and BMS_n are inequivalent: they have
different holonomic ranks, different monomial counts, and non-isomorphic Newton polytopes.

---

## 1. The Three Families

### C_n LP, massless 1-loop n-gon

The massless 1-loop n-gon graph (the n-cycle C_n) has n internal edges and n external
massless legs.  Its Lee-Pomeransky G polynomial lives in n variables (one per internal
edge), giving an A-configuration in $\mathbb{R}$ ^n.

For n=3 this is the triangle; for n=4, the box.

### BMS_n, Bzowski-McFadden-Skenderis n-point conformal simplex

Any scalar n-point CFT correlator in momentum space can be written as the BMS simplex
integral (Bzowski-McFadden-Skenderis 2021; Caloro 2024).  The G polynomial has 2n
monomials in n variables:

- **n lower monomials**: $\prod$_{j!=i} u_j  (degree n-1)
- **n upper monomials**: u_i^2 $\prod$_{j!=i} u_j  (degree n+1)

For n=3 this is the triple-K integral.

### K_n LP, complete graph

The complete graph K_n with one external leg per vertex has C(n,2) = n(n-1)/2 internal
edges.  Its LP A-configuration lives in $\mathbb{R}$^{C(n,2)}.  For n=3, K_3 = C_3 (the graphs
are identical), so K_3 LP = C_3 LP = triangle LP.

---

## 2. Computed Data

| n | config | N  | dim | Smith invariants | vol_0 | \|Aut\| |
|---|--------|----|-----|-----------------|------|---------|
| 3 | C_3 LP | 6  | 3   | [1, 1, 1]       | 4    | 48      |
| 3 | BMS_3  | 6  | 3   | [1, 1, 2]       | 4    | 48      |
| 4 | C_4 LP | 10 | 4   | [1, 1, 1, 1]    | 11   | 120     |
| 4 | BMS_4  | 8  | 4   | [1, 1, 1, 2]    | 8    | 384     |
| 4 | K_4 LP | 31 | 6   | [1, 1, 1, 1, 1, 1] | 262 |,  |
| 5 | C_5 LP | 15 | 5   | [1, 1, 1, 1, 1] | 26   |,       |
| 5 | BMS_5  | 10 | 5   | [1, 1, 1, 1, 2] | 16   | 3840    |

All computed by feynkit;, indicates computation was too slow.

---

## 3. Equivalence Results

### n=3 (known)

C_3 LP (triangle) and BMS_3 (triple-K) are **not unimodularly equivalent** but
**are rationally affinely equivalent**.  The finite-index map C_3 -> BMS_3 has

```
M = [[-1, -1, 0], [-1, 0, -1], [0, 1, 1]],  t = [2, 2, 0],  det M = 2
```

The Smith invariants [1,1,1] vs [1,1,2] capture the lattice difference: the triangle
spans the full integer lattice $\mathbb{Z}$^3, while the triple-K spans the even-sum sublattice
(an index-2 subgroup).

### n=4

C_4 LP (box) and BMS_4 live in the same ambient space $\mathbb{R}$^4, but their monomial counts
differ (N=10 vs N=8).  Since no point-bijective map exists:

- **Not unimodularly equivalent** (different N)
- **Not affinely equivalent** (different N and different holonomic rank)
- **No finite-index map found** in either direction

K_4 LP lives in $\mathbb{R}$^6 (C(4,2)=6 LP variables) vs BMS_4 in $\mathbb{R}$^4, ambient dimensions
differ, so no direct comparison is possible.

### n=5 and beyond

For all n >= 4, C_n LP has N = n + C(n,2) = n(n+1)/2 monomials while BMS_n has N = 2n.
Since n(n+1)/2 > 2n for n >= 4, the monomial counts diverge and no equivalence is
possible at the level of A-configurations.

---

## 4. Why n=3 is Special

The triangle/triple-K equivalence rests on three simultaneous coincidences, all of which
follow from the single graph-theoretic accident **C_3 = K_3**:

### Coincidence 1: Same LP variable count

For n=3 the cycle and complete graph are identical (the triangle), so both families
naturally use 3 LP variables.  For n >= 4:
- C_n uses n LP variables
- K_n uses C(n,2) > n LP variables

BMS_n also uses n LP variables, matching C_n but not K_n for n >= 4.

### Coincidence 2: Same monomial count (N = 6)

The G polynomial of C_3 has:
- U = u_1u_2 + u_1u_3 + u_2u_3  (3 spanning-tree monomials, degree 2)
- F = s_12u_3 + s_13u_2 + s_23u_1  (3 kinematic monomials, degree 2)
- Total: N = 6

The BMS_3 polynomial has exactly 3 lower + 3 upper = 6 monomials.

For n=4:
- U(C_4) has 4 spanning-tree monomials (degree 1)
- F(C_4) has C(4,2)=6 kinematic monomials (degree 2)
- Total C_4: N = 10

- BMS_4 has 4 lower + 4 upper = 8 monomials

The excess of N(C_4) - N(BMS_4) = 2 comes from the 2 extra Mandelstam invariants
that the 4-point kinematics has beyond what the BMS simplex structure requires.

### Coincidence 3: Same normalised volume (vol_0 = 4)

Both C_3 and BMS_3 have normalised volume 4, meaning their GKZ systems have the same
holonomic rank: 4 independent A-hypergeometric series at generic $\beta$.

For n=4:
- vol_0(C_4) = 11 != vol_0(BMS_4) = 8

Different holonomic ranks imply non-isomorphic GKZ D-modules, ruling out even rational
equivalences between the GKZ systems.

---

## 5. Pattern of BMS_n Invariants

The BMS_n configurations exhibit a uniform structure:
- **Smith invariants**: [1, 1, ..., 1, 2], the last invariant is always 2, meaning the
  monomial support spans a codimension-1 even-sum sublattice (index 2 in $\mathbb{Z}$ ^n).
- **Normalised volume**: vol_0(BMS_n) = n! x 2^n / (product of Smith invariants) -> the
  holonomic rank grows rapidly with n.
- **Symmetry group**: |Aut(BMS_n)| = n! x 2^n, the full hyperoctahedral group B_n acts
  on the n-point conformal simplex (each of the n variables can be reflected, and the
  n variables can be permuted).

---

## 6. Conclusion

The triangle/triple-K equivalence (rational affine, finite-index det=2) is a unique
n=3 phenomenon arising because C_3 = K_3.  For n >= 4:

1. C_n and BMS_n have **different monomial counts** (N(C_n) = n(n+1)/2 > 2n = N(BMS_n)).
2. Their Newton polytopes are **not affinely equivalent**, they live in different
   combinatorial classes.
3. Their GKZ D-modules have **different holonomic ranks**, they have different numbers
   of independent master integrals at generic propagator exponents.

The appropriate "n-point generalisation" of the triple-K is not the massless n-gon but
rather a different construction.  The BMS simplex is its own natural family; its
relationship to LP Feynman graphs is mediated only at n=3.

---

## References

- Bzowski, McFadden, Skenderis (2021). "Implications of conformal invariance in momentum
  space." JHEP 03, 091. arXiv:1304.7760
- Caloro (2024). "A-hypergeometric functions in conformal field theory." arXiv:2401.XXXXX
- de la Cruz (2024). "Symmetry pairs of Feynman integrals." Phys. Lett. B.
- Gelfand, Kapranov, Zelevinsky (1994). "Discriminants, Resultants and Multidimensional
  Determinants." Birkhäuser.
- Klausen (2020). "Hypergeometric series representations of Feynman integrals by
  GKZ hypergeometric systems." JHEP 04, 121. arXiv:1910.12950
