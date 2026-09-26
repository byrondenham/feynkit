# Torus point counts

Design note for counting the points of $V = \{G = 0\}$ in
$(\mathbb{F}_p^*)^N$ at a physical kinematic point to obtain a candidate Euler
characteristic and master count.

## Purpose

feynkit reports the normalised volume, which counts master integrals only for
generic coefficients and when the exponents span the lattice (as for every
graph in the test set), and leaves the Euler characteristic at physical
kinematics uncomputed. Counts that fit a polynomial in $q$ give a candidate,
not a certificate: Katz's theorem gives $\chi(V) = P(1)$ only if
$\#V(\mathbb{F}_q) = P(q)$ for every finite field whose characteristic avoids
a finite set, which no finite sample establishes.

## Mathematics

Let $X = (\mathbb{C}^*)^N \setminus V$, with $\mu = 1$.

1. The number of master integrals, subsectors included, symmetries unused and
   $D$ symbolic, is $C = (-1)^N \chi(X)$ (Bitoun, Bogner, Klausen and Panzer,
   arXiv:1712.09215, Corollary 37). For generic exponents $|\chi(X)|$ counts
   the critical points of $\sum_e \nu_e \log u_e - (D/2) \log G$ on $X$, all
   regular (Fevola, Mizera and Telen, Comput. Phys. Commun. 303 (2024) 109278,
   arXiv:2311.16219, proof of Theorem 3.1, eqs. (3.1)-(3.2), citing Huh
   (2013), Theorem 1).
2. For almost all coefficients $|\chi(X)| = N!\,\mathrm{Vol}(\mathrm{Newt}\,G)$,
   which is always an upper bound (Bitoun et al., Theorem 44, crediting
   Kouchnirenko, Invent. Math. 32 (1976) 1, Theorem IV, and Khovanskii) and
   equals the normalised volume times the lattice index.
3. Katz (appendix to Hausel and Rodriguez-Villegas, Invent. Math. 174 (2008)
   555, arXiv:math/0612668, Theorem 6.1.2(3)): if a spreading out of $Y$ over
   a finitely generated ring $R$ has $\#Y_\phi(\mathbb{F}_q) = P(q)$ for
   every $\phi: R \to \mathbb{F}_q$, then $\chi(Y) = E(Y; 1, 1) = P(1)$, with
   $P$ integral. Hausel and Rodriguez-Villegas, Remark 2.1.9, state that
   finitely many characteristics may be ignored; their Example 2.1.10
   illustrates this.
4. As $\chi$ is additive and vanishes on the torus, a polynomial count $N_V$
   gives $C = (-1)^{N+1} N_V(1) \ge 0$.
5. U and the kinematic part of F are square-free (`polynomials/symanzik.py`,
   $a_e$ renamed $u_e$), and the mass term $U \sum_e m_e^2 u_e$ adds one to
   the degree in $u_e$ when $m_e \ne 0$: $G$ has degree at most 2 in each
   $u_e$, 1 if massless, and edges of lattice length at most 2.
6. For the massive bubble $V$ is an affine conic, smooth for $s \ne 0$, with
   $1 + (\lambda/p)$ points at infinity ($\lambda$ the Källén function) and
   three on the axes. So $N_V(p) = p - 3 - (\lambda/p)$, polynomial only when
   $\lambda$ is a rational square, while $C = 3$ for all $\lambda \ne 0$; at
   $\lambda = 0$ the count is $p - 3$ and $C = 2$.

## Design

### Kinematics

- The drawn symbols are those of $G$'s coefficients, after `on_shell` is
  substituted into `momentum_products`: invariants of `standard_invariants`,
  reduced by momentum conservation, and masses. `random.Random(seed)` draws,
  in order of symbol name, each invariant with `rng.choice` from the nonzero
  integers of $[-20, 20]$ in increasing order, and each $m_e^2$ itself (not
  $m_e$), or any symbol occurring only to even powers, with
  `rng.randint(1, 20)`.
- The Landau analysis is `landau_analysis(fi, max_face_points=...)` with
  $\mu$ symbolic, then $\mu = 1$. (Analysed at $\mu = 1$, the two-mass
  bubble's 2-face becomes principal, adding the condition that $s$ be a
  square.)
- The Landau analysis factorises in the masses, so a factor $f$ of a
  discriminant can be odd in a mass $m$; the tests below replace it by its
  norm $f(m) f(-m)$, a polynomial in $m^2$.
- A draw is admissible when every coefficient and face discriminant is
  nonzero.
- Square test: on faces of dimension at least 1 with principal discriminants,
  write each discriminant in the $m_e^2$ and take its irreducible factors from
  `sp.factor_list`, dropping the constant content, sign included, as
  `landau._factor_list` does; the sign convention of
  `landau._univariate_discriminant` is then immaterial. The first of 200
  admissible draws with every factor a nonzero rational square is used, else
  the first admissible one; $|\chi|$ is constant on an open dense set (Fevola
  et al., Theorem 3.1).
- An explicit `point` on a Landau surface raises `ValidationError`; with
  `allow_singular=True` it is used and the result marked `on_landau_surface`,
  since $C$ may drop there, to 3 for the triangle at $\lambda = 0$.
- Excluded primes: 2 and divisors of numerators and denominators of
  coefficients, nonzero discriminant values and their nonzero factors, and
  the discriminant of $G$ on each edge in its lattice coordinate (see
  Fitting). The rule is heuristic, missing the integer content of
  eliminants, skipped faces and components absent from the Landau list, but
  a missed bad prime causes a rejection, not a wrong result: the two-mass
  bubble at $s = 1$, $m_e^2 = (2, 6)$ has no points at $p = 3$, where the
  fit predicts $-1$.
- If an edge of the Newton polytope at the point has lattice length at least
  3, or a face has lattice index at least 3, the counts may depend on
  characters of order above 2, which the check cannot cover: $v + u^6 + 108$
  counts $p - 1$ at every prime from 5 to 29 and fits $C = 0$, where $C = 6$.
  Only the fit primes are then counted, and no candidate is given. Graphs
  have edges of length at most 2 (item 5), and every graph tested has faces
  of index 1.

### Counting

- Eliminate the variable $x$ of lowest degree; if that exceeds 2,
  `count_torus_points` raises `ValidationError`. With $G = a x^2 + b x + c$,
  the roots in $\mathbb{F}_p^*$ number $1 + (\Delta/p) - [c = 0]$ if
  $a \ne 0$ ($\Delta = b^2 - 4ac$), $[c \ne 0]$ if $a = 0 \ne b$, and
  $(p - 1)[c = 0]$ if $a = b = 0$, with Legendre symbols from a table.
- The other variables run in chunks of $2^{22}$ points, the last two broadcast
  as a grid with $a$, $b$, $c$ grouped by exponents there; for $N = 2$ one
  variable forms the grid, for $N = 1$ the formula applies once. `int64`
  arithmetic is exact.

### Fitting

- Interpolate exactly with `Fraction` on the first $N + 1$ primes not
  excluded, degree at most $N$.
- Verify on at least the next `verification` primes (4 by default), extended
  until every non-trivial character of the coverage group takes both signs on
  the fit and verification primes together. The group is generated by the
  $(d/p)$ for $d = -1$ and the values at the point of every irreducible factor
  of every face discriminant, principal or not, every vertex coefficient, and
  the discriminant of $G$ on every edge in its lattice coordinate; the last
  two keep the constants, contents and lattice indices that the Landau
  analysis drops, and their primes are excluded too. Each $d$ is a vector in
  $\mathbb{Q}^*/(\mathbb{Q}^*)^2$ over $\mathbb{F}_2$ (its sign and the prime
  support of its square-free part); a basis $d_1, \dots, d_k$ gives each prime
  a vector of signs in $\mathbb{F}_2^k$, and every non-trivial character takes
  both signs exactly when these vectors affinely span $\mathbb{F}_2^k$, that
  is when their differences from the first have rank $k$. A count that depends
  on a character constant on the sample fits a polynomial that other primes
  break, so neither the square-test factors nor the characters one at a time
  are enough. The one-mass triangle's counts depend on
  $(\lambda(p_1^2, p_2^2, p_3^2)/p)$, a factor of the discriminant of its top
  face, which is not principal. The count of
  $u^2 + (x + y)u + (x^2 + xy + y^2)/4$ is $1 + (xy/p)$, and at $x = 2$,
  $y = 45$ the characters of $-1$, $2$ and $5$ each take both signs on the
  sampled primes from 7 to 29 while $(90/p) = -1$ on all of them. For the
  two-mass bubble at $s = 2$, $m_e^2 = (5, 8)$ ($\lambda = -39$; 2, 3, 5, 11,
  13 excluded) the fit primes 7, 17, 19 and verification primes 23, 29, 31, 37
  all have $(\lambda/p) = -1$ and accept $C = 1$; coverage adds 41, where the
  count, 37 against 39, rejects it.
- Accept integer coefficients, a zero $q^N$ term, agreement at every
  verification prime and $0 \le C \le N!\,\mathrm{Vol}$; otherwise report "not
  polynomial on the tested primes" with a `reason`.
- A pass is evidence, not proof: a count depending on $(d/p)$ for $d$ outside
  the coverage group, such as the constant or content of a discriminant of a
  face of dimension 2 or more, or on a character of higher order, passes if
  all sampled primes agree on it.

### Output

A frozen `TorusCount`: `variables`, `eliminated`, `point`, `on_shell`, `seed`,
`on_landau_surface`, the excluded, fit and verification primes, `counts`,
`candidate_polynomial`, `candidate_euler_characteristic`,
`candidate_master_count` (None without a fit), `reason`, `skipped_faces: int`,
`backend`.

## Interfaces

```python
# feynkit/point_count.py
def count_torus_points(polynomial: sp.Expr, variables: Sequence[sp.Symbol], *,
                       scale: sp.Symbol | None = None, seed: int = 0,
                       point: Mapping[sp.Expr, int | Fraction] | None = None,
                       allow_singular: bool = False, landau: LandauAnalysis | None = None,
                       volume_bound: int | None = None, verification: int = 4,
                       max_prime: int = 1000, max_evaluations: int = 2 * 10**9,
                       backend: str = "numpy") -> TorusCount: ...
def critical_point_count(polynomial: sp.Expr, variables: Sequence[sp.Symbol],
                         point: Mapping[sp.Expr, int | Fraction], *, seed: int = 0) -> int: ...
# FeynmanIntegral
def torus_count(self, *, seed: int = 0, point: Mapping[sp.Expr, int | Fraction] | None = None,
                allow_singular: bool = False, on_shell: Mapping[sp.Symbol, sp.Expr] | None = None,
                landau: LandauAnalysis | None = None, verification: int = 4,
                max_face_points: int = 12, max_evaluations: int = 2 * 10**9,
                backend: str = "numpy") -> TorusCount: ...
```

Without `landau`, `count_torus_points` runs
`landau_analysis_from_polynomial(polynomial, variables, scale=scale)`, then
sets `scale` to 1; above `max_evaluations` it raises `ValidationError`.
`torus_count` applies `on_shell` via `with_`, runs
`landau_analysis(self, max_face_points=...)` unless given `landau` (also with
$\mu$ symbolic), then sets $\mu = 1$, and rejects `kinematic_constraints`.
`critical_point_count` returns Singular's `vdim(std(I))` for Fevola et al.,
eq. (3.2), at random rational exponents, raising `RuntimeError` without
Singular. `backend="flint"` (found with `importlib.util.find_spec`,
`RuntimeError` if absent) counts roots of `nmod_poly([c, b, a], p)` per
point, a cross-check without a vectorised Legendre symbol, hence no
`"auto"`.

## Report and CLI integration

- `SECTION_NAMES` gains `"torus"`. A new `DEFAULT_SECTIONS` without it
  replaces `SECTION_NAMES` as the default of `from_integral`,
  `ReportOptions.sections` and `_report_options` (the `--sections` help says
  "all but torus by default"), so `--latex`, `--text` and `--json` count
  points only when `--sections` names `torus`. `AnalysisReport` gains `torus`,
  `from_integral` gains `torus_seed` and `torus_budget`, and one Landau
  analysis serves both sections.
- A `_report_shared.py` helper on `report.torus` builds the not-computed
  sentence: unchanged without the section; with a candidate, a pointer
  replaces the Euler characteristic (`\ref` in LaTeX, the heading "Candidate
  Euler characteristic from point counts" in text); otherwise it says the
  counts are not polynomial on the tested primes.
- The section, after the Newton polytope, gives the point, primes, counts and
  candidates, cites `bbkp2017` and the new `katz2008` and `fmt2024` (the CPC
  paper), and says finitely many primes prove nothing. `summary()` gains a
  candidate row only with it.
- On `fk analyse`, `--torus-count` (`store_true`, help "print the candidate
  Euler characteristic from finite-field point counts; slow") is a section
  flag: alone it prints the graph and that section, and no-flag runs leave it
  out. `--json` rejects it like the other section flags; JSON takes
  `--sections torus`. Given both, the count runs once.
- `--seed N` (help "seed for the kinematic point") and `--torus-budget N`
  (help "maximum evaluations") join `_VALUE_OPTIONS`, so the bare-form
  rewriter skips their values. Both default to None, meaning 0 and
  $2 \times 10^9$; each is an error unless `--torus-count` is given or
  `--sections` names `torus`, tested with `is not None` since 0 is valid.

## Testing and acceptance

Kernel: the per-prime count against brute force for $p \le 23$, item 5,
rejected fits, deterministic draws, the $p = 3$ example, flint against numpy.

At seeds 0 and 1 each case has an integer candidate polynomial and
`candidate_master_count`:

| Integral | $C$ | Source |
|---|---|---|
| `11e\|e\|:zz` | 1 | $V$ is a graph over $u_1$; Table 2 |
| `11e\|e\|:nz` | 2 | item 6 with $m_2 = 0$: two axis points, count $p - 3$ |
| `11e\|e\|:nn` | 3 | item 6; Table 2 |
| `12e\|2e\|e\|:zzz` | 4 | below; Table 2 |
| `12e\|3e\|3e\|e\|:zzzz`, $p_i^2 = 0$ | 3 | Table 1, A4 |
| `12e\|23\|3\|e\|:zzzzz` | 3 | Bitoun et al., Example 53 |

Tables are those of Fevola et al. The off-shell box at
$p_i^2 = (-12, -6, 5, -4)$, $s_{12} = 20$, $s_{23} = -41$, where the five
square-test factors are squares, gives 11 (Table 1, A4). For the triangle,
$G = A + u_3 B$ with $A$, $B$ free of $u_3$, and Bitoun et al., Lemma 48,
give $\chi(X) = -\chi((\mathbb{C}^*)^2 \setminus \{AB = 0\})$; the curves
are lines less two points meeting twice when $\lambda \ne 0$, so $C = 4$.

No candidate, while `critical_point_count` gives 3, 4 and 7: the bubble at
$s = m_e^2 = 1$ (counts per item 6), the triangle at $p_i^2 = 1$, the
three-mass sunrise at $s = 7$, $m_e^2 = (2, 3, 5)$. Elsewhere it matches the
candidate (skipped without Singular). The triangle at $p_i^2 = (1, 4, 9)$
raises, or gives 3 with `allow_singular`.

PLD: the test reads the archive from `FEYNKIT_PLD_DATA`, skipping when unset.
Chi keys start with the Greek letter (U+03C7), and F has the opposite sign,
so $G = U - F$. `A4_zero_generic` at $M = (-12, 5, -4, -6)$, $s = 4$,
$t = 20$ must give 11 and `A4_zero_zero` at $s = 16$, $t = 9$ must give 3;
other entries with at most five variables run at seed 0, and any candidate
must match. The archive has no licence file; MathRepo gives MIT for code and
CC BY 4.0 for content.

Report and CLI: the section only when named, the sentence cases, LaTeX
compiling, the option rules.

## Performance and limits

A prime costs $(p-1)^{N-1}$ evaluations; a run uses at least $N + 5$. At an
expected $10^7$ a second, $N \le 5$ takes seconds, $N = 6$ a minute,
$N = 7$ an hour or more, $N \ge 8$ days; the default budget admits $N \le 6$.
Massive graphs often give no candidate, as many factors must be squares at
once. For massless graphs, solving $A + Bx + Cy + Dxy = 0$ by cases on $A$,
$B$, $C$, $D$ and $AD - BC$ would cut the cost to $(p-1)^{N-2}$.

## Open questions

- Counting over $\mathbb{F}_{p^2}$ would add evidence, not proof.
- Split primes: by item 6 the bubble at $s = m_e^2 = 1$ fits with $C = 3$ on
  $p \equiv 1 \pmod 3$. Katz over $\mathbb{Z}[1/M, \sqrt{d}]$, with $M$ the
  product of the excluded primes, would need every finite field receiving
  that ring.
- Unchecked: components missing from the Landau list, the Khovanskii
  reference, and the lattice index behind the $\mathrm{vol}_0$ bound in
  mathematics reference 4.5.
