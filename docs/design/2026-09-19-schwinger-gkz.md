# Schwinger-representation GKZ system

Design note for adding the GKZ system of the Schwinger representation to the
`FeynmanIntegral` facade, following Jimenez-Santacruz, Lopez-Arcos and
Quintero Velez, arXiv:2609.16107 (2026).

## Purpose

The existing `fi.gkz` is the GKZ system of the Lee-Pomeransky polynomial
$G = U + F$ (de la Cruz 2019; Klausen 2020). The paper above starts from the
Schwinger representation instead and arrives at a generalised Euler integral in
two polynomials, the dehomogenised Symanzik polynomials $\tilde U$ and
$\tilde F$. Its GKZ system has a two-block Cayley structure that keeps the
topological and kinematic coefficients apart, which is what lets the paper read
off facet reductions and work with fewer variables. This note specifies how
feynkit builds that system and its reduction.

## Prior art

The same two-block system appears in Klausen's thesis (arXiv:2302.13184, section 3.4, the
equation labelled AprimeUF) as the "$\mathcal{U}\mathcal{F}$ Cayley form", with
$\beta = (d/2 - \omega, \omega, \nu_1, \ldots, \nu_{n-1})$ in his plus-sign convention and
$\omega = \sum\nu - L d/2$. Translated to feynkit's convention it is the vector below. Klausen
also states that this configuration is unimodularly equivalent to the Lee-Pomeransky one, which
gives a test: `is_unimodular_equivalent` between the Cayley matrix and `fi.gkz.a_matrix` must
succeed. The 2026 paper's contribution is the derivation from the Schwinger representation and the
use of the block structure for reductions.

## Mathematics and conventions

All references are to arXiv:2609.16107 unless stated.

1. Substitution (section 3, eq. 38): $\alpha = (t u_1, \ldots, t u_{N-1}, t)$.
   Homogeneity gives $U(\alpha) = t^L \tilde U(u)$ and
   $F(\alpha) = t^{L+1} \tilde F(u)$ with $\tilde U(u) = U(u_1, \ldots, u_{N-1}, 1)$
   and likewise for $\tilde F$.
2. Integrating out $t$ gives, with $\nu = \sum_{i=1}^N \nu_i$ and $\beta = D/2$,
   $$\Phi = \int_{u \geq 0} \prod_{i=1}^{N-1} du_i\, u_i^{\nu_i - 1}\,
     \tilde U(u)^{\nu - (L+1)\beta}\, \tilde F(u)^{L\beta - \nu}.$$
   The prefactor relating $I_\Gamma$ to $\Phi$ is feynkit's Schwinger prefactor
   times $\Gamma(\nu - L\beta)$ from the $t$ integral.
3. Cayley configuration (section 2.3 and eq. 47): two rows of ones, one per
   polynomial, above the exponent blocks $A_1$ (from $\tilde U$) and $A_2$ (from
   $\tilde F$).
4. Parameter vector, in the convention $(\sum_j a_{ij} z_j \partial_j - \beta_i)\Phi = 0$
   of eq. 2.16:
   $$\vec\beta = \bigl(\nu - (L+1)\beta,\; L\beta - \nu,\; -\nu_1, \ldots, -\nu_{N-1}\bigr).$$
   Each entry is the exponent of the corresponding polynomial, or minus the
   exponent of $u_i$. This is what the paper's worked examples use (eqs. 53,
   54, the triangle and the box) and it agrees with the convention feynkit now
   uses for `fi.gkz`, $\beta = (-D/2, -\nu)$.

Three points where the paper's text disagrees with its own examples, resolved
in favour of the examples and of the derivation:

- Section 2.3 writes the first two entries as $+\gamma_k$ with
  $\gamma_1 = (L+1)\beta - \nu$. The examples and the homogeneity argument give
  $-\gamma_k$.
- Eq. 46 writes $\Gamma(\nu - \beta)$ and $\tilde F^{\beta - \nu}$; the line
  before it has $t^{\nu - L\beta - 1}$, so the general form is
  $\Gamma(\nu - L\beta)$ and $\tilde F^{L\beta - \nu}$. The examples are all
  one-loop, where the two agree.
- The measure after the substitution keeps $1/\Gamma(\nu_i)$ only for
  $i \leq N-1$; the factor $1/\Gamma(\nu_N)$ is dropped. feynkit keeps it.

## Data model

New dataclass `CayleyGKZSystem` in `feynkit/systems/cayley.py`:

| Field | Meaning |
|---|---|
| `a_matrix` | The Cayley matrix, shape $(N+1) \times (n+m)$ |
| `w_variables`, `z_variables` | Coefficient variables of $\tilde U$ (paper's $w_j$) and $\tilde F$ (paper's $z_j$) |
| `u_support`, `f_support` | Monomial support of each polynomial, as (exponent, coefficient) pairs |
| `u_polynomial`, `f_polynomial`, `u_variables` | $\tilde U$, $\tilde F$ and $u_1, \ldots, u_{N-1}$ |
| `beta_parameters` | The vector above |
| `euler_equations` | One equation per row of `a_matrix`, from `create_euler_equations` |
| `prefactor` | Schwinger prefactor times $\Gamma(\nu - L D/2)$ |
| `dimension`, `propagator_exponents`, `loop_count` | Inputs, kept for reference |

Column order is the $\tilde U$ block then the $\tilde F$ block, each in the
graded reverse lexicographic order the rest of feynkit uses. Variable names
are `w_1 ... w_n` and `z_1 ... z_m`.

## Methods

- `toric_ideal(backend="auto")` calls `compute_toric_ideal_generators` on the
  Cayley matrix with the $w$ and $z$ variables.
- `restrict_to_f_block()` returns an ordinary `GKZSystem` with
  $A = (1 \cdots 1;\ A_2)$ and $\beta = (L D/2 - \nu, -\nu_1, \ldots, -\nu_{N-1})$,
  the paper's eq. 54 and its reduced box.

  What this reduction is, stated carefully. Britto, Grimm and Hoefnagels (arXiv:2606.09978,
  section 2.2) show that the solutions of a face subsystem $(\mathcal{A}_F, \beta)$ lie in the
  solution space of the full system, and treat the $\mathcal{U}$ facet only when its exponent
  vanishes, $\nu = (L+1)D/2$ (section 8.1). The converse does not hold in general: rescaling the
  $\tilde F$ coefficients along an exponent row rescales $u$, which changes
  $\tilde U^{\nu - (L+1)D/2}$ unless that exponent is zero, so the $\tilde F$-block Euler
  equations do not annihilate $\Phi$ at generic $D$. Vanhove (arXiv:1807.11466, section 3.2)
  uses the corresponding object for the maximal cut, where the torus cycle has no boundary. The
  docstring therefore says: the reduced system is a face subsystem whose solutions are solutions
  of the full system; it annihilates the Feynman integral itself only when the $\tilde U$
  exponent vanishes or on cut contours; the paper's comparison of its rank with the number of
  master integrals is the paper's own observation, not a bound.

## Facade

- `FeynmanIntegral.schwinger_gkz`, a cached property returning
  `CayleyGKZSystem`.
- `SchwingerParametrisation.get_A_matrix` becomes a thin call into the new
  module so the two cannot drift.

## Tests

Oracles taken from the paper, compared as sets of columns because the paper
orders columns differently:

- Two-mass bubble: full $3 \times 5$ matrix and $\beta$ (eq. 53); reduced
  $2 \times 3$ matrix and $\beta$ (eq. 54); the reduced toric ideal is generated
  by $z_1 z_3 - z_2^2$, matching the kernel vector $(1, -2, 1)$.
- Massless triangle: the $4 \times 6$ matrix and $\beta = (3 - 2\beta, \beta - 3, -1, -1)$
  at unit exponents.
- On-shell massless box: the $5 \times 6$ matrix and $\beta = (4 - 2\beta, \beta - 4, -1, -1, -1)$.
- Off-shell massless box: the reduced $4 \times 6$ matrix and $\beta$.
- Unimodular equivalence of the Cayley matrix and `fi.gkz.a_matrix` (Klausen 2023, section 3.4).
- Numerical homogeneity of the bubble's Cayley integral, in the style of the
  test added for `fi.gkz`, so the sign convention is pinned independently of
  the paper.
- General-$L$ exponent check on a two-loop sunrise: the prefactor and the
  first two $\beta$ entries use $L = 2$.

## Documentation

- New section in `docs/mathematics-reference.md` with the derivation and
  equation references, including the three discrepancies above.
- A section in `docs/guide.md` with a runnable example.
- Changelog entry under Unreleased. Together with the $\beta$ convention fix
  this is release 0.3.0.

## Out of scope

Pfaffian systems, Frobenius bases and the canonical $\epsilon$-form of the
paper's later sections. A general admissibility test for facet reductions.
