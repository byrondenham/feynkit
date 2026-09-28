# Landau robustness

Design note for making `landau_analysis` and the report's one-loop closed form work on one-loop
graphs with bridges and on the massless pentagon and hexagon, and for the rule a face follows when
its elimination ideal has several generators.

## Purpose

Three graphs show the defects.

- `01e|e|:zn`, a massless self-loop with a massive bridge, stops `fk analyse` with a `KeyError` in
  the report's closed form, which assumes that every vertex with a leg lies on the cycle. When the
  first internal edge is a bridge, as in `1e|22|e|:nnn`, the closed form uses a wrong cycle of four
  edges and gives wrong factors without an error.
- The massless hexagon `12e|3e|4e|5e|5e|e|:zzzzzz` fails with a `RecursionError` after about six
  minutes, and the massless pentagon `12e|3e|4e|4e|e|:zzzzz` had not finished after forty. One
  face of the hexagon takes Singular over four minutes and returns a generator of 13 730 terms,
  which SymPy's parser cannot read.
- On the massive kite of the principal Landau determinant database the face computation reports
  factors that are not components of the principal Landau determinant.

The face computation stays as it is: every face of the Newton polytope, its discriminant, and the
reduced product. What changes is the closed form for graphs with bridges, how a face is eliminated,
which factors a face with several generators contributes, and how the closed form's minors are
computed.

## Bridges

A bridge is an internal edge on no cycle. A one-loop graph is its cycle $C$ with trees attached.
Each bridge $b$ carries the momentum $q_b$ of the legs on its side away from the cycle, so
$$I_\Gamma = I_{C'} \prod_b \frac{1}{(m_b^2 - q_b^2)^{\nu_b}},$$
where $C'$ is the cycle with the legs of each tree moved to the vertex where the tree meets the
cycle. $U = U_C$ does not involve the bridge parameters $x_b$, and
$$G = U_C \Bigl(1 + \sum_b \frac{m_b^2 - q_b^2}{\mu^2}\, x_b\Bigr) + F_{C'}.$$
The reduced principal $A$-determinant of $G$ is therefore that of $C'$ times the poles
$m_b^2 - q_b^2$, and the face computation, run on the whole of $G$, already finds both.

The closed form follows the factorisation.

- The cycle is found by removing, again and again, a vertex on a single internal edge and carrying
  its legs to its neighbour. What is left is the cycle, each of its vertices holding the legs of
  its tree, and the edges removed are the bridges, each with the legs on its far side. A self-loop
  counts twice at its vertex. The modified Cayley matrix is built for $C'$ as before.
- `one_loop_bridge_poles(integral)`, a new public function, returns the distinct irreducible factors
  of the $m_b^2 - q_b^2$. `one_loop_landau_surfaces_by_type` keeps its two lists, which describe the
  cycle, and `one_loop_landau_surfaces`, with `one_loop_principal_a_determinant`, appends the bridge
  poles, since it is the closed form of the whole integral that the face computation is compared
  with.
- The report's Landau section lists the bridge poles after the first-type and second-type factors
  when there are any.

## The elimination

A face of dimension two or more that is not a simplex contributes the elimination ideal of
$\{f = 0,\ t_i \partial_i f = 0\}$ in the torus, with $f$ the restriction of $G$ to the face in
lattice coordinates. Three things change.

**The energy scale is set to 1.** The coefficients of $G$ are 1 on the monomials of $U$ and
$K/\mu^2$ on those of $F$, with $K$ free of $\mu$. Kept as a variable, $\mu$ adds a component at
$\mu = 0$, where only the monomials of $F$ survive once denominators are cleared; on the hexagon's
face it produced the 13 730-term generator. On a face the degree in the Lee-Pomeransky parameters,
less $L$, is an affine function of the exponents that takes the value 0 on $U$ and 1 on $F$, so
rescaling $\mu$ multiplies the coefficients by a character of the torus: $f$ at $\lambda\mu$ is a
constant times $f$ at $\mu$ taken at a rescaled point of the torus. The singular locus is therefore
a cone in $\mu$, and away from $\mu = 0$ it is determined by its slice at $\mu = 1$: eliminating at
$\mu = 1$ gives the same factors, except $\mu$ itself.

Faces with both $U$ and $F$ monomials whose extra generators came from $\mu = 0$ become principal,
and their discriminants lose the factors those generators added: the massive bubble's polygon
gives $s$, where it gave $\mu\, s\, (s - (m_1 + m_2)^2)(s - (m_1 - m_2)^2)$. The Landau surfaces
are unchanged on every graph tried. Vertex and edge discriminants are computed as before and keep
$\mu$. The point count's square test takes the factors of principal faces. To keep the faces it
took before, it leaves out a principal face with both kinds of monomial whose monomials of $F$ are
not affinely independent; on every graph tried these are exactly the faces that were not principal
with $\mu$ kept.

**Independent coefficients are renamed.** Let the distinct coefficients of the face that are not
constant be $L_1(y), \dots, L_r(y)$, linear forms with rational coefficients in atoms
$y_1, \dots, y_n$, each atom a kinematic symbol that occurs only to the first power or only
squared, as masses do. When the $L_i$ are linearly independent, each is replaced by a fresh symbol
$c_i$ before the elimination, and the factors are found in the $c_i$ and then substituted back one
by one. On the hexagon's face the elimination then takes well under a second instead of 255 s
and gives a generator of 22 terms in the $c_i$, which is the 145-term generator in the
invariants.

This is exact. Complete $L_1, \dots, L_r$ to a basis $L_1, \dots, L_n$ of the linear forms in the
atoms. Then $c_i = L_i(y)$ is a linear change of coordinates, an automorphism of
$\mathbb{Q}[w, t, y] = \mathbb{Q}[w, t, c_1, \dots, c_n]$, and the system involves only
$c_1, \dots, c_r$: its ideal is $I\,\mathbb{Q}[w, t, c_1, \dots, c_n]$ for an ideal $I$ of
$\mathbb{Q}[w, t, c_1, \dots, c_r]$. The other $c_j$ are free variables. A Gröbner basis of $I$ for
an elimination order remains one after they are adjoined, so eliminating $w$ and $t$ commutes with
adjoining them, and the generators of $I \cap \mathbb{Q}[c_1, \dots, c_r]$ with $c_i = L_i(y)$
generate the elimination ideal in the atoms. Adjoining variables and changing coordinates preserve
greatest common divisors and irreducibility, so the factors substituted back are the irreducible
factors in the atoms. When an atom is a square $m^2$, $\mathbb{Q}[m]$ is free over
$\mathbb{Q}[m^2]$ with basis $1, m$, and elimination and greatest common divisors still commute
with the inclusion, but a factor can split, as $m_1^2 - m_2^2$ does, so such factors are factored
again after substitution. Coefficients that fail the test are eliminated as they are.

**Singular's output is read term by term.** Singular prints each generator expanded on one line.
SymPy's `parse_expr` compiles the line as one nested sum and recurses once per term. Each line is
split into its terms and read into `Poly.from_dict`. A failure of Singular, output that is not a
polynomial in the kinematic variables and a run past an optional `timeout`, None by default, raise
`ComputationError`, naming the size of the face and of the output, which `fk` reports in one line.
The factors of each face are passed on to the reduction as they are found, so that none is factored
twice.

## Several generators

When the elimination ideal has generators $g_1, \dots, g_k$, the face contributes the irreducible
factors of $\gcd(g_1, \dots, g_k)$, not every factor of every generator. The codimension-one part
of $V(g_1, \dots, g_k)$ is $V(\gcd(g_1, \dots, g_k))$: an irreducible $h$ defines a component of
codimension one exactly when $V(h) \subseteq V(g_i)$ for every $i$, that is, since $(h)$ is prime,
when $h$ divides every $g_i$. A factor of only some of the generators need not vanish on any
hypersurface in the locus.

Fevola, Mizera and Telen (arXiv:2311.16219) list, for each diagram of their database, the
components of the principal Landau determinant and how each was computed. On the database's own
$U + F$, eliminating faces of up to 30 points and so skipping only the polytope itself:

| Entry | Components | computed from faces | every factor of every generator | gcd |
|---|---|---|---|---|
| `par_zero_generic` | 4 | 4 | the 4 | the 4 |
| `A4_zero_generic` | 12 | 12 | the 12 | the 12 |
| `kite_generic_generic` | 13 | 10 | 14 | the 10 |

The kite's three other components were found only with HyperInt: the thresholds
$\lambda(m_1^2, m_2^2, m_5^2)$ and $\lambda(m_3^2, m_4^2, m_5^2)$ of its two bubbles and a cubic.
The old rule found the two thresholds on faces whose generators do not all vanish on them,
together with two factors that are not components. With the new rule the face computation gives
exactly the database's components from faces. On feynkit's graphs the rule
removes $m_1 \pm m_2$ and a quartic from `11e|2|e|:nnz`, a massive bubble with a massless bridge.

## The report's minors

For a one-loop graph the report compares with the closed form of Dlapa, Helmer, Papathanasiou and
Tellander (2023): the $2^{n+1} - 1$ principal minors of the modified Cayley matrix $Y$. When the
order of the cycle differs from the order of the legs, the entries $Y_{ij}$ are sums of up to
eleven invariants, and expanding the minors took about 40 s for the massless pentagon.

The minors are now computed with a fresh symbol for each distinct entry that is not constant,
as determinants in the polynomial ring of those symbols, and factored there; each factor is then
substituted back. Substitution is a ring homomorphism, so a minor is the product of the images of
its factors and its irreducible factors are theirs; a minor vanishes identically when an image
does. When the entries are linearly independent forms in atoms that are not squared, as for
massless graphs, each image is already irreducible and is only normalised; otherwise it is
factored again. The pentagon's closed form then takes a few seconds.
