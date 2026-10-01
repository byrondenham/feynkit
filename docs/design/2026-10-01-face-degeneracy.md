# Face degeneracy

Design note for `feynkit/degeneracy.py`, which decides for each face of the Newton polytope of
$G = \mathcal U + \mathcal F$ whether $G$ restricted to it has a singular point in the torus; for
`FeynmanIntegral.face_degeneracy`; for `DecoratedFace.degenerate`; and for the `degeneracy`
section of the analysis report and `fk analyse -D`.

References: Gelfand, Kapranov and Zelevinsky, *Discriminants, Resultants, and Multidimensional
Determinants* (GKZ below, with the page numbers of the book); Fevola, Mizera and Telen,
arXiv:2311.16219 (FMT); Klausen, arXiv:1910.08651 and arXiv:2109.07584; Bitoun, Bogner, Klausen
and Panzer, arXiv:1712.09215. Page numbers of papers are those of their arXiv versions.

## Definitions

Let $z$ be a rational kinematic point and $G_z = \sum_{\alpha \in A_z} c_\alpha u^\alpha$ the
polynomial $G$ there, with the energy scale set to 1 and $A_z$ the exponents whose coefficients
do not vanish at $z$. Let $P_z = \mathrm{conv}(A_z)$. For a face $F$ of $P_z$, $P_z$ itself
included, $G_z|_F = \sum_{\alpha \in A_z \cap F} c_\alpha u^\alpha$, and $F$ is **degenerate at
$z$** when
$$G_z|_F = u_1 \partial_1 G_z|_F = \dots = u_N \partial_N G_z|_F = 0$$
has a solution in $(\mathbb C^*)^N$.

The reference polytope is $P_z$, not the Newton polytope $P$ of $G$ for generic kinematics. When
coefficients vanish at $z$, $P_z$ can be smaller than $P$, and the difference
$N!\,\mathrm{Vol}(P) - N!\,\mathrm{Vol}(P_z)$ is reported as **support loss**. A vertex of $P$
whose coefficient vanishes is then not a vertex of $P_z$, so it is not counted as a degenerate
face.

On a kinematic family $\mathcal E$, an affine space of kinematic symbols on which the coefficients
of $G$ are rational functions, a face $F$ of $P$ is **generically degenerate** when it is
degenerate at a Zariski-dense set of points of $\mathcal E$.

## Elementary facts

Let $F$ be a face of dimension $d$, $\alpha_0 \in A_z \cap F$, and $L$ the lattice spanned by the
differences of the points of $A_z \cap F$, with a basis $b_1, \ldots, b_d$, the columns of an
integer $N \times d$ matrix $B$. Every $\alpha \in A_z \cap F$ is $\alpha_0 + B k(\alpha)$ with
$k(\alpha) \in \mathbb Z^d$, the **lattice coordinates** of $\alpha$; `polytope.lattice_coordinates`
gives them in the Hermite-normal-form basis of $L$. Write
$$g_F(t) = \sum_{\alpha \in A_z \cap F} c_\alpha t^{k(\alpha)}, \qquad
G_z|_F(u) = u^{\alpha_0} g_F(u^{b_1}, \ldots, u^{b_d}).$$

**Proposition 1.** $F$ is degenerate at $z$ if and only if $g_F$ has a singular point in
$(\mathbb C^*)^d$: $g_F = \partial_1 g_F = \dots = \partial_d g_F = 0$ has a solution there.

*Proof.* Let $\varphi(u) = (u^{b_1}, \ldots, u^{b_d})$, a homomorphism
$(\mathbb C^*)^N \to (\mathbb C^*)^d$. It is surjective: write $B = U D V$ in Smith normal form
with $U, V$ unimodular and $D$ diagonal with non-zero entries $\delta_j$; unimodular matrices act
on tori as automorphisms, and $t \mapsto (t_1^{\delta_1}, \ldots, t_d^{\delta_d})$ is onto. With
$t = \varphi(u)$, the chain rule gives
$$u_i \partial_i G_z|_F(u) = \alpha_{0,i}\, G_z|_F(u) + u^{\alpha_0} \sum_{j=1}^d B_{ij}\,
(t_j \partial_j g_F)(t).$$
As $u^{\alpha_0} \ne 0$, the system for $G_z|_F$ at $u$ holds exactly when $g_F(t) = 0$ and
$B w = 0$ for $w_j = (t_j \partial_j g_F)(t)$. $B$ has rank $d$, so $Bw = 0$ means $w = 0$, and as
$t_j \ne 0$ this is $\partial_j g_F(t) = 0$ for every $j$. A singular point $t$ of $g_F$ lifts to a
solution $u$ because $\varphi$ is onto. $\square$

The proof works for any basis of $L$, and a change of basis is an automorphism of
$(\mathbb C^*)^d$, so neither the verdict nor the dimension of the singular locus depends on it.

**Proposition 2.** A vertex is never degenerate, and neither is a face whose points
$A_z \cap F$ are affinely independent.

*Proof.* For a vertex, $g_F = c_{\alpha_0}$ is a non-zero constant. If $A_z \cap F =
\{\alpha_0, \ldots, \alpha_d\}$ is affinely independent, the differences $b_j = \alpha_j -
\alpha_0$ are a basis of $L$, and in it $g_F = c_{\alpha_0} + \sum_j c_{\alpha_j} t_j$ has
$\partial_j g_F = c_{\alpha_j} \ne 0$. $\square$

**Proposition 3.** Let $F$ be an edge, with lattice coordinates $0 = k_0 < \dots < k_m$. Then $F$
is degenerate exactly when the discriminant of $g_F(t) = \sum_i c_i t^{k_i}$ vanishes, and the
total Tjurina number of its singular points is $m - \deg \operatorname{sqf}(g_F)$.

*Proof.* $g_F$ has degree $k_m$, non-zero leading coefficient and $g_F(0) = c_0 \ne 0$, so its
singular points in $\mathbb C^*$ are its multiple roots in $\mathbb C$, which exist exactly when
the discriminant vanishes. A root of multiplicity $\mu$ has Tjurina number
$\dim \mathbb C[t]_p/(g_F, g_F') = \mu - 1$, and these sum to $k_m$ minus the degree of the
square-free part. $\square$

For other faces let $J = (g_F, \partial_1 g_F, \ldots, \partial_d g_F)$ and
$m = t_1 \cdots t_d$, and let $S = J : m^\infty = \{h : h m^k \in J \text{ for some } k\}$ over a
field $K$, with $\bar K$ its algebraic closure.

**Proposition 4.** (a) $g_F$ has a singular point in $(\bar K^*)^d$ exactly when $S \ne (1)$.
(b) $V(S)$ is the Zariski closure of the singular points in the torus, so $\dim S$ is the
dimension of the singular locus there. (c) When that locus is finite, $\dim_K K[t]/S$ is its
total Tjurina number.

*Proof.* (a) $S = (1)$ exactly when $m^k \in J$ for some $k$, that is $m \in \sqrt J$, which by the
Nullstellensatz means that $m$ vanishes on $V(J)$: no point of $V(J)$ lies in the torus.
(b) If $p \in V(J)$ lies in the torus and $h m^k \in J$, then $h(p) m(p)^k = 0$ with
$m(p) \ne 0$, so $V(S)$ contains the torus points of $V(J)$. If $h$ vanishes on them, $hm$
vanishes on $V(J)$, so $(hm)^r \in J$ for some $r$ and $h^r \in S$; hence $V(S)$ lies in their
closure. (c) The closure of a finite set is the set, so $V(S)$ consists of torus points. At such
a point $p$ the local rings of $S$ and $J$ agree, $m$ being a unit there, and
$\dim K[t]/S = \sum_p \dim \mathcal O_p / J \mathcal O_p$, the sum of the Tjurina numbers. $\square$

The Tjurina number is taken in the lattice coordinates of $F$, those of $L$. Over the saturated
lattice $\mathrm{aff}(F) \cap \mathbb Z^N$, which contains $L$ with some index $i$, the change of
coordinates is an isogeny of degree $i$, unramified on the torus: each singular point has $i$
preimages with the same local Tjurina number, so the total there is $i$ times the one reported.

**Proposition 5.** Let $K = \mathbb Q(\mathcal E)$, and let $F$ be a face of $P$. Then $F$ is
generically degenerate on $\mathcal E$ exactly when $S \ne (1)$ over $K$.

*Proof.* Let $U \subseteq \mathcal E$ be the dense open set where the coefficients of $G$ on $F$
are defined and non-zero, so that $F$ is a face of $P_z$ with the same points. If $S = (1)$ over
$K$, then $m^k = \sum_i a_i f_i$ with $f_i$ the generators of $J$ and $a_i \in K[t]$; at every $z$
of $U$ where no denominator of an $a_i$ vanishes, a dense open set, the identity specialises and
$F$ is not degenerate. If $S \ne (1)$, the incidence variety
$W = \{(z, t) \in U \times (\mathbb C^*)^d : J_z(t) = 0\}$ has a non-empty fibre over the generic
point of $\mathcal E$, so its projection to $\mathcal E$, constructible by Chevalley's theorem,
contains the generic point and hence a dense open set, on which $F$ is degenerate. $\square$

So a generically degenerate face is one whose incidence variety has a component projecting onto a
dense subset of $\mathcal E$: a dominant component in the sense of FMT, Definition 3.5, p. 17,
which the principal Landau determinant leaves out and `FaceDiscriminant.dominant` flags.

## Non-degeneracy and the volume

**Proposition 6.** Let $f = \sum_{\alpha \in A} c_\alpha u^\alpha$ with every $c_\alpha \ne 0$ and
$Q = \mathrm{conv}(A)$ of dimension $N$. The principal A-determinant $E_A(f)$ vanishes exactly
when some face of $Q$, $Q$ included, is degenerate for $f$. So when no face is degenerate, the
complement $X$ of $\{f = 0\}$ in $(\mathbb C^*)^N$ has $|\chi(X)| = N!\,\mathrm{Vol}(Q)$.

*Proof.* By Proposition 1 the verdict on a face depends only on the lattice its points span, so
we may take the coordinates of the lattice $A$ affinely generates, as GKZ assume. By definition
$E_A(f) = R_A(u_1 \partial_1 f, \ldots, u_N \partial_N f, f)$, an A-resultant (GKZ, Ch. 10,
(1.1), p. 297). Each argument lies in $\mathbb C^A$, a linear form on the projective space in
which the toric variety $X_A$ lies, and $R_A$ is the resultant of these forms on $X_A$ (Ch. 8,
Prop. 2.1, p. 256), which vanishes exactly when the forms have a common zero on $X_A$ (Ch. 3,
(2.2), p. 101). $X_A$ is the union of the orbits $X^0(F)$ of the faces $F$ of $Q$, and $X^0(F)$
consists of the points with coordinates $y_\omega = \lambda u^\omega$ for $\omega \in A \cap F$
and $y_\omega = 0$ otherwise, $u \in (\mathbb C^*)^N$, $\lambda \ne 0$ (Ch. 5, Prop. 1.9 and its
proof, p. 171). At such a point the form of $f$ is $\lambda f|_F(u)$ and that of
$u_i \partial_i f$ is $\lambda (u_i \partial_i f|_F)(u)$, so a common zero on $X^0(F)$ is exactly
a solution of the system that makes $F$ degenerate. The last statement is FMT, Theorem 2.3,
p. 9, with $\chi(X) = -\chi(\{f = 0\})$ since $\chi$ is additive and vanishes on the torus, and
with their normalised volume equal to $N!\,\mathrm{Vol}(Q)$ for $Q$ of full dimension. $\square$

Bitoun, Bogner, Klausen and Panzer give this value for almost all coefficients (Theorem 44,
after Kouchnirenko); Proposition 6 is a test for it at a given point. The report states it at the
drawn point when no face is degenerate, and `critical_point_count` agrees with it on the property
tests.

## Algorithm

`face_degeneracy_from_polynomial(g, variables, point=None, *, scale, check, timeout)` reads the
support, evaluates the coefficients exactly at the point or keeps them as rational functions,
computes the faces of $P_z$ with `polytope_data`, and decides each face:

1. vertices and faces with affinely independent points: not degenerate (Proposition 2);
2. edges: `sympy.discriminant` of $g_F$ in the lattice coordinate, used only as a zero test,
   and the Tjurina number from the square-free part (Proposition 3);
3. other faces: $S$ with Singular's `sat`, `dim(std(S))` and, when the dimension is 0,
   `vdim` (Proposition 4).

All faces of an analysis go to Singular in one script, run from a temporary directory with a time
limit, 120 s by default. Verdicts are printed face by face. When the run passes the limit, the
faces it decided are kept, the face it was on runs alone, unless it already had the whole limit,
and the rest run together again. A face that passes the limit alone is recorded as undecided,
with the reason; it is never guessed, and `non_degenerate` is then None unless another face is
degenerate.

Without a point, the same script runs over the field of rational functions in the kinematic symbols,
a Singular ring with parameters (Proposition 5). The energy scale $\mu$ is set to 1 when every
coefficient is $\mu^{k(\alpha)}$ times a factor free of $\mu$ with $k$ affine in the exponent,
as for $G = \mathcal U + \mathcal F$: rescaling $\mu$ then acts through the torus, which maps the
singular locus of each face onto that at $\mu = 1$.

**The check over $\mathbb F_p$.** The faces that needed Singular are decided a second time over
$\mathbb F_p$, at the largest prime below $2^{29}$, the bound Singular accepts for the
characteristic of a ring with parameters, that divides no numerator or denominator of a
coefficient. A certificate $m^k = \sum_i a_i f_i$ over $\mathbb Q$ reduces modulo all but
finitely many primes, so the two verdicts differ only at finitely many primes. The verdict over
$\mathbb F_p$ is stored as `modular`, labelled probabilistic, and never decides.

## Interfaces

- `FaceDegeneracy`: the point indices of a face, its dimension, `degenerate` (None when
  undecided), `method` (`vertex`, `simplex`, `edge` or `groebner`), `singular_dimension` ($-1$
  when empty), `tjurina` (when the singular locus is finite), `modular` and `reason`.
- `DegeneracyAnalysis`: `mode` (`point` or `generic`), `point`, `points` (those of $P_z$),
  `volume` ($N!\,\mathrm{Vol}(P_z)$), `support_loss`, `faces`, `prime`, and the properties
  `degenerate_faces`, `undecided_faces` and `non_degenerate`.
- `face_degeneracy(fi, point=None, *, check=True, timeout=120)` and
  `FeynmanIntegral.face_degeneracy`, cached for each choice of the arguments. Kinematic
  constraints are refused, as `torus_count` refuses them.
- `kinematic_point(g, variables, *, scale, seed, landau)` in `feynkit.point_count`: the point
  `count_torus_points` draws with a seed, without counting.
- `FeynmanIntegral.face_lattice(..., degeneracy=True)` fills `DecoratedFace.degenerate` at the
  generic point; it is None otherwise and for the empty face.

**Report.** The `degeneracy` section decides the faces at the kinematic point of the torus counts,
drawn with the report's seed, where every coefficient of $G$ and every face discriminant found is
non-zero, so that $P_z = P$. At the generic point the decision can take minutes: in our timings the
pentagon with masses, both hexagons and the off-shell massless double box passed 120 s and the
massive kite took about 90 s, while none took more than a few seconds at a point. So the point is
the default and `degeneracy_mode="generic"` is an option. The section lists the degenerate faces
with their dimension, number of points, graph up to codimension 2, singular dimension and Tjurina
number, states Proposition 6 when no face is degenerate, and, where the torus counts give a
candidate at the same point, sets $N!\,\mathrm{Vol}(P_z) - |\chi(X)|$ beside the list. It is not
among the default sections, since it needs Singular and the Landau analysis. `fk analyse -D` prints
the same and adds the section to the reports.

## Tests

Faces worked by hand: the massive bubble at $\lambda(s, m_1^2, m_2^2) = 0$, where $G$ has the edge
$(u_1 - 2u_2)^2$; squares and cubes on an edge; the massless off-shell triangle and the massless
sunrise, with no degenerate face; the massive triangle with $p_1^2 = 0$, with one degenerate
facet at the generic point; the double bubble, whose degenerate faces have singular curves. The
shortcuts of Propositions 2 and 3 are compared with the Gröbner path on random polygons and
polytopes, the verdict over $\mathbb F_p$ with the exact one, generic mode with
`FaceDiscriminant.dominant`, and, when no face is degenerate, `critical_point_count` with the
volume (Proposition 6).

Published examples:

- FMT, Ex. 2.5 and Rem. 2.6, pp. 10-12, and Klausen, arXiv:1910.08651, Sec. 4, p. 25: for the
  fully massive sunrise the volume is 10 and the count 7, and the quadrilateral face whose
  discriminant vanishes on the kinematic space is degenerate.
- FMT, Ex. 3.9, pp. 21-22: the dense face of $(1 + x)(a + bx + cy + dxy)$ is degenerate for all
  coefficients, with a larger Tjurina number on $bc = ad$.
- FMT, Ex. 3.10, pp. 22-24, recorded: for $(y - 1)^2 - (x - z)x^2$ the only degenerate face is
  the edge on $x = 0$, at $z = 0$ as for generic $z$, and $P_z$ does not change at $z = 0$.
- FMT, Tab. 1, p. 13, and Sec. 4.2, p. 44: the bananas with two to four edges, with $2^E - 1$
  critical points against the volume $\binom{2E - 1}{E}$, where the verdict agrees with
  Proposition 6; the massless banana, with no degenerate face; the parachute, with degenerate
  faces.
- Klausen, arXiv:2109.07584, Sec. 4.3, pp. 28-29: $E_{A_\mathcal U}(\mathcal U)$ is 1 for every
  one-loop and banana graph and vanishes for the dunce's cap, whose square face with
  $\mathcal U = (u_1 + u_2)(u_3 + u_4)$ is degenerate; the degenerate faces of the sunset are
  proper faces with monomials of both $\mathcal U$ and $\mathcal F$.
