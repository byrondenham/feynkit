"""
feynkit, complete feature walkthrough for dissertation.

Covers the main public-facing capabilities in a single linear narrative:

  section 1   Graph construction (manual edges + Nickel notation)
  section 2   Symanzik polynomials  U, F, G
  section 3   Three parametrisations  (Schwinger * Feynman * Lee-Pomeransky)
  section 4   GKZ system  (A-matrix * Euler operators * beta-parameters)
  section 5   Newton polytope  (support * hull vertices * normalised volume * Smith)
  section 6   Toric ideal  (generators * toric operators)
  section 7   Polytope automorphisms  (Aut(P) * orbits)
  section 8   Symmetry pairs  (integer affine self-maps * transformation identities)
  section 9   Intrinsic lattice model  (SNF basis * intrinsic coordinates)
  section 10  Polytope equivalence  (unimodular * affine * point-configuration)
  section 11  Finite-index map  (triangle -> triple-K, det = 2)
  section 12  Pairing-matrix canonical form  (Grinis-Kasprzyk)
  section 13  Landau singularity analysis  (principal A-determinant)
  section 14  with_() , varying kinematics without rebuilding
  section 15  AConfiguration , standalone GKZ matrix objects
  section 16  Conformal artifacts  (BMS simplex * conformal companion)
  section 17  Database  (store * lookup * find_equivalent)
  section 18  Text and LaTeX output

Primary working example: massless triangle (1-loop, 3-point).
Supporting examples: bubble (1-loop, 2-point) and massive sunrise (2-loop).
"""

import sympy as sp

from feynkit import (
    AConfiguration,
    Edge,
    FeynmanIntegral,
    Graph,
    finite_index_map,
    intrinsic_lattice_model,
    landau_analysis,
)
from feynkit.artifacts.conformal import (
    bms_simplex_a_config,
    conformal_companion_a_config,
    massless_polygon_a_config,
)
from feynkit.artifacts.dissertation import (
    banana3_a_config,
    four_point_simplex_a_config,
    triangle_a_config,
    triple_k_a_config,
)
from feynkit.normal_forms import (
    is_point_config_equivalent,
    maximal_pairing_matrix,
)

W = 70


def hdr(n: int, title: str) -> None:
    print(f"\n{'='*W}\n  section{n}  {title}\n{'='*W}")


def sec(title: str) -> None:
    print(f"\n-- {title} {'-'*(W-4-len(title))}")


# -----------------------------------------------------------------------------
# section 1  Graph construction
# -----------------------------------------------------------------------------
hdr(1, "Graph construction")

# Method A: manual edge list.
# Internal edges carry a mass (symbolic or sp.Integer(0)) and a propagator
# exponent nu.  External legs have is_internal=False and no mass/nu.
m = sp.Integer(0)  # massless
nu1, nu2, nu3 = sp.symbols("nu1 nu2 nu3", positive=True)

triangle_edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m, nu=nu1),
    Edge(idx=2, v1=2, v2=3, is_internal=True, mass=m, nu=nu2),
    Edge(idx=3, v1=3, v2=1, is_internal=True, mass=m, nu=nu3),
    Edge(idx=4, v1=1, v2=4, is_internal=False),
    Edge(idx=5, v1=2, v2=5, is_internal=False),
    Edge(idx=6, v1=3, v2=6, is_internal=False),
]
g_tri = Graph(internal_vertices=3, external_legs=3, edges=triangle_edges)
fi_tri = FeynmanIntegral(
    g_tri,
    propagator_exponents={1: nu1, 2: nu2, 3: nu3},
)

sec("Manual construction")
print(f"  Internal propagators : {len(g_tri.get_internal_edges())}")
print(f"  External legs        : {g_tri.external_legs}")
print(f"  Loop count           : {fi_tri.loop_count}")
print(f"  Nickel index         : {fi_tri.nickel_index}")

# Method B: from a Nickel / coloured Nickel string.
# The cnickel encodes topology + mass pattern compactly.
fi_tri_b = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")  # zzz = all massless
fi_bubble = FeynmanIntegral.from_cnickel("11e|e|:zz")  # massless bubble
fi_sunrise = FeynmanIntegral.from_cnickel("111e|e|:nnn")  # massive sunrise

sec("Nickel / cnickel constructors")
print(f"  Massless triangle (cnickel) : {fi_tri_b.nickel_index}  ->  loops = {fi_tri_b.loop_count}")
print(
    f"  Massless bubble             : {fi_bubble.nickel_index}  ->  loops = {fi_bubble.loop_count}"
)
print(
    f"  Massive sunrise             : {fi_sunrise.nickel_index}  ->  loops = {fi_sunrise.loop_count}"
)


# -----------------------------------------------------------------------------
# section 2  Symanzik polynomials
# -----------------------------------------------------------------------------
hdr(2, "Symanzik polynomials  U, F, G")

sym = fi_tri.symanzik

sec("First Symanzik polynomial  U  (topology)")
print(f"  U = {sym.u}")
print("  U encodes spanning trees of the internal graph.")
print("  For the triangle: three spanning trees, each missing one edge.")

sec("Second Symanzik polynomial  F  (kinematics + masses)")
print(f"  F = {sym.f}")
print("  F encodes external momenta via 2-forests.")

sec("Lee-Pomeransky polynomial  G = U + F  (in u-variables)")
print(f"  G = {sym.g}")
print(f"  LP variables: {sym.lp_parameters}")
print("  G is the input to the GKZ hypergeometric system.")

sec("Bubble and sunrise for comparison")
print(f"  Bubble  U = {fi_bubble.symanzik.u}")
print(f"  Bubble  F = {fi_bubble.symanzik.f}")
print(f"  Sunrise U = {fi_sunrise.symanzik.u}")
print(f"  Sunrise F = {fi_sunrise.symanzik.f}")


# -----------------------------------------------------------------------------
# section 3  Parametrisations
# -----------------------------------------------------------------------------
hdr(3, "Three parametrisations")

sec("Schwinger parametrisation  (integration over R_{>0}^E)")
sch = fi_tri.schwinger
print(f"  Prefactor  : {sch.prefactor}")
print(f"  Parameters : {sch.parameters}   (Schwinger alpha variables)")
print(f"  Measure    : {sch.measure}")

sec("Feynman parametrisation  (simplex constraint)")
fey = fi_tri.feynman
print(f"  Prefactor  : {fey.prefactor}")
print(f"  Parameters : {fey.parameters}   (Feynman parameters, summing to 1)")
print(f"  Constraints: {fey.constraints}")

sec("Lee-Pomeransky parametrisation  (u in R_{>0}^E, no constraint)")
lp = fi_tri.lee_pomeransky
print(f"  Prefactor  : {lp.prefactor}")
print(f"  Parameters : {lp.parameters}   (LP u variables)")
print("  The integrand is u^(nu - 1) * G(u)^(-D/2) with G = U + F.")


# -----------------------------------------------------------------------------
# section 4  GKZ system
# -----------------------------------------------------------------------------
hdr(4, "GKZ system")

gkz = fi_tri.gkz
r, m_cols = gkz.a_matrix.shape

sec("A-matrix")
print(
    f"  Shape: {r} x {m_cols}  "
    f"({len(g_tri.get_internal_edges()) + 1} rows = E+1; {m_cols} cols = monomials of G)"
)
sp.pprint(gkz.a_matrix)
print("  First row is the homogenisation row (all ones).")
print("  Remaining rows give the exponent of each u_i in each monomial.")

sec("GKZ parameters")
print(f"  z-variables  : {gkz.z_variables}")
print(f"  beta-parameters : {gkz.beta_parameters}")
print("  beta encodes dimension D and propagator exponents nu_i via:")
print("    beta_0 = -D/2")
print("    beta_k = -nu_k   (one per Schwinger parameter)")

sec("Euler differential equations  Ehat_r * I_A = beta_r * I_A")
for i, eq in enumerate(gkz.euler_equations):
    print(f"  [{i}]  {eq}")
print("  Each equation: (sum_j A_rj * z_j d/dz_j) I_A = beta_r * I_A")

sec("GKZ for bubble and sunrise")
gkz_b = fi_bubble.gkz
gkz_s = fi_sunrise.gkz
print(f"  Bubble  A-matrix shape : {gkz_b.a_matrix.shape};  beta = {gkz_b.beta_parameters}")
sp.pprint(gkz_b.a_matrix)
print(f"  Sunrise A-matrix shape : {gkz_s.a_matrix.shape};  beta = {gkz_s.beta_parameters}")
sp.pprint(gkz_s.a_matrix)


# -----------------------------------------------------------------------------
# section 5  Newton polytope
# -----------------------------------------------------------------------------
hdr(5, "Newton polytope")

np_tri = fi_tri.newton_polytope

sec("Monomial support  (all columns of A, as exponent vectors)")
print(f"  {len(np_tri.points)} monomials:")
for pt, coeff in np_tri.support:
    print(f"    u^{pt}  coeff = {coeff}")

sec("Hull vertices, volume, Smith invariants  (via AConfiguration)")
cfg_tri = triangle_a_config()  # convenience wrapper
print(f"  Newton polytope vertices: {cfg_tri.newton_polytope_points}")
print(f"  Number of vertices      : {len(cfg_tri.newton_polytope_points)}")
print(f"  Ambient dimension       : {cfg_tri.ambient_dim}")
print(f"  Affine dimension        : {cfg_tri.affine_dim}")
print(f"  Normalised volume       : {cfg_tri.normalized_volume}")
print("    (= holonomic rank for generic beta; an upper bound for the number of master integrals)")
print(f"  Smith invariants of A   : {cfg_tri.smith_invariants}")
print("    (Smith = [1,1,1] -> A-columns span the full Z^3 lattice)")

sec("Intrinsic model, same computation, explicit SNF basis")
model = cfg_tri.intrinsic_model()
print(f"  Base point       : {model.base_point}")
print(f"  Intrinsic rank   : {model.intrinsic_rank}")
print(f"  Smith invariants : {model.smith_invariants}")
print(f"  Intrinsic coords (first 3): {list(model.intrinsic_coords[:3])}")


# -----------------------------------------------------------------------------
# section 6  Toric ideal
# -----------------------------------------------------------------------------
hdr(6, "Toric ideal  (toric operators, an analogue of IBP relations)")

ti = fi_tri.toric_ideal

sec("Generators in z-variables")
print(f"  Number of generators: {len(ti.generators)}")
for i, gen in enumerate(ti.generators):
    print(f"  [{i}]  {gen} = 0")
print(f"  z-variables: {ti.z_variables}")

sec("Toric operators")
print("  Each generator  z^u - z^v  (u, v in N^m, A u = A v)  gives the operator")
print("  d^u - d^v, with d_j = d/dz_j, which annihilates the generalised integral")
print("  I_A(beta, z) with the z_j independent: an analogue of IBP relations")
print("  (Chestnov et al. 2022), not an IBP relation itself.")
print("  The toric ideal is the kernel of the ring map Z[z] -> Z[t^+/-1]")
print("  defined by  z_j -> t^{a_j}  (a_j = j-th column of A).")

sec("Bubble toric ideal (trivial case)")
ti_b = fi_bubble.toric_ideal
print(f"  Bubble generators: {len(ti_b.generators)}")
if len(ti_b.generators) == 0:
    print("  (empty: the three columns of A are linearly independent;")
    print("   that says nothing about the number of master integrals)")


# -----------------------------------------------------------------------------
# section 7  Polytope automorphisms
# -----------------------------------------------------------------------------
hdr(7, "Polytope automorphisms  Aut(P)")

aut = fi_tri.polytope_automorphisms

sec("Unimodular automorphism group")
print(f"  |Aut(P)| = {aut.order}")
print("  Each (U, t) satisfies:  U in GL_n(Z),  |det U| = 1,  {U alpha + t} = {alpha},")
print("  the exponent vectors of G")
for k, (U, t) in enumerate(aut.maps[:3]):
    print(f"\n  [{k}]  U =")
    sp.pprint(U)
    print(f"       t = {t.T.tolist()}")
if aut.order > 3:
    print(f"\n  ... ({aut.order - 3} further automorphisms)")

sec("Vertex orbits under Aut(P)")
print("  Orbits (indices into hull vertex list):")
for orb in aut.vertex_orbits:
    print(f"    {orb}")
n_orbits = len(aut.vertex_orbits)
print(f"  The triangle LP Newton polytope has {n_orbits} orbit{'' if n_orbits == 1 else 's'}.")
if n_orbits == 1:
    print("  Aut(P) maps the 3 U-monomials (degree 1) and the 3 F-monomials (degree 2)")
    print("  into one another: its elements need not preserve the degree.")

sec("Graph automorphisms (subgroup)")
g_aut = fi_tri.graph_automorphisms
print(f"  |Aut(graph)| = {len(g_aut)}")
print("  These are the vertex permutations of the triangle that preserve its")
print("  edge structure, the dihedral group D_3 of order 6, embedded in Aut(P).")


# -----------------------------------------------------------------------------
# section 8  Symmetry pairs  (integer affine self-maps of the A-configuration)
# -----------------------------------------------------------------------------
hdr(8, "Symmetry pairs , transformation identities")

sec("Definition")
print("  A symmetry pair (M, t, P) satisfies  T*A = A*Pi_P")
print("  where T = [[1, 0^T], [t, M]] in GL_{n+1}(Z)  and  Pi_P is a column permutation.")
print("  It gives the integral identity, for I_A without Gamma prefactors:")
print("    I_A(beta, z_P) = I_A(T beta, z),  z_P = (z_{P(0)}, ..., z_{P(N-1)})")
print("  Every pair has det M = +/-1: P has finite order k, so M^k = I.")
print("  Maps with |det M| > 1 relate two different configurations (finite_index_map).")

sp_list = fi_tri.symmetry_pairs

sec(f"Triangle LP symmetry pairs  ({len(sp_list)} total)")
n_unimodular = sum(s.is_unimodular for s in sp_list)
print(f"  |det M| = 1 for {n_unimodular} of {len(sp_list)} pairs, as for every self-map")

print("\n  First pair:")
s0 = sp_list[0]
print("    M =")
sp.pprint(s0.linear_map)
print(f"    t = {s0.translation.T.tolist()[0]}")
print(f"    column permutation P = {s0.column_permutation}")
print("    Integral identity: I_A(beta, z_P) = I_A(T*beta, z)")

sec("Using AConfiguration.symmetry_pairs  (standalone)")
sp_cfg = cfg_tri.symmetry_pairs()
print(
    f"  Same result via AConfiguration: {len(sp_cfg)} pairs "
    f"({sum(s.is_unimodular for s in sp_cfg)} unimodular)"
)


# -----------------------------------------------------------------------------
# section 9  Intrinsic lattice model
# -----------------------------------------------------------------------------
hdr(9, "Intrinsic lattice model")

sec("Concept")
print("  Express a point configuration in the basis of the lattice it spans.")
print("  Removes the accident of the ambient Z ^n embedding.")
print("  Smith decomposition: D = U*diffs*V  (integer SNF).")
print(f"  Intrinsic rank = affine dimension = {cfg_tri.affine_dim}.")

model_tri = intrinsic_lattice_model(cfg_tri.affine_points)
print("\n  Triangle LP:")
print(f"    base point        : {model_tri.base_point}")
print(f"    Smith invariants  : {model_tri.smith_invariants}")
print(f"    intrinsic rank    : {model_tri.intrinsic_rank}")
print("    intrinsic coords  :")
for c in model_tri.intrinsic_coords:
    print(f"      {c}")

# Compare with triple-K which has Smith [1,1,2].
cfg_tk = triple_k_a_config()
model_tk = intrinsic_lattice_model(cfg_tk.affine_points)
print("\n  Triple-K star:")
print(f"    Smith invariants  : {model_tk.smith_invariants}")
print("    (Smith [1,1,2] -> columns span an index-2 sublattice of Z^3,")
print("     the even-parity sublattice {x : sum x_i == 0 mod 2}.)")


# -----------------------------------------------------------------------------
# section 10  Polytope equivalence
# -----------------------------------------------------------------------------
hdr(10, "Polytope equivalence")

sec("Unimodular equivalence  (Liu-Cai, arXiv:2506.23846)")
print("  Tests: there exists U in GL_n(Z), t in Z ^n  such that  {Uv + t} = vertices of Q.")
print("  When the map takes every column, not just the hull vertices, onto a column,")
print("  the GKZ systems agree up to a relabelling of the z_j and beta -> T beta.")

# Two diagrams with the same graph (different edge labelling) are unimodular.
fi_tri2 = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
uni_self = fi_tri.is_unimodular_equivalent_to(fi_tri2)
print(f"\n  Triangle (labelling A) <-> Triangle (labelling B): {uni_self.equivalent}")
if uni_self.witness_map is not None:
    print("  Witness U =")
    sp.pprint(uni_self.witness_map)
    print(f"  Translation t = {uni_self.translation.T.tolist() if uni_self.translation else None}")

# Massless triangle vs massive triangle: non-equivalent.
fi_tri_mass = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
uni_mass = fi_tri.is_unimodular_equivalent_to(fi_tri_mass)
print(f"\n  Massless triangle <-> Massive triangle: {uni_mass.equivalent}")

sec("Affine equivalence  (rational maps, hull vertices only)")
print("  Tests: there exists M in GL_n(Q), t in Q ^n  with  M*vert(P) + t = vert(Q).")
print("  Weaker than unimodular; it keeps the combinatorial shape, up to a rational affine map.")

# Same-topology diagrams.
aff_self = fi_tri.is_affinely_equivalent_to(fi_tri2)
print(
    f"\n  Massless triangle <-> itself (rel): {aff_self.equivalent}, "
    f"det = {aff_self.determinant}"
)

sec("Point-configuration equivalence  (stricter GKZ condition)")
print("  Tests the same map on ALL A-columns, not just hull vertices.")
print("  Such a map makes the GKZ systems agree up to a relabelling of the z_j and")
print("  beta -> T beta, with the factor |det M| in the identity.")
pts_a = cfg_tri.affine_points
pts_b = triple_k_a_config().affine_points
pc_eq = is_point_config_equivalent(pts_a, pts_b)
print(f"\n  Triangle LP <-> Triple-K (all columns): {pc_eq.equivalent}")
if pc_eq.equivalent:
    print(f"  det M = {pc_eq.determinant}")


# -----------------------------------------------------------------------------
# section 11  Finite-index map  (triangle -> triple-K)
# -----------------------------------------------------------------------------
hdr(11, "Finite-index map  (det != +/-1 integer maps)")

sec("Concept")
print("  An integer affine map x -> Mx + t, M an integer matrix with |det M| > 1.")
print("  Such a map takes every source point to a target point, but the image")
print("  lattice has index |det M| in the target lattice.")
print("  A map of all columns with |det M| = k gives I_A(beta, z_P) = k I_B(T beta, z)")
print("  and keeps the normalised volume in the lattice the points span, so the")
print("  holonomic ranks for generic beta agree.")

fi_triangle = finite_index_map(cfg_tri, cfg_tk)

sec("Triangle LP -> Triple-K  (hull vertices of each)")
cfg_tk_verts = AConfiguration(
    sp.Matrix([[1] * 6] + [[v[i] for v in cfg_tk.newton_polytope_points] for i in range(3)]),
    is_homogenized=True,
)
cfg_tri_verts = AConfiguration(
    sp.Matrix([[1] * 6] + [[v[i] for v in cfg_tri.newton_polytope_points] for i in range(3)]),
    is_homogenized=True,
)
fi2 = cfg_tri_verts.is_affinely_equivalent_to(cfg_tk_verts)
print(f"  Equivalent (polytope): {fi2.equivalent}")
print(f"  det M = {fi2.determinant}")
if fi2.witness_map is not None:
    print("  M =")
    sp.pprint(fi2.witness_map)
    print(f"  t = {fi2.translation.T.tolist() if fi2.translation else None}")

sec("Full point-configuration finite-index map")
fi_full = finite_index_map(cfg_tri, cfg_tk)
print(
    f"  Found: {fi_full.found},  det = {fi_full.determinant},  "
    f"unimodular: {fi_full.is_unimodular}"
)
if fi_full.found:
    print("  M =")
    sp.pprint(fi_full.witness_matrix)
    print(f"  t = {fi_full.translation.T.tolist() if fi_full.translation else None}")
    print(f"  column permutation: {fi_full.column_permutation}")


# -----------------------------------------------------------------------------
# section 12  Pairing-matrix canonical form
# -----------------------------------------------------------------------------
hdr(12, "Pairing-matrix canonical form  (Grinis-Kasprzyk)")

sec("Definition")
print("  Given a matrix M, the pairing-matrix canonical form is the unique")
print("  lexicographically maximal form under row and column permutations.")
print("  Used to canonicalise Newton polytope support matrices.")

A_tri = gkz.a_matrix  # the triangle LP A-matrix
pm = maximal_pairing_matrix(A_tri)

sec("Triangle LP A-matrix")
print(f"  Input A (rows x cols) = {A_tri.shape}")
print("  PM_max (canonical form):")
sp.pprint(pm.PM_max)
print(f"  Row permutation    : {pm.row_permutation}")
print(f"  Column permutation : {pm.col_permutation}")
print(f"  Symmetry vector    : {pm.symmetry_vector}")

sec("Checking whether the canonical form is maximal")
from feynkit.normal_forms import is_canonical

print(f"  is_canonical(PM_max) = {is_canonical(pm.PM_max)}")


# -----------------------------------------------------------------------------
# section 13  Landau singularity analysis
# -----------------------------------------------------------------------------
hdr(13, "Landau singularity analysis")

sec("Method: the principal A-determinant")
print("  The singular locus of the GKZ system is the zero set of the principal")
print("  A-determinant E_A, the product over all faces of the Newton polytope")
print("  of the discriminant of G restricted to that face (GKZ 1994, ch. 10).")

la_tri = landau_analysis(fi_tri)

sec("Triangle")
print("  Reduced principal A-determinant:")
print(f"    E_A = {la_tri.principal_a_determinant}")
print("  Landau surfaces (irreducible factors):")
for i, surf in enumerate(la_tri.landau_surfaces):
    print(f"    [{i}]  {surf} = 0")
print("\n  Interpretation: each surface is a candidate singularity of the integral on")
print("  some sheet; it need not be singular on the physical sheet (Fevola, Mizera,")
print("  Telen 2023).")

la_b = landau_analysis(fi_bubble)
sec("Bubble")
print(f"  Principal A-det   : {la_b.principal_a_determinant}")
print(f"  Landau surfaces   : {la_b.landau_surfaces}")
print("  (Massless bubble: the single factor s = p^2 = 0; the massive bubble")
print("   adds the thresholds s = (m1 +/- m2)^2 and the mass singularities.)")

la_s = landau_analysis(fi_sunrise)
sec("Massive sunrise")
print(f"  Landau surfaces ({len(la_s.landau_surfaces)} total):")
for i, surf in enumerate(la_s.landau_surfaces):
    print(f"    [{i}]  {surf} = 0")


# -----------------------------------------------------------------------------
# section 14  with_() , varying kinematics without rebuilding
# -----------------------------------------------------------------------------
hdr(14, "with_() , parametric specialisations")

sec("Concept")
print("  FeynmanIntegral.with_(**overrides) returns a new instance with the")
print("  specified fields replaced and all cached properties cleared.")
print("  Used to specialise kinematics, impose equal masses, etc.")

# Specialise the triangle to equal propagator exponents.
nu = sp.Symbol("nu", positive=True)
fi_sym = fi_tri.with_(propagator_exponents={1: nu, 2: nu, 3: nu})

sec("Equal exponents  nu_1 = nu_2 = nu_3 = nu")
sym_sym = fi_sym.symanzik
print(f"  G = {sym_sym.g}")
gkz_sym = fi_sym.gkz
print(f"  beta-parameters : {gkz_sym.beta_parameters}")
print("  (All three nu_i collapse to a single symbol nu.)")

# Specialise dimension.
D = sp.Integer(4)
fi_4d = fi_tri.with_(dimension=D)
print("\n  In D = 4 dimensions:")
print(f"  beta_0 = {fi_4d.gkz.beta_parameters[0]}")
print(f"  beta_k = {fi_4d.gkz.beta_parameters[1]}  (same nu-symbols, now numeric D)")


# -----------------------------------------------------------------------------
# section 15  AConfiguration , standalone GKZ matrix objects
# -----------------------------------------------------------------------------
hdr(15, "AConfiguration , standalone GKZ inputs")

sec("Standard dissertation configurations")
configs = {
    "triangle (4 x 6)": triangle_a_config(),
    "triple-K (4 x 6)": triple_k_a_config(),
    "4-simplex Delta_4 (5 x 5)": four_point_simplex_a_config(),
    "banana_3 (4 x 4)": banana3_a_config(),
}

print(f"  {'Name':<26} {'shape':>8}  {'dim':>4}  {'verts':>5}  " f"{'norm-vol':>9}  {'Smith'}")
print("  " + "-" * 62)
for name, cfg in configs.items():
    print(
        f"  {name:<26} {str(cfg.matrix.shape):>8}  "
        f"{cfg.ambient_dim:>4}  {len(cfg.newton_polytope_points):>5}  "
        f"{cfg.normalized_volume:>9}  {cfg.smith_invariants}"
    )

sec("AConfiguration equivalence methods")
cfg_t = triangle_a_config()
cfg_tk = triple_k_a_config()

print("\n  triangle.is_unimodular_equivalent_to(triangle)")
r_uu = cfg_t.is_unimodular_equivalent_to(cfg_t)
print(f"    equivalent = {r_uu.equivalent}")

print("\n  triangle.is_affinely_equivalent_to(triple-K)  [hull vertices]")
r_at = cfg_t.is_affinely_equivalent_to(cfg_tk)
print(f"    equivalent = {r_at.equivalent},  det M = {r_at.determinant}")

print("\n  triangle.is_point_config_equivalent_to(triple-K)  [all columns]")
r_pt = cfg_t.is_point_config_equivalent_to(cfg_tk)
print(f"    equivalent = {r_pt.equivalent},  det M = {r_pt.determinant}")

print("\n  triangle.finite_index_map_to(triple-K)")
r_fi = cfg_t.finite_index_map_to(cfg_tk)
print(
    f"    found = {r_fi.found},  det = {r_fi.determinant},  " f"unimodular = {r_fi.is_unimodular}"
)

print("\n  triangle.automorphisms()")
aut_t = cfg_t.automorphisms()
print(f"    |Aut(P)| = {aut_t.order}")
print(f"    orbits   = {aut_t.vertex_orbits}")


# -----------------------------------------------------------------------------
# section 16  Conformal artifacts
# -----------------------------------------------------------------------------
hdr(16, "Conformal artifacts  (BMS simplex * companion)")

sec("massless_polygon_a_config(n) , 1-loop n-gon")
for n in [3, 4]:
    cfg = massless_polygon_a_config(n)
    print(f"  n={n}: {cfg.matrix.shape}, dim={cfg.ambient_dim}, " f"Smith={cfg.smith_invariants}")

sec("bms_simplex_a_config(n) , BMS n-point conformal integral")
print("  G_n(u) = sum_i p_i^2 prod_{j!=i} u_j  +  4 sum_i u_i^2 prod_{j!=i} u_j")
print("  2n monomials (n lower + n upper), ambient R ^n")
print("  Smith [1,...,1,2] -> even-parity sublattice (for all n)")
for n in [3, 4]:
    cfg = bms_simplex_a_config(n)
    print(
        f"  n={n}: {cfg.matrix.shape}, dim={cfg.ambient_dim}, "
        f"norm-vol={cfg.normalized_volume}, Smith={cfg.smith_invariants}"
    )

sec("conformal_companion_a_config(n) , finite-index companion to BMS_n")
print("  G_n(u) = sum_i prod_{j!=i} u_j  +  sum_i p_i^2 u_i")
print("  Same lower monomials as BMS_n; upper monomials are the standard basis e_i")
for n in [3, 4]:
    cfg_comp = conformal_companion_a_config(n)
    cfg_bms = bms_simplex_a_config(n)
    fi_map = cfg_comp.finite_index_map_to(cfg_bms)
    print(
        f"  n={n}: companion -> BMS_n  found={fi_map.found}, "
        f"det={fi_map.determinant if fi_map.found else ' - '}"
    )
print("  (every map of all columns has |det M| = 2/(n-2): an integer map exists only")
print("   for n = 3; for n = 4 the determinant is 1 but no integer map exists)")


# -----------------------------------------------------------------------------
# section 17  Database
# -----------------------------------------------------------------------------
hdr(17, "Database  (store * lookup * find_equivalent)")

import pathlib
import tempfile

from feynkit import FeynkitDatabase

# Use a temporary file so this example is self-contained.
tmp = pathlib.Path(tempfile.mktemp(suffix=".db"))

sec("Store and retrieve integrals")
db = FeynkitDatabase(str(tmp))
db.store(fi_tri, label="massless_triangle")
db.store(fi_bubble, label="massless_bubble")
db.store(fi_sunrise, label="massive_sunrise")

print(f"  Database summary:\n{db.summary()}")

rec = db.lookup(fi_tri)
print(f"  Lookup massless triangle: label = {rec.label if rec else None}")

sec("find_equivalent , equivalence-based retrieval")
# The massive triangle has the same topology but different masses,
# so its Newton polytope differs; it should not be found.
fi_tri_m = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
db.store(fi_tri_m, label="massive_triangle")
matches = db.find_equivalent(fi_tri, relation="unimodular")
print(f"  Integrals unimodular-equivalent to massless triangle: " f"{[r.label for r in matches]}")

tmp.unlink(missing_ok=True)


# -----------------------------------------------------------------------------
# section 18  Text and LaTeX output
# -----------------------------------------------------------------------------
hdr(18, "Text and LaTeX output")

sec("to_text() , plain-text analysis report (first 20 lines)")
txt = fi_tri.to_text()
for line in txt.splitlines()[:20]:
    print(" ", line)
if txt.count("\n") > 20:
    print(f"  ... ({txt.count(chr(10))} lines total)")

sec("to_latex() , self-contained LaTeX document (preamble only)")
latex = fi_tri.to_latex()
preamble_lines = [ln for ln in latex.splitlines() if ln.strip().startswith("\\")][:6]
for line in preamble_lines:
    print(" ", line)
print(f"  ... ({latex.count(chr(10))} lines total)")
print("  Compile with pdflatex to get a standalone analysis document.")


# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
hdr(0, "Summary, key invariants of the massless triangle")

print(f"""
  Diagram           : 1-loop massless triangle  (K_3)
  Nickel index      : {fi_tri.nickel_index}
  Loops             : {fi_tri.loop_count}
  Propagators       : {len(fi_tri.graph.get_internal_edges())}

  Symanzik U        : {fi_tri.symanzik.u}
  Symanzik F        : {fi_tri.symanzik.f}
  LP poly G         : {fi_tri.symanzik.g}

  GKZ A-matrix      : {fi_tri.gkz.a_matrix.shape[0]} x {fi_tri.gkz.a_matrix.shape[1]}
  beta-parameters      : {fi_tri.gkz.beta_parameters}
  Toric generators  : {len(fi_tri.toric_ideal.generators)}

  Newton polytope   : {len(fi_tri.newton_polytope.points)} monomials,
                      {len(cfg_tri.newton_polytope_points)} hull vertices
  Ambient dim       : {cfg_tri.ambient_dim}
  Normalised vol    : {cfg_tri.normalized_volume}  (= holonomic rank for generic beta)
  Smith invariants  : {cfg_tri.smith_invariants}

  |Aut(P)|          : {fi_tri.polytope_automorphisms.order}
  Symmetry pairs    : {len(fi_tri.symmetry_pairs)}  ({sum(s.is_unimodular for s in fi_tri.symmetry_pairs)} unimodular)

  Landau surfaces   : {len(la_tri.landau_surfaces)}
""")

print("=" * W)
print("  All sections completed.")
print("=" * W)
