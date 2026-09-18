"""
feynkit — complete feature walkthrough for dissertation.

Covers every public-facing capability in a single linear narrative:

  §1   Graph construction (manual edges + Nickel notation)
  §2   Symanzik polynomials  U, F, G
  §3   Three parametrisations  (Schwinger · Feynman · Lee–Pomeransky)
  §4   GKZ system  (A-matrix · Euler operators · β-parameters)
  §5   Newton polytope  (support · hull vertices · normalised volume · Smith)
  §6   Toric ideal  (generators · IBP interpretation)
  §7   Polytope automorphisms  (Aut(P) · orbits)
  §8   Symmetry pairs  (integer affine self-maps · transformation identities)
  §9   Intrinsic lattice model  (SNF basis · intrinsic coordinates)
  §10  Polytope equivalence  (unimodular · affine · point-configuration)
  §11  Finite-index map  (triangle → triple-K, det = 2)
  §12  Pairing-matrix canonical form  (Grinis–Kasprzyk)
  §13  Landau singularity analysis  (edge-part principal A-determinant)
  §14  with_()  — varying kinematics without rebuilding
  §15  AConfiguration  — standalone GKZ matrix objects
  §16  Conformal artifacts  (BMS simplex · conformal companion)
  §17  Database  (store · lookup · find_equivalent)
  §18  Text and LaTeX output

Primary working example: massless triangle (1-loop, 3-point).
Supporting examples: bubble (1-loop, 2-point) and massive sunrise (2-loop).
"""

import sympy as sp

from feynkit import (
    Edge,
    FeynmanIntegral,
    Graph,
    PolytopeEquivalence,
    PolytopeAutomorphisms,
    AConfiguration,
    FiniteIndexResult,
    finite_index_map,
    intrinsic_lattice_model,
    symmetry_pairs,
    landau_analysis,
)
from feynkit.artifacts.conformal import (
    bms_simplex_a_config,
    conformal_companion_a_config,
    massless_polygon_a_config,
)
from feynkit.artifacts.dissertation import (
    triangle_a_config,
    triple_k_a_config,
    four_point_simplex_a_config,
    banana3_a_config,
)
from feynkit.normal_forms import (
    is_unimodular_equivalent,
    is_affinely_equivalent,
    is_point_config_equivalent,
    maximal_pairing_matrix,
)
from feynkit.normal_forms._invariants import hull_vertex_indices

W = 70


def hdr(n: int, title: str) -> None:
    print(f"\n{'='*W}\n  §{n}  {title}\n{'='*W}")


def sec(title: str) -> None:
    print(f"\n── {title} {'─'*(W-4-len(title))}")


# ─────────────────────────────────────────────────────────────────────────────
# §1  Graph construction
# ─────────────────────────────────────────────────────────────────────────────
hdr(1, "Graph construction")

# Method A: manual edge list.
# Internal edges carry a mass (symbolic or sp.Integer(0)) and a propagator
# exponent ν.  External legs have is_internal=False and no mass/ν.
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

# Method B: from a Nickel / colored-Nickel string.
# The cnickel encodes topology + mass pattern compactly.
fi_tri_b = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")  # zzz = all massless
fi_bubble = FeynmanIntegral.from_cnickel("11e|e|:zz")  # massless bubble
fi_sunrise = FeynmanIntegral.from_cnickel("111e|e|:nnn")  # massive sunrise

sec("Nickel / cnickel constructors")
print(f"  Massless triangle (cnickel) : {fi_tri_b.nickel_index}  →  loops = {fi_tri_b.loop_count}")
print(
    f"  Massless bubble             : {fi_bubble.nickel_index}  →  loops = {fi_bubble.loop_count}"
)
print(
    f"  Massive sunrise             : {fi_sunrise.nickel_index}  →  loops = {fi_sunrise.loop_count}"
)


# ─────────────────────────────────────────────────────────────────────────────
# §2  Symanzik polynomials
# ─────────────────────────────────────────────────────────────────────────────
hdr(2, "Symanzik polynomials  U, F, G")

sym = fi_tri.symanzik

sec("First Symanzik polynomial  U  (topology)")
print(f"  U = {sym.u}")
print(f"  U encodes spanning trees of the internal graph.")
print(f"  For the triangle: three spanning trees, each missing one edge.")

sec("Second Symanzik polynomial  F  (kinematics + masses)")
print(f"  F = {sym.f}")
print(f"  F encodes external momenta via 2-forests.")

sec("Lee–Pomeransky polynomial  G = U + F  (in u-variables)")
print(f"  G = {sym.g}")
print(f"  LP variables: {sym.lp_parameters}")
print(f"  G is the input to the GKZ hypergeometric system.")

sec("Bubble and sunrise for comparison")
print(f"  Bubble  U = {fi_bubble.symanzik.u}")
print(f"  Bubble  F = {fi_bubble.symanzik.f}")
print(f"  Sunrise U = {fi_sunrise.symanzik.u}")
print(f"  Sunrise F = {fi_sunrise.symanzik.f}")


# ─────────────────────────────────────────────────────────────────────────────
# §3  Parametrisations
# ─────────────────────────────────────────────────────────────────────────────
hdr(3, "Three parametrisations")

sec("Schwinger parametrisation  (integration over R_{>0}^E)")
sch = fi_tri.schwinger
print(f"  Prefactor  : {sch.prefactor}")
print(f"  Parameters : {sch.parameters}   (Schwinger α variables)")
print(f"  Measure    : {sch.measure}")

sec("Feynman parametrisation  (simplex constraint)")
fey = fi_tri.feynman
print(f"  Prefactor  : {fey.prefactor}")
print(f"  Parameters : {fey.parameters}   (Feynman x variables, Σxᵢ = 1)")
print(f"  Constraints: {fey.constraints}")

sec("Lee–Pomeransky parametrisation  (u ∈ R_{>0}^E, no constraint)")
lp = fi_tri.lee_pomeransky
print(f"  Prefactor  : {lp.prefactor}")
print(f"  Parameters : {lp.parameters}   (LP u variables)")
print(f"  The integrand is u^β · G^{{-β₀}} with G = U + F.")


# ─────────────────────────────────────────────────────────────────────────────
# §4  GKZ system
# ─────────────────────────────────────────────────────────────────────────────
hdr(4, "GKZ system")

gkz = fi_tri.gkz
r, m_cols = gkz.a_matrix.shape

sec("A-matrix")
print(
    f"  Shape: {r} × {m_cols}  "
    f"({fi_tri.loop_count + 1} rows = loops+1; {m_cols} cols = monomials of G)"
)
sp.pprint(gkz.a_matrix)
print(f"  First row is the homogenisation row (all ones).")
print(f"  Remaining rows give the exponent of each u_i in each monomial.")

sec("GKZ parameters")
print(f"  z-variables  : {gkz.z_variables}")
print(f"  β-parameters : {gkz.beta_parameters}")
print(f"  β encodes dimension D and propagator exponents νᵢ via:")
print(f"    β₀ = (L·D)/2 − Σνᵢ")
print(f"    βₖ = νₖ   (one per Schwinger parameter)")

sec("Euler differential equations  Ê_r · I_A = β_r · I_A")
for i, eq in enumerate(gkz.euler_equations):
    print(f"  [{i}]  {eq}")
print(f"  Each equation: (Σⱼ Aᵣⱼ · zⱼ ∂/∂zⱼ) I_A = βᵣ · I_A")

sec("GKZ for bubble and sunrise")
gkz_b = fi_bubble.gkz
gkz_s = fi_sunrise.gkz
print(f"  Bubble  A-matrix shape : {gkz_b.a_matrix.shape};  β = {gkz_b.beta_parameters}")
sp.pprint(gkz_b.a_matrix)
print(f"  Sunrise A-matrix shape : {gkz_s.a_matrix.shape};  β = {gkz_s.beta_parameters}")
sp.pprint(gkz_s.a_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# §5  Newton polytope
# ─────────────────────────────────────────────────────────────────────────────
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
print("    (= holonomic rank = #independent master integrals for generic β)")
print(f"  Smith invariants of A   : {cfg_tri.smith_invariants}")
print(f"    (Smith = [1,1,1] → A-columns span the full ℤ³ lattice)")

sec("Intrinsic model — same computation, explicit SNF basis")
model = cfg_tri.intrinsic_model()
print(f"  Base point       : {model.base_point}")
print(f"  Intrinsic rank   : {model.intrinsic_rank}")
print(f"  Smith invariants : {model.smith_invariants}")
print(f"  Intrinsic coords (first 3): {list(model.intrinsic_coords[:3])}")


# ─────────────────────────────────────────────────────────────────────────────
# §6  Toric ideal
# ─────────────────────────────────────────────────────────────────────────────
hdr(6, "Toric ideal  (IBP relations)")

ti = fi_tri.toric_ideal

sec("Generators in z-variables")
print(f"  Number of generators: {len(ti.generators)}")
for i, gen in enumerate(ti.generators):
    print(f"  [{i}]  {gen} = 0")
print(f"  z-variables: {ti.z_variables}")

sec("IBP interpretation")
print(f"  Each generator  z^u − z^v = 0  (u,v ∈ ℤ^m, Au = Av)")
print(f"  encodes an integration-by-parts identity for I_A:")
print(f"  the differential operator ∂^u − ∂^v annihilates I_A.")
print(f"  The toric ideal is the kernel of the ring map ℤ[z] → ℤ[t^±1]")
print(f"  defined by  zⱼ ↦ t^{{aⱼ}}  (aⱼ = j-th column of A).")

sec("Bubble toric ideal (trivial case)")
ti_b = fi_bubble.toric_ideal
print(f"  Bubble generators: {len(ti_b.generators)}")
if len(ti_b.generators) == 0:
    print(f"  (empty — bubble GKZ system has no toric relations; it is a master integral)")


# ─────────────────────────────────────────────────────────────────────────────
# §7  Polytope automorphisms
# ─────────────────────────────────────────────────────────────────────────────
hdr(7, "Polytope automorphisms  Aut(P)")

aut = fi_tri.polytope_automorphisms

sec("Unimodular automorphism group")
print(f"  |Aut(P)| = {aut.order}")
print(f"  Each (U, t) satisfies:  U ∈ GL_n(ℤ),  |det U| = 1,  {{Uv + t}} = vertices")
for k, (U, t) in enumerate(aut.maps[:3]):
    print(f"\n  [{k}]  U =")
    sp.pprint(U)
    print(f"       t = {t.T.tolist()}")
if aut.order > 3:
    print(f"\n  … ({aut.order - 3} further automorphisms)")

sec("Vertex orbits under Aut(P)")
print(f"  Orbits (indices into hull vertex list):")
for orb in aut.vertex_orbits:
    print(f"    {orb}")
print(f"  The triangle LP Newton polytope has {len(aut.vertex_orbits)} orbits:")
print(f"  {3} U-monomials (degree-2 face) and {3} F-monomials (degree-1 face)")
print(f"  form separate orbits under the symmetry group.")

sec("Graph automorphisms (subgroup)")
g_aut = fi_tri.graph_automorphisms
print(f"  |Aut(graph)| = {len(g_aut)}")
print(f"  These are the vertex permutations of the triangle that preserve its")
print(f"  edge structure — the dihedral group D₃ (order 6 for the equilateral")
print(f"  triangle), embedded in Aut(P).")


# ─────────────────────────────────────────────────────────────────────────────
# §8  Symmetry pairs  (integer affine self-maps of the A-configuration)
# ─────────────────────────────────────────────────────────────────────────────
hdr(8, "Symmetry pairs  — transformation identities")

sec("Definition")
print(f"  A symmetry pair (M, t, P) satisfies  T·A = A·Π_P")
print(f"  where T = [[1, 0ᵀ], [t, M]] ∈ GL_{{n+1}}(ℤ)  and  Π_P is a column permutation.")
print(f"  It gives the integral identity:")
print(f"    I_A(β, z) = I_A(T β, z_P)")
print(f"  Unimodular maps (|det M| = 1) are polytope automorphisms;")
print(f"  non-unimodular maps (|det M| > 1) are finite-index self-embeddings.")

sp_list = fi_tri.symmetry_pairs

sec(f"Triangle LP symmetry pairs  ({len(sp_list)} total)")
unimod = [s for s in sp_list if s.is_unimodular]
nonuni = [s for s in sp_list if not s.is_unimodular]
print(f"  Unimodular (|det| = 1): {len(unimod)}")
print(f"  Finite-index (|det| > 1): {len(nonuni)}")

print(f"\n  First unimodular pair:")
s0 = unimod[0]
print(f"    M =")
sp.pprint(s0.linear_map)
print(f"    t = {s0.translation.T.tolist()[0]}")
print(f"    column permutation P = {s0.column_permutation}")
print(f"    Integral identity: I_A(β, z) = I_A(T·β, z_P)")

sec("Using AConfiguration.symmetry_pairs  (standalone)")
sp_cfg = cfg_tri.symmetry_pairs()
print(
    f"  Same result via AConfiguration: {len(sp_cfg)} pairs "
    f"({sum(s.is_unimodular for s in sp_cfg)} unimodular)"
)


# ─────────────────────────────────────────────────────────────────────────────
# §9  Intrinsic lattice model
# ─────────────────────────────────────────────────────────────────────────────
hdr(9, "Intrinsic lattice model")

sec("Concept")
print(f"  Express a point configuration in the basis of the lattice it spans.")
print(f"  Removes the accident of the ambient ℤⁿ embedding.")
print(f"  Smith decomposition: D = U·diffs·V  (integer SNF).")
print(f"  Intrinsic rank = affine dimension = {cfg_tri.affine_dim}.")

model_tri = intrinsic_lattice_model(cfg_tri.affine_points)
print(f"\n  Triangle LP:")
print(f"    base point        : {model_tri.base_point}")
print(f"    Smith invariants  : {model_tri.smith_invariants}")
print(f"    intrinsic rank    : {model_tri.intrinsic_rank}")
print(f"    intrinsic coords  :")
for c in model_tri.intrinsic_coords:
    print(f"      {c}")

# Compare with triple-K which has Smith [1,1,2].
cfg_tk = triple_k_a_config()
model_tk = intrinsic_lattice_model(cfg_tk.affine_points)
print(f"\n  Triple-K star:")
print(f"    Smith invariants  : {model_tk.smith_invariants}")
print(f"    (Smith [1,1,2] → columns span an index-2 sublattice of ℤ³,")
print(f"     the even-parity sublattice {{x : Σxᵢ ≡ 0 mod 2}}.)")


# ─────────────────────────────────────────────────────────────────────────────
# §10  Polytope equivalence
# ─────────────────────────────────────────────────────────────────────────────
hdr(10, "Polytope equivalence")

sec("Unimodular equivalence  (Liu–Cai, arXiv:2506.23846)")
print(f"  Tests: ∃ U ∈ GL_n(ℤ), t ∈ ℤⁿ  such that  {{Uv + t}} = vertices of Q.")
print(f"  Unimodular equivalence ⟹ GKZ systems have identical analytic structure.")

# Two diagrams with the same graph (different edge labelling) are unimodular.
fi_tri2 = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
uni_self = fi_tri.is_unimodular_equivalent_to(fi_tri2)
print(f"\n  Triangle (labelling A) ↔ Triangle (labelling B): {uni_self.equivalent}")
if uni_self.witness_map is not None:
    print(f"  Witness U =")
    sp.pprint(uni_self.witness_map)
    print(f"  Translation t = {uni_self.translation.T.tolist() if uni_self.translation else None}")

# Massless triangle vs massive triangle: non-equivalent.
fi_tri_mass = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
uni_mass = fi_tri.is_unimodular_equivalent_to(fi_tri_mass)
print(f"\n  Massless triangle ↔ Massive triangle: {uni_mass.equivalent}")

sec("Affine equivalence  (rational maps, hull vertices only)")
print(f"  Tests: ∃ M ∈ GL_n(ℚ), t ∈ ℚⁿ  with  M·vert(P) + t = vert(Q).")
print(f"  Weaker than unimodular; captures same combinatorial shape up to scaling.")

# Same-topology diagrams.
aff_self = fi_tri.is_affinely_equivalent_to(fi_tri2)
print(
    f"\n  Massless triangle ↔ itself (rel): {aff_self.equivalent}, " f"det = {aff_self.determinant}"
)

sec("Point-configuration equivalence  (stricter GKZ condition)")
print(f"  Tests the same map on ALL A-columns, not just hull vertices.")
print(f"  This is the correct condition for GKZ-system isomorphism.")
pts_a = cfg_tri.affine_points
pts_b = triple_k_a_config().affine_points
pc_eq = is_point_config_equivalent(pts_a, pts_b)
print(f"\n  Triangle LP ↔ Triple-K (all columns): {pc_eq.equivalent}")
if pc_eq.equivalent:
    print(f"  det M = {pc_eq.determinant}")


# ─────────────────────────────────────────────────────────────────────────────
# §11  Finite-index map  (triangle → triple-K)
# ─────────────────────────────────────────────────────────────────────────────
hdr(11, "Finite-index map  (det ≠ ±1 integer maps)")

sec("Concept")
print(f"  An integer affine map x ↦ Mx + t with M ∈ GL_n(ℤ) but |det M| > 1.")
print(f"  Such a map takes every source point to a target point, but the image")
print(f"  lattice has index |det M| in the target lattice.")
print(f"  For GKZ systems, a det = k map implies the holonomic ranks differ by k.")

fi_triangle = finite_index_map(cfg_tri, cfg_tk)

sec("Triangle LP → Triple-K  (hull vertices of each)")
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
print(
    f"  det M = {fi2.determinant}   (ratio of normalised volumes: "
    f"{cfg_tk.normalized_volume} / {cfg_tri.normalized_volume} = "
    f"{cfg_tk.normalized_volume // cfg_tri.normalized_volume})"
)
if fi2.witness_map is not None:
    print(f"  M =")
    sp.pprint(fi2.witness_map)
    print(f"  t = {fi2.translation.T.tolist() if fi2.translation else None}")

sec("Full point-configuration finite-index map")
fi_full = finite_index_map(cfg_tri, cfg_tk)
print(
    f"  Found: {fi_full.found},  det = {fi_full.determinant},  "
    f"unimodular: {fi_full.is_unimodular}"
)
if fi_full.found:
    print(f"  M =")
    sp.pprint(fi_full.witness_matrix)
    print(f"  t = {fi_full.translation.T.tolist() if fi_full.translation else None}")
    print(f"  column permutation: {fi_full.column_permutation}")


# ─────────────────────────────────────────────────────────────────────────────
# §12  Pairing-matrix canonical form
# ─────────────────────────────────────────────────────────────────────────────
hdr(12, "Pairing-matrix canonical form  (Grinis–Kasprzyk)")

sec("Definition")
print(f"  Given a matrix M, the pairing-matrix canonical form is the unique")
print(f"  lexicographically maximal form under row and column permutations.")
print(f"  Used to canonicalise Newton polytope support matrices.")

A_tri = gkz.a_matrix  # the triangle LP A-matrix
pm = maximal_pairing_matrix(A_tri)

sec("Triangle LP A-matrix")
print(f"  Input A (rows × cols) = {A_tri.shape}")
print(f"  PM_max (canonical form):")
sp.pprint(pm.PM_max)
print(f"  Row permutation    : {pm.row_permutation}")
print(f"  Column permutation : {pm.col_permutation}")
print(f"  Symmetry vector    : {pm.symmetry_vector}")

sec("Checking the canonical form is indeed maximal")
from feynkit.normal_forms import is_canonical, symbolic_compare

print(f"  is_canonical(PM_max) = {is_canonical(pm.PM_max)}")


# ─────────────────────────────────────────────────────────────────────────────
# §13  Landau singularity analysis
# ─────────────────────────────────────────────────────────────────────────────
hdr(13, "Landau singularity analysis")

sec("Method: edge-part of the principal A-determinant")
print(f"  The Landau singularities of I_A are encoded in the principal")
print(f"  A-determinant E_A.  The edge-part E_A^(1) is the product over")
print(f"  all edges of the discriminant of G restricted to each edge.")

la_tri = landau_analysis(fi_tri)

sec("Triangle")
print(f"  Landau polynomial (product of edge discriminants):")
print(f"    L = {la_tri.landau_polynomial}")
print(f"  Landau surfaces (irreducible factors):")
for i, surf in enumerate(la_tri.landau_surfaces):
    print(f"    [{i}]  {surf} = 0")
print(f"\n  Interpretation: each surface L_k = 0 is a threshold in the")
print(f"  external kinematics where the integral develops a leading singularity.")

la_b = landau_analysis(fi_bubble)
sec("Bubble")
print(f"  Landau polynomial : {la_b.landau_polynomial}")
print(f"  Landau surfaces   : {la_b.landau_surfaces}")
print(f"  (Single surface: the familiar threshold p² = 4m² for massive bubble,")
print(f"   or p² = 0 for the massless case.)")

la_s = landau_analysis(fi_sunrise)
sec("Massive sunrise")
print(f"  Landau surfaces ({len(la_s.landau_surfaces)} total):")
for i, surf in enumerate(la_s.landau_surfaces):
    print(f"    [{i}]  {surf} = 0")


# ─────────────────────────────────────────────────────────────────────────────
# §14  with_()  — varying kinematics without rebuilding
# ─────────────────────────────────────────────────────────────────────────────
hdr(14, "with_()  — parametric specialisations")

sec("Concept")
print(f"  FeynmanIntegral.with_(**overrides) returns a new instance with the")
print(f"  specified fields replaced and all cached properties cleared.")
print(f"  Used to specialise kinematics, impose equal masses, etc.")

# Specialise the triangle to equal propagator exponents.
nu = sp.Symbol("nu", positive=True)
fi_sym = fi_tri.with_(propagator_exponents={1: nu, 2: nu, 3: nu})

sec("Equal exponents  ν₁ = ν₂ = ν₃ = ν")
sym_sym = fi_sym.symanzik
print(f"  G = {sym_sym.g}")
gkz_sym = fi_sym.gkz
print(f"  β-parameters : {gkz_sym.beta_parameters}")
print(f"  (All three νᵢ collapse to a single symbol ν.)")

# Specialise dimension.
D = sp.Integer(4)
fi_4d = fi_tri.with_(dimension=D)
print(f"\n  In D = 4 dimensions:")
print(f"  β₀ = {fi_4d.gkz.beta_parameters[0]}")
print(f"  βₖ = {fi_4d.gkz.beta_parameters[1]}  (same ν-symbols, now numeric D)")


# ─────────────────────────────────────────────────────────────────────────────
# §15  AConfiguration  — standalone GKZ matrix objects
# ─────────────────────────────────────────────────────────────────────────────
hdr(15, "AConfiguration  — standalone GKZ inputs")

sec("Standard dissertation configurations")
configs = {
    "triangle (4×6)": triangle_a_config(),
    "triple-K (4×6)": triple_k_a_config(),
    "4-simplex Δ₄ (5×5)": four_point_simplex_a_config(),
    "banana₃ (4×4)": banana3_a_config(),
}

print(f"  {'Name':<26} {'shape':>8}  {'dim':>4}  {'verts':>5}  " f"{'norm-vol':>9}  {'Smith'}")
print("  " + "─" * 62)
for name, cfg in configs.items():
    print(
        f"  {name:<26} {str(cfg.matrix.shape):>8}  "
        f"{cfg.ambient_dim:>4}  {len(cfg.newton_polytope_points):>5}  "
        f"{cfg.normalized_volume:>9}  {cfg.smith_invariants}"
    )

sec("AConfiguration equivalence methods")
cfg_t = triangle_a_config()
cfg_tk = triple_k_a_config()

print(f"\n  triangle.is_unimodular_equivalent_to(triangle)")
r_uu = cfg_t.is_unimodular_equivalent_to(cfg_t)
print(f"    equivalent = {r_uu.equivalent}")

print(f"\n  triangle.is_affinely_equivalent_to(triple-K)  [hull vertices]")
r_at = cfg_t.is_affinely_equivalent_to(cfg_tk)
print(f"    equivalent = {r_at.equivalent},  det M = {r_at.determinant}")

print(f"\n  triangle.is_point_config_equivalent_to(triple-K)  [all columns]")
r_pt = cfg_t.is_point_config_equivalent_to(cfg_tk)
print(f"    equivalent = {r_pt.equivalent},  det M = {r_pt.determinant}")

print(f"\n  triangle.finite_index_map_to(triple-K)")
r_fi = cfg_t.finite_index_map_to(cfg_tk)
print(
    f"    found = {r_fi.found},  det = {r_fi.determinant},  " f"unimodular = {r_fi.is_unimodular}"
)

print(f"\n  triangle.automorphisms()")
aut_t = cfg_t.automorphisms()
print(f"    |Aut(P)| = {aut_t.order}")
print(f"    orbits   = {aut_t.vertex_orbits}")


# ─────────────────────────────────────────────────────────────────────────────
# §16  Conformal artifacts
# ─────────────────────────────────────────────────────────────────────────────
hdr(16, "Conformal artifacts  (BMS simplex · companion)")

sec("massless_polygon_a_config(n)  — 1-loop n-gon")
for n in [3, 4]:
    cfg = massless_polygon_a_config(n)
    print(f"  n={n}: {cfg.matrix.shape}, dim={cfg.ambient_dim}, " f"Smith={cfg.smith_invariants}")

sec("bms_simplex_a_config(n)  — BMS n-point conformal integral")
print(f"  G_n(u) = Σᵢ pᵢ² ∏_{{j≠i}} uⱼ  +  4 Σᵢ uᵢ² ∏_{{j≠i}} uⱼ")
print(f"  2n monomials (n lower + n upper), ambient ℝⁿ")
print(f"  Smith [1,…,1,2] → even-parity sublattice (for all n)")
for n in [3, 4]:
    cfg = bms_simplex_a_config(n)
    print(
        f"  n={n}: {cfg.matrix.shape}, dim={cfg.ambient_dim}, "
        f"norm-vol={cfg.normalized_volume}, Smith={cfg.smith_invariants}"
    )

sec("conformal_companion_a_config(n)  — finite-index companion to BMS_n")
print(f"  G_n(u) = Σᵢ ∏_{{j≠i}} uⱼ  +  Σᵢ pᵢ² uᵢ")
print(f"  Same lower monomials as BMS_n; upper monomials are the standard basis eᵢ")
for n in [3, 4]:
    cfg_comp = conformal_companion_a_config(n)
    cfg_bms = bms_simplex_a_config(n)
    fi_map = cfg_comp.finite_index_map_to(cfg_bms)
    print(
        f"  n={n}: companion → BMS_n  found={fi_map.found}, "
        f"det={fi_map.determinant if fi_map.found else '—'}"
    )
print(f"  (det=2 map exists for n=3; for n≥4 the map is broken by S_n → S_{{n-1}})")


# ─────────────────────────────────────────────────────────────────────────────
# §17  Database
# ─────────────────────────────────────────────────────────────────────────────
hdr(17, "Database  (store · lookup · find_equivalent)")

import tempfile, pathlib
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

sec("find_equivalent  — equivalence-based retrieval")
# The massive triangle has the same topology but different masses,
# so its Newton polytope differs; it should not be found.
fi_tri_m = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
db.store(fi_tri_m, label="massive_triangle")
matches = db.find_equivalent(fi_tri, relation="unimodular")
print(f"  Integrals unimodular-equivalent to massless triangle: " f"{[r.label for r in matches]}")

tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# §18  Text and LaTeX output
# ─────────────────────────────────────────────────────────────────────────────
hdr(18, "Text and LaTeX output")

sec("to_text()  — plain-text analysis report (first 20 lines)")
txt = fi_tri.to_text()
for line in txt.splitlines()[:20]:
    print(" ", line)
if txt.count("\n") > 20:
    print(f"  … ({txt.count(chr(10))} lines total)")

sec("to_latex()  — self-contained LaTeX document (preamble only)")
latex = fi_tri.to_latex()
preamble_lines = [l for l in latex.splitlines() if l.strip().startswith("\\")][:6]
for line in preamble_lines:
    print(" ", line)
print(f"  … ({latex.count(chr(10))} lines total)")
print(f"  Compile with pdflatex to get a standalone analysis document.")


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
hdr(0, "Summary — key invariants of the massless triangle")

print(f"""
  Diagram           : 1-loop massless triangle  (K₃)
  Nickel index      : {fi_tri.nickel_index}
  Loops             : {fi_tri.loop_count}
  Propagators       : {len(fi_tri.graph.get_internal_edges())}

  Symanzik U        : {fi_tri.symanzik.u}
  Symanzik F        : {fi_tri.symanzik.f}
  LP poly G         : {fi_tri.symanzik.g}

  GKZ A-matrix      : {fi_tri.gkz.a_matrix.shape[0]} × {fi_tri.gkz.a_matrix.shape[1]}
  β-parameters      : {fi_tri.gkz.beta_parameters}
  Toric generators  : {len(fi_tri.toric_ideal.generators)}

  Newton polytope   : {len(fi_tri.newton_polytope.points)} monomials, "
  {len(cfg_tri.newton_polytope_points)} hull vertices
  Ambient dim       : {cfg_tri.ambient_dim}
  Normalised vol    : {cfg_tri.normalized_volume}  (= holonomic rank for generic β)
  Smith invariants  : {cfg_tri.smith_invariants}

  |Aut(P)|          : {fi_tri.polytope_automorphisms.order}
  Symmetry pairs    : {len(fi_tri.symmetry_pairs)}  ({sum(s.is_unimodular for s in fi_tri.symmetry_pairs)} unimodular)

  Landau surfaces   : {len(la_tri.landau_surfaces)}
""")

print("=" * W)
print("  All sections completed.")
print("=" * W)
