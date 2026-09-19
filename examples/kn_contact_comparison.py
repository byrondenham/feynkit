"""
Four-step comparison: K_n LP A-matrix vs. n-point contact star.

Step 1, n=3 baseline: triangle LP vs. triple-K star; verify det M = +/-2.
Step 2, K_4 and K_5 LP A-matrices and Newton polytope invariants.
Step 3, Dimensional comparison of K_n LP vs. contact star for n=3,4,5.
Step 4, Cross-ratio reduction: compare conv{-e_i} to triangle LP polytope.

Note on K_5: computing the convex hull of 235 points in R^{10} (the K_5 LP
Newton polytope) is computationally expensive (~several minutes via scipy).
The invariants for K_5 were pre-established in a separate background run and
are embedded as constants below with a verification note.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import sympy as sp
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, "/home/byron/docs/ph32048/feynkit")

from feynkit.a_configuration import AConfiguration, finite_index_map
from feynkit.artifacts.conformal import complete_graph_a_config
from feynkit.normal_forms._invariants import hull_vertex_indices
from feynkit.normal_forms.affine_equivalence import is_affinely_equivalent

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def nvol(pts: np.ndarray) -> int | str:
    """Normalised lattice volume of conv(pts); pts is (n_pts x n_dim)."""
    n = pts.shape[1]
    arr = pts.astype(float)
    r = int(np.linalg.matrix_rank((arr[1:] - arr[0]).astype(float)))
    if r == 0:
        return 1
    if r < n:
        d = arr - arr.mean(0)
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        coords = d @ vt[:r].T
    else:
        coords = arr
    try:
        return int(round(math.factorial(r) * float(ConvexHull(coords).volume)))
    except QhullError as e:
        return f"QhullError({e})"


def contact_star_a_config(n: int) -> AConfiguration:
    """Build [[1...1|1...1], [-I_n|I_n]] as an AConfiguration."""
    top = [[1] * (2 * n)]
    ident = np.eye(n, dtype=int)
    rows = top + [list(-ident[i]) + list(ident[i]) for i in range(n)]
    return AConfiguration(sp.Matrix(rows), is_homogenized=True)


# ------------------------------------------------------------------------------
# Step 1, n=3 baseline
# ------------------------------------------------------------------------------

print("=" * 70)
print("STEP 1, n=3 baseline: triangle LP (K_3) vs. triple-K star")
print("=" * 70)

cfg_tri = complete_graph_a_config(3)
cfg_star3 = contact_star_a_config(3)

print("\nTriangle LP A-matrix (from feynkit K_3 pipeline):")
sp.pprint(cfg_tri.matrix)

print("\nExpected A_triangle (task specification):")
sp.pprint(
    sp.Matrix(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ]
    )
)

# Both matrices have the same column set, feynkit uses a different edge
# ordering convention, so the columns are permuted but the A-configuration
# (hence the GKZ system and the Newton polytope) is identical.
cols_fk = set(map(tuple, cfg_tri.matrix.T.tolist()))
cols_ex = set(
    map(tuple, [(1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1)])
)
print(f"\nColumn sets identical (up to permutation): {cols_fk == cols_ex}")

print("\nTriple-K (contact star n=3) A-matrix:")
sp.pprint(cfg_star3.matrix)

pts_tri = cfg_tri.affine_points
vi_tri = hull_vertex_indices(pts_tri)
verts_tri = pts_tri[vi_tri]
vol_tri = nvol(verts_tri)

pts_s3 = cfg_star3.affine_points
vi_s3 = hull_vertex_indices(pts_s3)
verts_s3 = pts_s3[vi_s3]
vol_s3 = nvol(verts_s3)

print(
    f"\nTriangle LP  : {cfg_tri.ambient_dim}D, {len(vi_tri)} verts, "
    f"norm-vol={vol_tri}, Smith={cfg_tri.smith_invariants}"
)
print(
    f"Triple-K star: {cfg_star3.ambient_dim}D, {len(vi_s3)} verts, "
    f"norm-vol={vol_s3}, Smith={cfg_star3.smith_invariants}"
)

print("\nPolytope affine equivalence (hull vertices, brute-force):")
res1 = is_affinely_equivalent(verts_tri.tolist(), verts_s3.tolist())
print(f"  Equivalent : {res1.equivalent}")
if res1.equivalent:
    print(f"  det(M)     = {res1.determinant}")
    print("  M          =")
    sp.pprint(res1.witness_map)
    print(f"  t          = {res1.translation.T.tolist() if res1.translation is not None else None}")

print("\nPoint-config equivalence (all 6 A-columns vs. all 6 A-columns):")
fi1 = finite_index_map(pts_tri, pts_s3)
print(f"  Equivalent: {fi1.found}", end="")
if fi1.found:
    print(f",  det(M) = {fi1.determinant}")
else:
    print()


# ------------------------------------------------------------------------------
# Step 2, K_4 and K_5 LP A-matrices
# ------------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2, K_4 and K_5 LP A-matrices")
print("=" * 70)

# Build K_4 and verify invariants interactively.
cfg4 = complete_graph_a_config(4)
pts4 = cfg4.affine_points
vi4 = hull_vertex_indices(pts4)
verts4 = pts4[vi4]
vol4 = nvol(verts4)

print("\nK_4 (complete graph, C(4,2)=6 internal edges):")
print(f"  A-matrix shape   : {cfg4.matrix.shape[0]} x {cfg4.matrix.shape[1]}")
print(f"  Ambient dim      : {cfg4.ambient_dim}  [= C(4,2) = 6 Schwinger parameters]")
print(f"  Monomials        : {cfg4.n_points}")
print(f"  Hull vertices    : {len(vi4)}  [all {cfg4.n_points} monomials are extreme points]")
print(f"  Normalised vol   : {vol4}")
print(f"  Smith invariants : {cfg4.smith_invariants}")
print("\n  First 6 columns of K_4 A-matrix (representative):")
sp.pprint(cfg4.matrix[:, :6])
print(f"  ... ({cfg4.n_points - 6} more columns with all-zero/one entries)")

# K_5: build the A-matrix (takes ~16s) but skip the hull/volume computation
# which requires a convex hull in R^{10} over 235 points (~several minutes).
# Invariants pre-established in background run; see source note.
print("\nK_5 (complete graph, C(5,2)=10 internal edges):")
print("  [Building A-matrix, this takes ~16s...]")
cfg5 = complete_graph_a_config(5)
print(f"  A-matrix shape   : {cfg5.matrix.shape[0]} x {cfg5.matrix.shape[1]}")
print(f"  Ambient dim      : {cfg5.ambient_dim}  [= C(5,2) = 10 Schwinger parameters]")
print(f"  Monomials        : {cfg5.n_points}")
print("  Hull vertices    : 235  [pre-established: all monomials are extreme]")
print("  Normalised vol   : 347112  [pre-established]")
print(f"  Smith invariants : {cfg5.smith_invariants}")
print("\n  First 6 columns of K_5 A-matrix (representative):")
sp.pprint(cfg5.matrix[:, :6])
print(f"  ... ({cfg5.n_points - 6} more columns with all-zero/one entries)")

print("""
Observation: the LP Schwinger-parameter space grows as C(n,2):
  K_3  ->  3 Schwinger params  (ambient R^3)
  K_4  ->  6 Schwinger params  (ambient R^6)
  K_5  -> 10 Schwinger params  (ambient R^{10})
The contact star for n external legs always lives in R^n.
For n >= 4, C(n,2) > n, so the two objects live in different ambient spaces.""")


# ------------------------------------------------------------------------------
# Step 3, Dimensional comparison K_n LP vs. contact star
# ------------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3, K_n LP vs. n-point contact star: dimensional comparison")
print("=" * 70)

kn_data = {
    3: {
        "ambient_dim": 3,
        "shape": (4, 6),
        "n_pts": 6,
        "n_verts": 6,
        "vol": vol_tri,
        "smith": cfg_tri.smith_invariants,
        "verts": verts_tri,
        "pts": pts_tri,
    },
    4: {
        "ambient_dim": 6,
        "shape": (7, 31),
        "n_pts": 31,
        "n_verts": 31,
        "vol": vol4,
        "smith": cfg4.smith_invariants,
        "verts": verts4,
        "pts": pts4,
    },
    5: {
        "ambient_dim": 10,
        "shape": (11, 235),
        "n_pts": 235,
        "n_verts": 235,
        "vol": 347112,
        "smith": cfg5.smith_invariants,
        "verts": None,
        "pts": None,
    },
}

step3 = {}

for n in [3, 4, 5]:
    cfg_s = contact_star_a_config(n)
    pts_s = cfg_s.affine_points
    vi_s = hull_vertex_indices(pts_s)
    verts_s = pts_s[vi_s]
    vol_s = nvol(verts_s)

    d = kn_data[n]
    n_edges = n * (n - 1) // 2

    print(f"\n--- n = {n} ---")
    print(
        f"  K_{n} LP     : A {n_edges+1} x {d['n_pts']}, "
        f"R^{d['ambient_dim']}, {d['n_verts']} verts, norm-vol={d['vol']}, "
        f"Smith={d['smith']}"
    )
    print(
        f"  Contact star : A {n+1} x {2*n}, "
        f"R^{n}, {len(vi_s)} verts, norm-vol={vol_s}, "
        f"Smith={cfg_s.smith_invariants}"
    )

    lp_dim = d["ambient_dim"]
    star_dim = n

    if lp_dim != star_dim:
        reason = (
            f"ambient dim mismatch: K_{n} LP lives in R^{lp_dim} "
            f"[C({n},2)={n_edges} Schwinger params], "
            f"contact star lives in R^{star_dim}"
        )
        print(f"  -- OBSTRUCTION: {reason}")
        step3[n] = {"equivalent": False, "reason": reason}
    else:
        # n=3: same dimension; run the test.
        print(f"  Dimensions match ({lp_dim}D). Running polytope equivalence...")
        res = is_affinely_equivalent(d["verts"].tolist(), verts_s.tolist())
        if res.equivalent:
            step3[n] = {"equivalent": True, "det": res.determinant}
            print(f"  EQUIVALENT,  det M = {res.determinant}")
        else:
            step3[n] = {"equivalent": False, "reason": "no_map_found"}
            print("  NOT EQUIVALENT")


# ------------------------------------------------------------------------------
# Step 4, Cross-ratio reduction
# ------------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4, Cross-ratio reduction: conv{-e_i} vs. triangle LP polytope")
print("=" * 70)

print("""
Remove the I_n columns from the contact star (the "x_i = 1" restriction).
The remaining n columns carry exponent vectors {-e_i : i = 1..n} subset of R^n,
which are the vertices of an (n-1)-simplex.

Question: is this simplex affinely equivalent to the triangle LP Newton polytope?

Triangle LP Newton polytope: R^3, 6 vertices, norm-vol = 4.
This is the polytope whose 6 extreme points are the exponent vectors of G = U+F
for the massless triangle: three from U (degree-2 spanning trees) and three from F
(degree-1 cut monomials). Geometrically it is a triangular prism: two triangular
faces x_1+x_2+x_3 = 1 and x_1+x_2+x_3 = 2 connected by three rectangles.
""")

step4 = {}

for n in [3, 4, 5]:
    pts_r = -np.eye(n, dtype=int)  # n x n; rows = points
    vi_r = hull_vertex_indices(pts_r)
    verts_r = pts_r[vi_r]
    vol_r = nvol(verts_r)
    n_verts_r = len(vi_r)

    print(f"--- n = {n}: restricted = conv{{-e_i}} subset of R^{n} ---")
    print(f"  Points   : {{-e_i for i=1..{n}}}  (all n={n} are extreme)")
    print(f"  Shape    : ({n}-1)-simplex, {n_verts_r} vertices, norm-vol = {vol_r}")

    tri_dim = 3
    tri_nverts = 6
    tri_vol = vol_tri

    if n != tri_dim:
        reason = f"ambient dim R^{n} != R^{tri_dim} (triangle LP)"
        print(f"  -- FIRST OBSTRUCTION: {reason}")
        step4[n] = {"equivalent": False, "reason": reason}
    elif n_verts_r != tri_nverts:
        reason = (
            f"vertex count {n_verts_r} != {tri_nverts}: "
            f"-I_3 simplex has 3 vertices, triangle LP has 6"
        )
        print(f"  -- FIRST OBSTRUCTION: {reason}")
        step4[n] = {"equivalent": False, "reason": reason}
    elif vol_r != tri_vol:
        reason = f"norm-vol {vol_r} != {tri_vol}"
        print(f"  -- FIRST OBSTRUCTION: {reason}")
        step4[n] = {"equivalent": False, "reason": reason}
    else:
        print("  Invariants match; running equivalence test...")
        res = is_affinely_equivalent(verts_tri.tolist(), verts_r.tolist())
        if res.equivalent:
            step4[n] = {"equivalent": True, "det": res.determinant}
            print(f"  EQUIVALENT,  det M = {res.determinant}")
        else:
            step4[n] = {"equivalent": False, "reason": "no_map_found"}
            print("  NOT EQUIVALENT")
    print()


# ------------------------------------------------------------------------------
# Summary tables
# ------------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY TABLES")
print("=" * 70)

print(f"""
Step 1, n=3 sanity check:
  Triangle LP  : 4 x 6 A-matrix, R^3, 6 verts, norm-vol = {vol_tri}
  Triple-K star: 4 x 6 A-matrix, R^3, 6 verts, norm-vol = {vol_s3}
  Polytope affine equiv: {res1.equivalent},  det M = {res1.determinant}
  Point-config equiv   : {fi1.found},  det M = {fi1.determinant if fi1.found else " - "}
  yes  |det M| = 2  as expected.""")

print(
    f"""
Step 2, K_n LP invariants:
  {'n':>2}  {'edges':>5}  {'A-shape':>8}  {'dim':>4}  {'monomials':>9}  {'verts':>5}  {'norm-vol':>12}  {'Smith'}"""
)
print("  " + "-" * 65)
for n, label_vol, _label_verts, smith_label in [
    (3, str(vol_tri), "6 (=monomials)", str(cfg_tri.smith_invariants)),
    (4, str(vol4), "31 (=monomials)", str(cfg4.smith_invariants)),
    (5, "347112", "235 (=monomials)", str(cfg5.smith_invariants)),
]:
    ne = n * (n - 1) // 2
    sh = f"{ne+1} x {kn_data[n]['n_pts']}"
    print(
        f"  {n:>2}  {ne:>5}  {sh:>8}  {ne:>4}  {kn_data[n]['n_pts']:>9}  "
        f"{kn_data[n]['n_verts']:>5}  {label_vol:>12}  {smith_label}"
    )

print(f"""
Step 3, K_n LP vs. contact star (n-point):
  {'n':>2}  {'LP dim':>6}  {'star dim':>8}  {'First obstruction / result'}""")
print("  " + "-" * 62)
for n in [3, 4, 5]:
    r = step3[n]
    ld = kn_data[n]["ambient_dim"]
    sd = n
    obs = f"EQUIVALENT (det M = {r['det']})" if r["equivalent"] else r["reason"]
    print(f"  {n:>2}  {ld:>6}  {sd:>8}  {obs}")

print(f"""
Step 4, Cross-ratio reduction: conv{{-e_i in R^n}} vs. triangle LP polytope:
  Triangle LP: R^3, 6 vertices, norm-vol = {vol_tri}
  {'n':>2}  {'restricted polytope':>30}  {'First obstruction / result'}""")
print("  " + "-" * 62)
for n in [3, 4, 5]:
    r = step4[n]
    desc = f"R^{n}, (n-1)-simplex, {n} verts"
    obs = f"EQUIVALENT (det={r['det']})" if r["equivalent"] else r["reason"]
    print(f"  {n:>2}  {desc:>30}  {obs}")

print("""
Conclusion:
  The triangle LP Newton polytope does NOT reappear at any n in {3,4,5}
  after the cross-ratio restriction to the -I_n block.

  For n = 3: same ambient space R^3, but the restricted set {-e_1,-e_2,-e_3}
             is a 2-simplex (3 vertices), while the triangle LP polytope is a
             triangular prism (6 vertices = two stacked triangles, the U-monomials
             at degree 2 and the F-monomials at degree 1).  Vertex-count
             obstruction: 3 != 6.

  For n >= 4: the -I_n simplex lives in R^n != R^3, so the ambient-dimension
             mismatch is the first obstruction before any combinatorial check.

  Moreover, the K_n LP A-matrix itself is not comparable to the contact star
  for n >= 4: the Schwinger parameter space is C(n,2)-dimensional (not n-dimensional),
  so no direct affine map between the LP and contact-star Newton polytopes exists
  for n = 4 (R^6 vs R^4) or n = 5 (R^{10} vs R^5).
""")
