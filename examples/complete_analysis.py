"""
Complete Feynman diagram analysis.

Analyses six diagrams of increasing complexity and compares their Newton
polytopes using affine and unimodular equivalence.

Diagrams
--------
1. Massless triangle    , 1-loop, 3 massless propagators
2. Massive triangle     , 1-loop, 3 massive propagators
3. One-mass triangle    , 1-loop, 1 massive + 2 massless propagators
4. Massless box         , 1-loop, 4 massless propagators
5. Massive sunrise      , 2-loop (banana), 3 massive propagators
6. Planar double box    , 2-loop, 7 massive propagators
7. Tetrahedron (K4)     , 3-loop, 6 massive propagators, 4-point
"""

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.algebra import is_binomial_ideal

W = 72


# -- Formatting helpers --------------------------------------------------------


def header(title: str) -> None:
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)


def section(title: str) -> None:
    pad = max(2, W - 4 - len(title))
    print(f"\n-- {title} " + "-" * pad)


def show_poly(label: str, expr: sp.Expr, max_terms: int = 12) -> None:
    terms = sp.Add.make_args(sp.expand(expr))
    if len(terms) <= max_terms:
        print(f"  {label} = {sp.expand(expr)}")
    else:
        print(f"  {label} has {len(terms)} terms (abbreviated):")
        shown = list(terms)[:3]
        print(f"    {sp.Add(*shown)} + ... ({len(terms)} total)")


# -- Analysis function ---------------------------------------------------------


def analyse(
    name: str,
    fi: FeynmanIntegral,
    *,
    show_toric: bool = True,
    toric_max: int = 6,
) -> None:
    header(name)

    g = fi.graph
    internal = g.get_internal_edges()

    section("Topology")
    print(f"  Internal propagators : {len(internal)}")
    print(f"  External legs        : {g.external_legs}")
    print(f"  Internal vertices    : {g.internal_vertices}")
    print(f"  Loop count           : {fi.loop_count}")

    section("Edge list")
    print(f"  {'idx':<5} {'v1':<4} {'v2':<4}  {'mass':<22} nu")
    for e in internal:
        mass = str(e.get_mass())
        print(f"  {e.idx:<5} {e.v1:<4} {e.v2:<4}  {mass:<22} {e.nu}")

    section("Symanzik polynomials")
    sym = fi.symanzik
    show_poly("U", sym.u)
    show_poly("F", sym.f)

    section("Lee-Pomeransky polynomial  G = U + F")
    show_poly("G", sym.g, max_terms=16)

    section("Lee-Pomeransky parametrisation")
    lp = fi.lee_pomeransky
    print(f"  Prefactor : {lp.prefactor}")
    print(f"  Parameters: {lp.parameters}")

    section("GKZ system")
    gkz = fi.gkz
    r, m = gkz.a_matrix.shape
    print(f"  A-matrix shape : {r} x {m}  (r = loops+1, m = #monomials)")
    if m <= 20:
        print("  A =")
        for row in gkz.a_matrix.tolist():
            print(f"      {row}")
    else:
        print(f"  A : {r} x {m}  [too wide to display]")
    print(f"  z-variables    : z_1, ..., z_{m}")
    print(f"  beta-parameters   : {gkz.beta_parameters}")
    if gkz.euler_equations:
        print(f"  Euler eq [0]   : {gkz.euler_equations[0]}")

    section("Newton polytope")
    np_ = fi.newton_polytope
    pts = np_.points
    print(f"  Support size (monomials) : {len(pts)}")
    if len(pts) <= 16:
        for pt in pts:
            print(f"    {pt}")
    else:
        print(f"  First 4 points: {pts[:4]}")
        print(f"  ...  ({len(pts)} total)")

    section("Toric ideal")
    if show_toric:
        ti = fi.toric_ideal
        n = len(ti.generators)
        print(f"  Generators : {n}")
        if n == 0:
            print("  (trivial, no IBP relations; this is likely a master integral)")
        else:
            print(f"  Binomial   : {is_binomial_ideal(ti.generators)}")
            for i, gen in enumerate(ti.generators[:toric_max]):
                print(f"  [{i}] {gen} = 0")
            if n > toric_max:
                print(f"  ... ({n - toric_max} further generators)")
    else:
        print(f"  A-matrix is {r} x {m}, skipped (install 4ti2 and use backend='4ti2')")

    print()


# -- Diagram constructors ------------------------------------------------------


def _triangle(masses) -> FeynmanIntegral:
    nu = sp.symbols("nu1:4", positive=True)
    edges = [
        Edge(idx=1, v1=1, v2=2, is_internal=True, mass=masses[0], nu=nu[0]),
        Edge(idx=2, v1=2, v2=3, is_internal=True, mass=masses[1], nu=nu[1]),
        Edge(idx=3, v1=3, v2=1, is_internal=True, mass=masses[2], nu=nu[2]),
        Edge(idx=4, v1=1, v2=4, is_internal=False),
        Edge(idx=5, v1=2, v2=5, is_internal=False),
        Edge(idx=6, v1=3, v2=6, is_internal=False),
    ]
    g = Graph(internal_vertices=3, external_legs=3, edges=edges)
    return FeynmanIntegral(g, propagator_exponents={i + 1: nu[i] for i in range(3)})


def massless_triangle() -> FeynmanIntegral:
    z = sp.Integer(0)
    return _triangle([z, z, z])


def massive_triangle() -> FeynmanIntegral:
    m = sp.symbols("m1:4", nonnegative=True)
    return _triangle(list(m))


def one_mass_triangle(mass_on: int = 0) -> FeynmanIntegral:
    """One massive propagator (index mass_on in {0,1,2}), two massless."""
    m = sp.Symbol("m", nonnegative=True)
    masses = [m if i == mass_on else sp.Integer(0) for i in range(3)]
    return _triangle(masses)


def massless_box() -> FeynmanIntegral:
    nu = sp.symbols("nu1:5", positive=True)
    z = sp.Integer(0)
    internal = [
        Edge(idx=i + 1, v1=(i % 4) + 1, v2=((i + 1) % 4) + 1, is_internal=True, mass=z, nu=nu[i])
        for i in range(4)
    ]
    external = [Edge(idx=5 + i, v1=i + 1, v2=5 + i, is_internal=False) for i in range(4)]
    g = Graph(internal_vertices=4, external_legs=4, edges=internal + external)
    return FeynmanIntegral(g, propagator_exponents={i + 1: nu[i] for i in range(4)})


def massive_sunrise() -> FeynmanIntegral:
    """2-loop sunrise (banana): 3 parallel propagators between 2 vertices."""
    m = sp.symbols("m1:4", nonnegative=True)
    nu = sp.symbols("nu1:4", positive=True)
    edges = [
        Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m[0], nu=nu[0]),
        Edge(idx=2, v1=1, v2=2, is_internal=True, mass=m[1], nu=nu[1]),
        Edge(idx=3, v1=1, v2=2, is_internal=True, mass=m[2], nu=nu[2]),
        Edge(idx=4, v1=1, v2=3, is_internal=False),
        Edge(idx=5, v1=2, v2=4, is_internal=False),
    ]
    g = Graph(internal_vertices=2, external_legs=2, edges=edges)
    return FeynmanIntegral(g, propagator_exponents={i + 1: nu[i] for i in range(3)})


def planar_double_box() -> FeynmanIntegral:
    """
    2-loop planar double box (H-topology), 7 propagators, 4 external legs.

    Vertex layout::

        p1-> v1 -[1]- v2 <-p2
             |         |
            [2]       [3]
             |         |
             v3 -[4]- v4
             |         |
            [5]       [6]
             |         |
        p3-> v5 -[7]- v6 <-p4
    """
    m = sp.symbols("m1:8", nonnegative=True)
    nu = sp.symbols("nu1:8", positive=True)
    edges = [
        Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m[0], nu=nu[0]),  # top
        Edge(idx=2, v1=1, v2=3, is_internal=True, mass=m[1], nu=nu[1]),  # left-top
        Edge(idx=3, v1=2, v2=4, is_internal=True, mass=m[2], nu=nu[2]),  # right-top
        Edge(idx=4, v1=3, v2=4, is_internal=True, mass=m[3], nu=nu[3]),  # middle
        Edge(idx=5, v1=3, v2=5, is_internal=True, mass=m[4], nu=nu[4]),  # left-bottom
        Edge(idx=6, v1=4, v2=6, is_internal=True, mass=m[5], nu=nu[5]),  # right-bottom
        Edge(idx=7, v1=5, v2=6, is_internal=True, mass=m[6], nu=nu[6]),  # bottom
        Edge(idx=8, v1=1, v2=7, is_internal=False),
        Edge(idx=9, v1=2, v2=8, is_internal=False),
        Edge(idx=10, v1=5, v2=9, is_internal=False),
        Edge(idx=11, v1=6, v2=10, is_internal=False),
    ]
    g = Graph(internal_vertices=6, external_legs=4, edges=edges)
    return FeynmanIntegral(g, propagator_exponents={i + 1: nu[i] for i in range(7)})


def tetrahedron() -> FeynmanIntegral:
    """
    3-loop complete graph K4 (tetrahedron), 6 propagators, 4-point function.

    All C(4,2)=6 pairs of the 4 internal vertices are connected by a
    propagator.  With one external leg per vertex this is a 3-loop
    4-point integral.
    """
    m = sp.symbols("m1:7", nonnegative=True)
    nu = sp.symbols("nu1:7", positive=True)
    edges = [
        Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m[0], nu=nu[0]),
        Edge(idx=2, v1=1, v2=3, is_internal=True, mass=m[1], nu=nu[1]),
        Edge(idx=3, v1=1, v2=4, is_internal=True, mass=m[2], nu=nu[2]),
        Edge(idx=4, v1=2, v2=3, is_internal=True, mass=m[3], nu=nu[3]),
        Edge(idx=5, v1=2, v2=4, is_internal=True, mass=m[4], nu=nu[4]),
        Edge(idx=6, v1=3, v2=4, is_internal=True, mass=m[5], nu=nu[5]),
        Edge(idx=7, v1=1, v2=5, is_internal=False),
        Edge(idx=8, v1=2, v2=6, is_internal=False),
        Edge(idx=9, v1=3, v2=7, is_internal=False),
        Edge(idx=10, v1=4, v2=8, is_internal=False),
    ]
    g = Graph(internal_vertices=4, external_legs=4, edges=edges)
    return FeynmanIntegral(g, propagator_exponents={i + 1: nu[i] for i in range(6)})


# -- Equivalence helper --------------------------------------------------------


def compare(
    name_a: str,
    fi_a: FeynmanIntegral,
    name_b: str,
    fi_b: FeynmanIntegral,
) -> None:
    pts_a = len(fi_a.newton_polytope.points)
    pts_b = len(fi_b.newton_polytope.points)
    a_shape = fi_a.gkz.a_matrix.shape
    b_shape = fi_b.gkz.a_matrix.shape

    print(f"\n  {name_a}  vs  {name_b}")
    print(f"    A-matrix : {a_shape[0]} x {a_shape[1]}  vs  {b_shape[0]} x {b_shape[1]}")
    print(f"    #monomials : {pts_a}  vs  {pts_b}")

    if pts_a != pts_b:
        print("    -> NOT equivalent (different support sizes)")
        return

    uni = fi_a.is_unimodular_equivalent_to(fi_b)
    print(f"    Unimodular equivalent : {uni.equivalent}")
    if uni.equivalent and uni.witness_map is not None:
        print(f"    Witness map U        :\n{uni.witness_map}")
    if uni.equivalent and uni.vertex_correspondence is not None:
        print(f"    Vertex correspondence: {uni.vertex_correspondence}")

    if not uni.equivalent:
        aff = fi_a.is_affinely_equivalent_to(fi_b)
        print(f"    Affine equivalent     : {aff.equivalent}")


# -- Main ----------------------------------------------------------------------

print("=" * W)
print("  COMPLETE FEYNMAN DIAGRAM ANALYSIS")
print("=" * W)

print("\nConstructing diagrams ...")
tri0 = massless_triangle()
tri1 = massive_triangle()
tri1m0 = one_mass_triangle(mass_on=0)  # mass on propagator 1 (v1-v2)
tri1m1 = one_mass_triangle(mass_on=1)  # mass on propagator 2 (v2-v3)
box0 = massless_box()
sun = massive_sunrise()
dbox = planar_double_box()
tet = tetrahedron()
print("Done.\n")

# -- Per-diagram analysis ------------------------------------------------------

analyse("1. Massless triangle  (1-loop, 3 massless propagators)", tri0)
analyse("2. Massive triangle   (1-loop, 3 massive propagators)", tri1)
analyse("3. One-mass triangle  (1-loop, mass on prop 1)", tri1m0)
analyse("4. One-mass triangle  (1-loop, mass on prop 2)", tri1m1)
analyse("5. Massless box       (1-loop, 4 massless propagators)", box0)
analyse("6. Massive sunrise    (2-loop, 3 massive propagators)", sun)
analyse("7. Planar double box  (2-loop, 7 massive propagators)", dbox, show_toric=False)
analyse("8. Tetrahedron / K4   (3-loop, 6 massive propagators)", tet, show_toric=False)

# -- Equivalence comparisons ---------------------------------------------------

header("POLYTOPE EQUIVALENCE COMPARISONS")

print("""
  Tests whether the GKZ Newton polytopes of two Feynman integrals are
  related by a unimodular (det +/-1 integer) or affine (rational) map.
  Unimodular equivalence implies the GKZ systems have identical analytic
  structure; affine equivalence is strictly weaker.
""")

print("-" * W)

compare("Massless triangle", tri0, "Massive triangle", tri1)
compare("Massless triangle", tri0, "One-mass tri (prop 1)", tri1m0)
compare("Massive triangle", tri1, "One-mass tri (prop 1)", tri1m0)
compare("Massless triangle", tri0, "Massless box", box0)
compare("Massless box", box0, "Massive sunrise", sun)
compare("Massive sunrise", sun, "Massive triangle", tri1)
compare("One-mass tri (prop 1)", tri1m0, "One-mass tri (prop 2)", tri1m1)

print("\n" + "-" * W)
print(f"""
  Note: the planar double box (A: {dbox.gkz.a_matrix.shape}) and
  tetrahedron/K4 (A: {tet.gkz.a_matrix.shape}) have Newton polytopes
  with {len(dbox.newton_polytope.points)} and {len(tet.newton_polytope.points)} monomials
  respectively, too large for the brute-force affine equivalence backend.
  Use specialised software for those comparisons.
""")

print("=" * W)
print("  ANALYSIS COMPLETE")
print("=" * W)
