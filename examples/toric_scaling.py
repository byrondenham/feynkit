"""
Toric ideal scaling survey.

Runs massless and massive variants of increasing complexity, printing
each result as it completes.
"""

import time

import sympy as sp

from feynkit import Edge, FeynmanIntegral, Graph
from feynkit.algebra import compute_toric_ideal_generators

# -- helpers ------------------------------------------------------------------


def row(label, fi):
    A = fi.gkz.a_matrix
    t0 = time.perf_counter()
    gens = compute_toric_ideal_generators(A)
    t = time.perf_counter() - t0
    r, c = A.shape
    print(f"  {label:<44} {r} x {c:<5}  {len(gens):>5}  {t:.3f}s", flush=True)


def polygon(n, n_masses=0):
    nus = sp.symbols(f"nu1:{n + 1}", positive=True)
    ms = sp.symbols(f"m1:{n + 1}", nonnegative=True) if n_masses else []

    def mass(i):
        return ms[i] if i < n_masses else sp.Integer(0)

    if n == 2:
        edges = [
            Edge(idx=1, v1=1, v2=2, is_internal=True, mass=mass(0), nu=nus[0]),
            Edge(idx=2, v1=1, v2=2, is_internal=True, mass=mass(1), nu=nus[1]),
            Edge(idx=3, v1=1, v2=3, is_internal=False),
            Edge(idx=4, v1=2, v2=4, is_internal=False),
        ]
        g = Graph(internal_vertices=2, external_legs=2, edges=edges)
    else:
        internal = [
            Edge(
                idx=i + 1,
                v1=(i % n) + 1,
                v2=((i + 1) % n) + 1,
                is_internal=True,
                mass=mass(i),
                nu=nus[i],
            )
            for i in range(n)
        ]
        external = [
            Edge(idx=n + 1 + i, v1=i + 1, v2=n + 1 + i, is_internal=False) for i in range(n)
        ]
        g = Graph(internal_vertices=n, external_legs=n, edges=internal + external)
    return FeynmanIntegral(g, propagator_exponents={i + 1: nus[i] for i in range(n)})


def banana(n_props):
    """(n_props - 1)-loop banana: n_props parallel edges between 2 vertices."""
    ms = sp.symbols(f"m1:{n_props + 1}", nonnegative=True)
    nus = sp.symbols(f"nu1:{n_props + 1}", positive=True)
    edges = [
        Edge(idx=i + 1, v1=1, v2=2, is_internal=True, mass=ms[i], nu=nus[i]) for i in range(n_props)
    ] + [
        Edge(idx=n_props + 1, v1=1, v2=3, is_internal=False),
        Edge(idx=n_props + 2, v1=2, v2=4, is_internal=False),
    ]
    g = Graph(internal_vertices=2, external_legs=2, edges=edges)
    return FeynmanIntegral(g, propagator_exponents={i + 1: nus[i] for i in range(n_props)})


# -- header --------------------------------------------------------------------

print()
print(f"  {'diagram':<44} {'A':>5}       {'#gens':>5}   time")
print("  " + "-" * 65)

# -- series 1: massless polygons -----------------------------------------------

print("\n  -- massless polygons --------------------------------------")
poly_names = {
    2: "bubble",
    3: "triangle",
    4: "box",
    5: "pentagon",
    6: "hexagon",
    7: "heptagon",
    8: "octagon",
}
for n in range(2, 9):
    row(f"massless {poly_names[n]}  ({n} props)", polygon(n, 0))

# -- series 2: triangle + masses -----------------------------------------------

print("\n  -- triangle: adding masses --------------------------------")
for k in range(4):
    row(f"triangle, {k} massive prop(s)", polygon(3, k))

# -- series 3: box + masses ----------------------------------------------------

print("\n  -- box: adding masses -------------------------------------")
for k in range(5):
    row(f"box, {k} massive prop(s)", polygon(4, k))

# -- series 4: pentagon + masses -----------------------------------------------

print("\n  -- pentagon: adding masses --------------------------------")
for k in range(6):
    row(f"pentagon, {k} massive prop(s)", polygon(5, k))

# -- series 5: hexagon + masses ------------------------------------------------

print("\n  -- hexagon: adding masses ---------------------------------")
for k in range(7):
    row(f"hexagon, {k} massive prop(s)", polygon(6, k))

# -- series 6: banana (increasing loop order) ---------------------------------

print("\n  -- banana / sunrise (increasing loop order) ---------------")
for n in range(2, 8):
    row(f"banana, {n} props  ({n - 1}-loop)", banana(n))
