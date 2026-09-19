# Feynkit User Guide

Feynkit is a Python library for symbolic computation of Feynman integrals. Starting from a graph
description it computes Symanzik polynomials, parametric representations, the GKZ hypergeometric
system, the Newton polytope, and the toric ideal (IBP generators). It can compare polytopes by
unimodular, affine, or point-configuration equivalence, detect Landau singularities, and provides
a `fk` command-line tool for instant analysis of any diagram.

---

## Table of Contents

1. [Installation](#installation)
2. [CLI: fk](#cli-fk)
3. [Core concepts](#core-concepts)
4. [Building a Feynman diagram](#building-a-feynman-diagram)
5. [Nickel and CNickel index](#nickel-and-cnickel-index)
6. [The FeynmanIntegral facade](#the-feynmanintegral-facade)
7. [Symanzik polynomials](#symanzik-polynomials)
8. [Parametric representations](#parametric-representations)
9. [GKZ system](#gkz-system)
10. [Newton polytope](#newton-polytope)
11. [Toric ideal and IBP relations](#toric-ideal-and-ibp-relations)
12. [Polytope equivalence](#polytope-equivalence)
13. [Automorphism groups and symmetry pairs](#automorphism-groups-and-symmetry-pairs)
14. [Deriving modified integrals](#deriving-modified-integrals)
15. [AConfiguration: arbitrary GKZ inputs](#aconfiguration-arbitrary-gkz-inputs)
16. [Landau singularities](#landau-singularities)
17. [Conformal and BMS artifact factories](#conformal-and-bms-artifact-factories)
18. [The database](#the-database)
19. [Visualisation and export](#visualisation-and-export)
20. [Standard diagram library](#standard-diagram-library)

---

## Installation

### From source

```bash
git clone https://github.com/byrondenham/feynkit.git
cd feynkit
pip install -e .
# or with uv:
uv sync
```

### Optional: 4ti2 backend for toric ideals

The default SymPy backend for computing toric ideals is exact but slow at pentagon scale and
beyond. The 4ti2 backend is ~15 x  faster and handles large problems that would otherwise not
terminate.

```bash
# Arch Linux
sudo pacman -S 4ti2

# macOS
brew install 4ti2

# Ubuntu / Debian
sudo apt install 4ti2
```

Feynkit auto-detects 4ti2 at runtime (`backend="auto"`, the default). You can also force a
specific backend:

```python
from feynkit.algebra import compute_toric_ideal_generators
gens = compute_toric_ideal_generators(a_matrix, backend="4ti2")
gens = compute_toric_ideal_generators(a_matrix, backend="sympy")
```

---

## CLI: fk

After `pip install -e .` or `uv sync`, the `fk` command is available on the PATH.

### Single-diagram analysis

```
fk <cnickel> [section flags] [--db PATH]
```

With no section flags, `fk` runs all sections in sequence. Pass one or more flags to restrict
output.

```bash
fk "12e|2e|e|:zzz"                  # full analysis of the massless triangle
fk "12e|2e|e|:zzz" -g -n            # GKZ and Newton polytope only
fk "111e|e|:zzz"                     # massless 3-propagator banana
fk 12e|2e|e|                         # bare Nickel topology (massless assumed)
fk "12e|2e|e|:nzz" -S               # symmetries only for one-mass triangle
```

Section flags:

| Flag | Long form | Section |
|------|-----------|---------|
| `-s` | `--symanzik` | Symanzik polynomials U, F, G |
| `-p` | `--params` | Parametrisations (Schwinger, Feynman, Lee-Pom.) |
| `-g` | `--gkz` | GKZ A-matrix and Euler equations |
| `-t` | `--toric` | Toric ideal (IBP generators in z-space) |
| `-n` | `--newton` | Newton polytope (vertices, volume, Smith invariants) |
| `-S` | `--symmetries` | Polytope automorphisms and symmetry pairs |

The graph header (CNickel, loop count, propagators, external legs) and database record are
always printed regardless of section flags.

#### Database option

Results are written to `feynkit.db` by default. Override with `--db`:

```bash
fk "12e|2e|e|:zzz" --db my_survey.db
```

### Pairwise equivalence

Pass two CNickel strings to run an equivalence analysis:

```
fk <cnickel1> <cnickel2> [--db PATH]
```

```bash
fk "12e|2e|e|:zzz" "11e|e|:zz"          # triangle vs bubble
fk "12e|2e|e|:nzz" "12e|2e|e|:znz"      # one-mass triangles: different mass placements
```

The pairwise mode:
1. Prints a brief per-diagram summary (Symanzik polynomials, A-matrix, Smith invariants).
2. Runs unimodular, affine-polytope, point-configuration, and finite-index equivalence checks.
3. For any found map, prints the explicit change-of-variables and the induced GKZ parameter
   transformation `$\beta$ -> T*$\beta$`.

### Example output (single diagram)

```
====================================================================
  Feynman integral  12e|2e|e|:zzz
====================================================================
  Nickel index                 12e|2e|e|
  Loop count                   1
  Propagators                  3
  External legs                3

--------------------------------------------------------------------
  Symanzik polynomials
--------------------------------------------------------------------
  U  =  a_1 + a_2 + a_3
  F  =  ...
  G  =  U + F  =  ...

--------------------------------------------------------------------
  GKZ hypergeometric system
--------------------------------------------------------------------
  A-matrix  (4 x 6)  [rows = coordinates; cols = monomials of G]
    [ 1  1  1  1  1  1 ]
    [ 1  1  1  0  0  0 ]
    [ 1  0  0  1  1  0 ]
    [ 0  1  0  1  0  1 ]
  ...
```

---

## Core concepts

Feynkit works with Feynman integrals in the **Lee-Pomeransky representation**:

```
I ~ int [du] G(u)^(d/2-E/2) prod u_i^(nu_i-1)
```

where `G = U + F` is the Lee-Pomeransky polynomial, `U` is the first Symanzik polynomial
(spanning trees), `F` is the second (spanning 2-forests weighted by momenta), `E` is the total
propagator degree, and `d` is the spacetime dimension.

The monomial support of `G` defines the **Newton polytope**. Its column-homogenised form is the
**GKZ A-matrix**. The kernel of the monomial map `z -> u^A` is the **toric ideal**, whose
generators are the IBP (integration-by-parts) relations.

---

## Building a Feynman diagram

### Edge

Every edge, both internal propagators and external legs, is an `Edge` object:

```python
from feynkit import Edge
import sympy as sp

m = sp.Symbol("m", nonnegative=True)
nu = sp.Symbol("nu", positive=True)

# Internal propagator with mass m and exponent nu
e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m, nu=nu)

# Massless internal propagator (default mass=0, nu=1)
e2 = Edge(idx=2, v1=2, v2=3, is_internal=True)

# External leg (is_internal=False; mass and nu are ignored)
ex = Edge(idx=3, v1=1, v2=4, is_internal=False)
```

Parameters:

| Name | Type | Description |
|------|------|-------------|
| `idx` | `int` | Unique positive integer identifier |
| `v1`, `v2` | `int` | Vertex indices at each end |
| `is_internal` | `bool` | `True` for propagators, `False` for external legs |
| `mass` | `sp.Expr` | Propagator mass (default `0`) |
| `nu` | `sp.Expr` | Propagator exponent (default `1`) |

Vertex index conventions:
- Internal vertices are labelled `1, 2, ..., V`.
- External vertices are `V+1, V+2, ...`, each attached to exactly one external leg.

### Graph

```python
from feynkit import Edge, Graph
import sympy as sp

# 1-loop triangle: 3 internal vertices, 3 external legs
nu = sp.symbols("nu1:4", positive=True)
z  = sp.Integer(0)

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z, nu=nu[0]),
    Edge(idx=2, v1=2, v2=3, is_internal=True, mass=z, nu=nu[1]),
    Edge(idx=3, v1=3, v2=1, is_internal=True, mass=z, nu=nu[2]),
    Edge(idx=4, v1=1, v2=4, is_internal=False),
    Edge(idx=5, v1=2, v2=5, is_internal=False),
    Edge(idx=6, v1=3, v2=6, is_internal=False),
]
graph = Graph(internal_vertices=3, external_legs=3, edges=edges)
```

`Graph` validates the edge list at construction time and raises `ValidationError` if:
- An `idx` is duplicated
- An internal edge references a non-existent vertex
- An external leg attaches to more than one internal vertex

Useful methods:

```python
graph.get_internal_edges()   # sorted list of internal Edge objects
graph.get_external_edges()   # sorted list of external Edge objects
graph.get_loop_count()       # E_int - V_int + 1 (for connected graphs)
graph.nickel_index()         # canonical topology string, e.g. "12e|2e|e|"
graph.cnickel()              # topology + mass colouring,  e.g. "12e|2e|e|:zzz"
```

Auto-generated symbols:

```python
graph.schwinger_parameters   # {edge_idx: a_idx} for internal edges
graph.external_parameters    # {leg_num: b_j}   for external legs
```

---

## Nickel and CNickel index

### Format

Every Feynman graph (up to vertex relabelling) has a canonical **Nickel index**, a compact
string that uniquely identifies its topology. The **CNickel** (Colored Nickel) index extends
this with a per-propagator mass colouring, making it a complete identifier for an integral
family.

**Nickel format:** internal vertices are numbered `0, 1, ..., V-1`. For each vertex `i` in
order, list its higher-numbered internal neighbors (sorted ascending), then one `e` per
external leg. Entries are separated by `|`. The canonical form is the lexicographically
smallest string over all `V!` vertex labellings.

**CNickel format:** `<nickel>:<colours>`, where `<colours>` is a string of `z` (zero / massless)
and `n` (nonzero / massive) characters, one per internal edge in the order they appear left-
to-right in `<nickel>`. The canonical form jointly minimises topology and mass colouring.

Common examples:

| Diagram | Nickel | CNickel (massless) |
|---------|--------|--------------------|
| Bubble | `11e\|e\|` | `11e\|e\|:zz` |
| Triangle | `12e\|2e\|e\|` | `12e\|2e\|e\|:zzz` |
| Box | `12e\|3e\|3e\|e\|` | `12e\|3e\|3e\|e\|:zzzz` |
| 3-prop banana | `111e\|e\|` | `111e\|e\|:zzz` |

For a one-mass triangle, all three placements of the single massive edge give the same CNickel
(`"12e|2e|e|:nzz"`) because the canonical form absorbs graph automorphisms, the mass colouring
is minimised ('`n`' < '`z`' in ASCII) over all equivalent labellings.

### Constructing a graph from CNickel

Both `Graph` and `FeynmanIntegral` can be built directly from a CNickel string, without writing
out the individual `Edge` objects:

```python
from feynkit import Graph, FeynmanIntegral

# Graph only
g = Graph.from_cnickel("12e|2e|e|:nzz")   # one-mass triangle
g = Graph.from_nickel("12e|2e|e|")         # massless triangle (bare topology)

# Full FeynmanIntegral (default symbolic exponents nu_1, nu_2, ... and dimension D)
fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
fi = FeynmanIntegral.from_nickel("111e|e|")   # massless 3-prop banana

# Override defaults via keyword arguments
import sympy as sp
fi = FeynmanIntegral.from_cnickel(
    "12e|2e|e|:zzz",
    dimension=sp.Integer(4),
)
```

Massive edges (colour `'n'`) receive a unique symbolic mass `m_<idx>` with assumptions
`{nonnegative: True, real: True}`. Massless edges receive `mass = 0`.

Non-canonical input is accepted: the resulting `Graph` will report the canonical CNickel when
`cnickel()` is called.

### Reading the index from an integral

```python
fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")

print(fi.nickel_index)     # "12e|2e|e|"  , topology only
print(fi.cnickel)          # "12e|2e|e|:zzz"

# Same methods on the Graph object directly
g = fi.graph
print(g.nickel_index())
print(g.cnickel())
```

### Round-trip property

For any canonical CNickel string `s`:

```python
Graph.from_cnickel(s).cnickel() == s   # True
```

For non-canonical input the resulting graph's `cnickel()` returns the canonical form. This
makes CNickel suitable as a stable, human-readable identifier for integral families.

---

## The FeynmanIntegral facade

`FeynmanIntegral` is the main entry point. All representations are computed **lazily** and
**cached** on first access.

```python
from feynkit import FeynmanIntegral

fi = FeynmanIntegral(
    graph,
    propagator_exponents={1: nu[0], 2: nu[1], 3: nu[2]},
)
```

Constructor parameters:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `graph` | `Graph` | required | The Feynman graph |
| `dimension` | `sp.Expr` | `Symbol("D")` | Spacetime dimension |
| `propagator_exponents` | `dict[int, Expr]` | `{idx: $\nu$_idx}` | Exponent for each internal edge |
| `loop_count` | `int` | from graph | Override if graph topology is ambiguous |
| `momentum_products` | `dict` | generated | `{(i,j): p_i*p_j}` kinematic variables |
| `use_mandelstam` | `bool` | `True` | Use Mandelstam variables for kinematics |
| `kinematic_constraints` | `list[Expr]` | `[]` | Extra symbolic constraints |
| `database` | `FeynkitDatabase` | `None` | Attach a cache database |

Alternative constructors (no `Edge` boilerplate required):

```python
fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")   # one-mass triangle
fi = FeynmanIntegral.from_nickel("12e|2e|e|")          # massless triangle
```

Read-only properties:

```python
fi.graph                # the Graph object
fi.dimension            # spacetime dimension symbol
fi.loop_count           # number of loops
fi.propagator_exponents # {edge_idx: nu_i}
fi.momentum_products    # {(i,j): p_i*p_j}
fi.nickel_index         # canonical Nickel topology string
fi.cnickel              # canonical CNickel string (topology:mass_colors)
```

Computed lazy properties (expensive on first access, then cached):

```python
fi.symanzik             # SymanzikPolynomials
fi.schwinger            # ParametrisationResult (Schwinger form)
fi.feynman              # ParametrisationResult (Feynman form)
fi.lee_pomeransky       # ParametrisationResult (Lee-Pomeransky form)
fi.gkz                  # GKZSystem
fi.newton_polytope      # NewtonPolytope
fi.toric_ideal          # ToricIdeal
fi.polytope_automorphisms   # PolytopeAutomorphisms
fi.graph_automorphisms      # list[list[int]], vertex permutations
fi.symmetry_pairs           # list[SymmetryPair], all integer affine maps
```

---

## Symanzik polynomials

```python
sym = fi.symanzik
```

`SymanzikPolynomials` attributes:

| Attribute | Description |
|-----------|-------------|
| `sym.u` | First Symanzik polynomial (Schwinger parameters `a_i`) |
| `sym.f` | Second Symanzik polynomial (Schwinger parameters) |
| `sym.u_lp` | U in Lee-Pomeransky parameters `u_i` |
| `sym.f_lp` | F in Lee-Pomeransky parameters |
| `sym.g` | G = U\_lp + F\_lp |
| `sym.schwinger_parameters` | `[a1, a2, ...]` in edge-index order |
| `sym.lp_parameters` | `[u1, u2, ...]` in edge-index order |

### Example: massless triangle

```python
import sympy as sp
from feynkit import Edge, Graph, FeynmanIntegral

nu = sp.symbols("nu1:4", positive=True)
z  = sp.Integer(0)

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z, nu=nu[0]),
    Edge(idx=2, v1=2, v2=3, is_internal=True, mass=z, nu=nu[1]),
    Edge(idx=3, v1=3, v2=1, is_internal=True, mass=z, nu=nu[2]),
    Edge(idx=4, v1=1, v2=4, is_internal=False),
    Edge(idx=5, v1=2, v2=5, is_internal=False),
    Edge(idx=6, v1=3, v2=6, is_internal=False),
]
graph = Graph(internal_vertices=3, external_legs=3, edges=edges)
fi    = FeynmanIntegral(graph, propagator_exponents={1: nu[0], 2: nu[1], 3: nu[2]})

sym = fi.symanzik
print("U =", sym.u)           # a_1 + a_2 + a_3  (degree L=1)
print("F =", sym.f)           # kinematic second Symanzik
print("G =", sym.g)
print("Schwinger params:", sym.schwinger_parameters)
print("LP params:", sym.lp_parameters)
```

### Example: massive propagator

```python
m1, m2 = sp.symbols("m1 m2", nonnegative=True)
nu = sp.symbols("nu1:3", positive=True)

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu[0]),
    Edge(idx=2, v1=1, v2=2, is_internal=True, mass=m2, nu=nu[1]),
    Edge(idx=3, v1=1, v2=3, is_internal=False),
    Edge(idx=4, v1=2, v2=4, is_internal=False),
]
graph  = Graph(internal_vertices=2, external_legs=2, edges=edges)
bubble = FeynmanIntegral(graph, propagator_exponents={1: nu[0], 2: nu[1]})

sym = bubble.symanzik
# U = a1 + a2
# F contains the mass terms m1^2*a1^2 + m2^2*a2^2 + ...
```

---

## Parametric representations

Feynkit provides three parametric representations of the integral prefactor and measure.

```python
sch = fi.schwinger          # Schwinger representation
fey = fi.feynman            # Feynman representation
lp  = fi.lee_pomeransky     # Lee-Pomeransky representation
```

Each is a `ParametrisationResult` with attributes:

| Attribute | Description |
|-----------|-------------|
| `.prefactor` | Symbolic prefactor (Gamma factors, powers of $\pi$, etc.) |
| `.parameters` | Integration variables |
| `.integrand` | The integrand expression |
| `.measure` | Integration measure |

### Example

```python
lp = fi.lee_pomeransky
print("Prefactor:", lp.prefactor)
print("Parameters:", lp.parameters)
```

---

## GKZ system

The GKZ (Gelfand-Kapranov-Zelevinsky) system associates a system of
D-module equations to the Newton polytope. The key object is the A-matrix, whose columns are the
homogenised exponent vectors of the monomials of G.

```python
gkz = fi.gkz
```

`GKZSystem` attributes:

| Attribute | Description |
|-----------|-------------|
| `gkz.a_matrix` | `sp.Matrix`, rows = LP parameters + homogenisation, cols = monomials of G |
| `gkz.beta_parameters` | The $\beta$ parameter vector of the GKZ system |
| `gkz.euler_equations` | Symbolic Euler operators `$\sum$ a_ij z_j d/dz_j - $\beta$_i` |
| `gkz.z_variables` | `[z_1, ..., z_m]`, the monomial variables |

The first row of A is a row of ones (homogenisation). Subsequent rows encode the exponent of each
LP parameter `u_i` in each monomial of G.

### Example: massless triangle

```python
gkz = fi.gkz
r, m = gkz.a_matrix.shape
print(f"A is {r} x {m}")
print("A =")
for row in gkz.a_matrix.tolist():
    print(" ", row)
# A is 4 x 6  (r = 4, m = 6 monomials of G for the triangle)
# A =
#   [1, 1, 1, 1, 1, 1]   <- homogenisation row
#   [1, 1, 1, 0, 0, 0]   <- exponent of u1
#   [1, 0, 0, 1, 1, 0]   <- exponent of u2
#   [0, 1, 0, 1, 0, 1]   <- exponent of u3
```

---

## Newton polytope

```python
np_ = fi.newton_polytope
```

`NewtonPolytope` attributes:

| Attribute | Description |
|-----------|-------------|
| `np_.support` | `list[(exponent_vector, coefficient)]` |
| `np_.a_matrix` | GKZ A-matrix (same as `fi.gkz.a_matrix`) |
| `np_.parameters` | LP parameters |
| `np_.points` | Just the exponent vectors |
| `np_.coefficients` | Just the coefficients |

### Example

```python
pts = fi.newton_polytope.points
print(f"G has {len(pts)} monomials")
for pt in pts:
    print(pt)
```

The polytope dimension equals `loops + 1` (one row per LP parameter plus the homogenisation row,
minus one for the affine span). For a 1-loop triangle this is 1, so the Newton polytope is a
line segment.

---

## Toric ideal and IBP relations

The toric ideal is the ideal of polynomial relations among the monomials of G. Its generators
correspond to IBP (integration-by-parts) identities for the integral family.

```python
ti = fi.toric_ideal
```

`ToricIdeal` attributes:

| Attribute | Description |
|-----------|-------------|
| `ti.generators` | `list[sp.Expr]`, polynomials in `z_1, ..., z_m` |
| `ti.a_matrix` | The GKZ A-matrix |
| `ti.z_variables` | `[z_1, ..., z_m]` |

An empty generator list (`len(ti.generators) == 0`) means the toric ideal is trivial: the
monomial map is injective, which often indicates a master integral with no further reductions.

### Checking if the ideal is binomial

```python
from feynkit.algebra import is_binomial_ideal
print(is_binomial_ideal(ti.generators))   # True if all gens have <= 2 terms
```

### Backend selection

```python
from feynkit.algebra import compute_toric_ideal_generators

# Auto: uses 4ti2 if installed, SymPy otherwise
gens = compute_toric_ideal_generators(fi.gkz.a_matrix)

# Force 4ti2 (faster, required for pentagon and above)
gens = compute_toric_ideal_generators(fi.gkz.a_matrix, backend="4ti2")

# Force SymPy (exact, slow)
gens = compute_toric_ideal_generators(fi.gkz.a_matrix, backend="sympy")
```

Rough timings (with 4ti2):

| Diagram | Propagators | A-matrix | Generators | Time |
|---------|-------------|----------|------------|------|
| Bubble | 2 | 2 x 2 | 0 | <0.01 s |
| Triangle | 3 | 2 x 4 | 1 | <0.01 s |
| Box | 4 | 2 x 6 | 3 | <0.01 s |
| Pentagon | 5 | 2 x 8 | 5 | ~0.06 s |
| Hexagon | 6 | 2 x 10 | 7 | ~0.3 s |
| Sunrise (massive) | 3 | 3 x 8 | 5 | ~0.01 s |
| Double box (massive) | 7 | 3 x 64 | large | ~2 s |

### Complete example: toric ideal of the massless triangle

```python
import sympy as sp
from feynkit import Edge, Graph, FeynmanIntegral
from feynkit.algebra import is_binomial_ideal

nu = sp.symbols("nu1:4", positive=True)
z  = sp.Integer(0)

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z, nu=nu[0]),
    Edge(idx=2, v1=2, v2=3, is_internal=True, mass=z, nu=nu[1]),
    Edge(idx=3, v1=3, v2=1, is_internal=True, mass=z, nu=nu[2]),
    Edge(idx=4, v1=1, v2=4, is_internal=False),
    Edge(idx=5, v1=2, v2=5, is_internal=False),
    Edge(idx=6, v1=3, v2=6, is_internal=False),
]
g  = Graph(internal_vertices=3, external_legs=3, edges=edges)
fi = FeynmanIntegral(g, propagator_exponents={1: nu[0], 2: nu[1], 3: nu[2]})

ti = fi.toric_ideal
print(f"Generators: {len(ti.generators)}")
print(f"Binomial  : {is_binomial_ideal(ti.generators)}")
for i, gen in enumerate(ti.generators):
    print(f"  [{i}] {gen} = 0")
```

### Ideal arithmetic: quotients, intersections and syzygies

The toric ideal is an ordinary polynomial ideal, so the helpers in
`feynkit.algebra` that manipulate ideals apply to it directly. All of them take
the list of generators and the list of ring variables; the toric ideal's
variables are the `z_i` coefficients of G, available as `ti.z_variables`.

```python
from feynkit import FeynmanIntegral
from feynkit.algebra import (
    compute_syzygy_module,
    ideal_quotient,
    intersect_ideals,
    is_in_ideal,
)

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
ti = fi.toric_ideal
gens, zs = ti.generators, ti.z_variables

# Membership: is a product of generators in the ideal?  Pass the same
# monomial order the basis was computed in.
import sympy as sp
basis = list(sp.groebner(gens, *zs, order="grevlex").exprs)
print(is_in_ideal(sp.expand(gens[0] * zs[0]), basis, zs, order="grevlex"))   # True

# Quotient  I : J  = { f : f g in I for every g in J }.
# Quotienting by one of the z variables saturates that direction away.
q = ideal_quotient(gens, [zs[0]], zs)

# Intersection of two ideals.
both = intersect_ideals(gens, [zs[0] * zs[1]], zs)

# Syzygies: every list h satisfies sum(h_i * gens_i) == 0.
for h in compute_syzygy_module(gens, zs):
    assert sp.expand(sum(c * g for c, g in zip(h, gens, strict=True))) == 0
```

`ideal_quotient` and `intersect_ideals` return a reduced grevlex Gröbner
basis, with the zero ideal returned as an empty list and the whole ring as
`[1]`. Both are computed by elimination with an auxiliary variable, so they
are exact but slow for large ideals.

`compute_syzygy_module` returns generators of the first syzygy module, found
by running Buchberger's algorithm while tracking how each new basis element is
expressed in the original generators (Schreyer's construction). The returned
vectors generate all relations among the generators over the polynomial ring;
they are not guaranteed to be a minimal generating set.

---

## Polytope equivalence

Two Feynman integrals have **unimodularly equivalent** Newton polytopes if there is a unimodular
matrix `U in GL_n($\mathbb{Z}$)` and translation `t in $\mathbb{Z}$ ^n` mapping the lattice points of one to the other.
Unimodular equivalence implies the GKZ systems are isomorphic: the two integral families share
the same analytic structure.

**Affine equivalence** is the same question over $\mathbb{Q}$ (not just $\mathbb{Z}$), it is strictly weaker.

### Via the FeynmanIntegral facade

```python
result = fi_a.is_unimodular_equivalent_to(fi_b)

if result.equivalent:
    print("Equivalent!")
    print("Witness U:", result.witness_map)
    print("Vertex map:", result.vertex_correspondence)
else:
    print("Not unimodularly equivalent")
    aff = fi_a.is_affinely_equivalent_to(fi_b)
    print("Affinely equivalent:", aff.equivalent)
```

### PolytopeEquivalence result

| Attribute | Type | Description |
|-----------|------|-------------|
| `.equivalent` | `bool` | Verdict |
| `.relation` | `str` | `"unimodular"` or `"affine"` |
| `.witness_map` | `sp.Matrix or None` | `U in GL_n($\mathbb{Z}$)` for unimodular results |
| `.vertex_correspondence` | `list[int] or None` | `vertex[i]` of source -> `vertex[j]` of target |

### Example: one-mass triangles

```python
from feynkit import FeynmanIntegral

# Three variants: mass on edge 0-1, 0-2, 1-2 respectively
fi0 = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")  # canonical
fi1 = FeynmanIntegral.from_cnickel("12e|2e|e|:znz")
fi2 = FeynmanIntegral.from_cnickel("12e|2e|e|:zzn")

# All three collapse to the same canonical CNickel
print(fi0.cnickel)   # "12e|2e|e|:nzz"
print(fi1.cnickel)   # "12e|2e|e|:nzz"
print(fi2.cnickel)   # "12e|2e|e|:nzz"

result = fi0.is_unimodular_equivalent_to(fi2)
print(result.equivalent)        # True
print(result.witness_map)       # permutation matrix in GL_3(Z)
```

### Calling the equivalence functions directly

```python
from feynkit.normal_forms.affine_equivalence import (
    is_unimodular_equivalent,
    is_affinely_equivalent,
)

pts_a = fi_a.newton_polytope.points
pts_b = fi_b.newton_polytope.points

result = is_unimodular_equivalent(pts_a, pts_b)
result = is_affinely_equivalent(pts_a, pts_b)
```

---

## Automorphism groups and symmetry pairs

Feynkit computes the unimodular automorphism group of the Newton polytope of G,
the graph automorphism group (vertex permutations), the coefficient-preserving
subgroup, and the full set of symmetry pairs (including finite-index maps).

See [`docs/automorphism_groups.md`](automorphism_groups.md) for the mathematical
background, physical interpretation, and full results for standard diagrams.

### Three levels of symmetry

| Object | API | Description |
|--------|-----|-------------|
| `PolytopeAutomorphisms` | `fi.polytope_automorphisms` | Full Aut(P): all (U,t) with U in GL_n($\mathbb{Z}$), \|det U\|=1 |
| `list[list[int]]` | `fi.graph_automorphisms` | Vertex permutations preserving topology and mass colouring |
| `list[int]` | `coefficient_preserving_indices(fi, auts)` | Indices into auts.maps whose (U,t) also preserves G's coefficients |

### Symmetry pairs

`fi.symmetry_pairs` returns all integer affine maps between the point configuration and itself,
including finite-index maps (|det| > 1). Each `SymmetryPair` records:

| Field | Type | Description |
|-------|------|-------------|
| `witness_matrix` | `sp.ImmutableMatrix` | The integer linear map M |
| `translation` | `sp.ImmutableMatrix` | Translation vector t |
| `determinant` | `int` | \|det(M)\|, 1 for unimodular, >1 for finite-index |
| `is_unimodular` | `bool` | Shorthand for `determinant == 1` |
| `column_permutation` | `list[int]` | Induced permutation on A-matrix columns |

```python
pairs = fi.symmetry_pairs
uni  = [p for p in pairs if p.is_unimodular]
fi_p = [p for p in pairs if not p.is_unimodular]
print(f"{len(uni)} unimodular, {len(fi_p)} finite-index")
```

### Quick example

```python
from feynkit import FeynmanIntegral
from feynkit.normal_forms.polytope_automorphisms import coefficient_preserving_indices

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")  # massless triangle

auts = fi.polytope_automorphisms
print(auts.order)              # 48  (hyperoctahedral B_3)
print(auts.vertex_orbits)     # [[0,1,2,3,4,5]], single orbit

gauts = fi.graph_automorphisms
print(len(gauts))              # 6  (S_3 on three vertices)

cp = coefficient_preserving_indices(fi, auts)
print(len(cp))                 # 1  (identity only, generic Mandelstam coefficients)

# Massless banana: all monomials have equal coefficients -> full group preserved
banana = FeynmanIntegral.from_cnickel("111e|e|:zzz")
auts_b = banana.polytope_automorphisms
print(auts_b.order)            # 24  (S_4)
cp_b = coefficient_preserving_indices(banana, auts_b)
print(len(cp_b))               # 6  (= 3! edge permutations)
```

### PolytopeAutomorphisms fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `maps` | `list[tuple[ImmutableMatrix, ImmutableMatrix]]` | All (U, t) pairs |
| `order` | `int` | Group order \|Aut(P)\| |
| `vertex_permutations` | `list[list[int]]` | Induced permutation on hull vertices per automorphism |
| `vertex_orbits` | `list[list[int]]` | Partition of vertices into equivalence classes |

---

## Deriving modified integrals

`FeynmanIntegral` is immutable. Use `.with_()` to create a new instance with selected fields
overridden and a completely fresh cache:

```python
# Different spacetime dimension
fi4 = fi.with_(dimension=sp.Integer(4))

# Different propagator exponents (e.g. unit exponents)
fi_unit = fi.with_(propagator_exponents={1: sp.Integer(1), 2: sp.Integer(1), 3: sp.Integer(1)})

# Replace the graph entirely
fi_new = fi.with_(graph=other_graph)
```

Any keyword argument accepted by the constructor can be passed to `.with_()`. Unchanged
arguments are copied from the original object.

---

## AConfiguration: arbitrary GKZ inputs

`AConfiguration` wraps any homogenised integer A-matrix (not just one derived from a Feynman
graph) and exposes the full equivalence, polytope, and symmetry machinery. This is the
entry point for studying GKZ systems that do not come from Feynman diagrams, such as conformal
simplex families or custom point configurations.

```python
from feynkit import AConfiguration
import sympy as sp

A = sp.Matrix([
    [1, 1, 1, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
])
cfg = AConfiguration(A, is_homogenized=True)
```

`AConfiguration` attributes:

| Attribute | Description |
|-----------|-------------|
| `cfg.n_points` | Number of A-columns (monomials) |
| `cfg.ambient_dim` | Ambient dimension (number of rows of A) |
| `cfg.affine_dim` | Dimension of the affine span of the Newton polytope |
| `cfg.smith_invariants` | Diagonal of the Smith normal form |
| `cfg.normalized_volume` | Normalised volume of the Newton polytope (= holonomic rank) |
| `cfg.newton_polytope_points` | Hull vertex coordinates |
| `cfg.intrinsic_model()` | `IntrinsicModel`, the configuration in a minimal lattice basis |

### Equivalence

```python
cfg2 = AConfiguration(other_A, is_homogenized=True)

res = cfg.is_unimodular_equivalent_to(cfg2)
print(res.equivalent, res.witness_map)

res = cfg.is_affinely_equivalent_to(cfg2)
res = cfg.is_point_config_equivalent_to(cfg2)

from feynkit import finite_index_map, FiniteIndexResult
fi_res: FiniteIndexResult = finite_index_map(cfg, cfg2)
print(fi_res.found, fi_res.determinant)
```

### Symmetry pairs

```python
from feynkit import symmetry_pairs, AConfiguration

pairs = symmetry_pairs(cfg)
uni   = [p for p in pairs if p.is_unimodular]
fi_p  = [p for p in pairs if not p.is_unimodular]
```

### Intrinsic lattice model

```python
model = cfg.intrinsic_model()
print(model.base_point)       # reference point in the original lattice
print(model.basis)            # columns span the lattice of the configuration
print(model.intrinsic_points) # points re-expressed in the intrinsic basis
```

---

## Landau singularities

`feynkit.landau` computes the edge-part of the principal A-determinant,

```
E_A^(1)(G) = prod_{tau edge of New(G)}  Delta_{A_tau}(G|_tau)
```

whose zero locus in kinematic space gives the leading Landau singularity surfaces (normal
thresholds and IR singularities).

### From a FeynmanIntegral

```python
from feynkit import FeynmanIntegral, landau_analysis

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")   # massless triangle
la = landau_analysis(fi)

print(la.edge_discriminants)   # list of EdgeDiscriminant, one per Newton-polytope edge
print(la.principal_a_det)      # E_A^(1) as a SymPy expression
```

### From a polynomial directly

```python
from feynkit import landau_analysis_from_polynomial
import sympy as sp

u1, u2, u3 = sp.symbols("u1:4")
G = u1*u2 + u2*u3 + u1*u3   # G-polynomial of the massless triangle
la = landau_analysis_from_polynomial(G, [u1, u2, u3])
```

### LandauAnalysis fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `la.edge_discriminants` | `list[EdgeDiscriminant]` | One per Newton-polytope edge |
| `la.principal_a_det` | `sp.Expr` | Product E_A^(1) |

### EdgeDiscriminant fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `ed.edge_exponents` | `tuple[tuple[int,...],...]` | Exponent vectors on the edge |
| `ed.edge_coefficients` | `tuple[sp.Expr,...]` | Kinematic coefficients |
| `ed.discriminant` | `sp.Expr` | Discriminant of the edge restriction |

### Physical interpretation

The zero locus of each edge discriminant corresponds to:
- **Massive bubble edges**: normal threshold `p^2 = ($\sum$ m_i)^2`
- **Massless edges at a vertex**: IR/collinear singularity (`p_i^2 = 0` for n-gon polygons)
- **BMS_n higher polygons**: `p_i^2 = 0` for each external leg, a novel family of IR
  singularities identified by feynkit

---

## Conformal and BMS artifact factories

The `feynkit.artifacts` module provides ready-made A-configurations for families studied in
conformal field theory and the BMS/celestial amplitude literature:

```python
from feynkit import (
    massless_polygon_a_config,
    bms_simplex_a_config,
    complete_graph_a_config,
    conformal_companion_a_config,
)

# Massless n-gon polygon (1-loop, n external legs)
tri_cfg = massless_polygon_a_config(n=3)   # triangle
box_cfg = massless_polygon_a_config(n=4)   # box

# BMS_n simplex A-configuration (Smith invariants [1,...,1,n-2])
bms3 = bms_simplex_a_config(n=3)    # BMS_3: equivalent to massless triangle
bms4 = bms_simplex_a_config(n=4)    # BMS_4

# Complete graph K_n A-configuration
k4 = complete_graph_a_config(n=4)   # K_4 (tetrahedron)
k5 = complete_graph_a_config(n=5)

# Conformal companion for dimension n (breaks S_n -> S_{n-1} symmetry)
comp4 = conformal_companion_a_config(n=4)
```

All factory functions return an `AConfiguration` instance and can be compared via the standard
equivalence API:

```python
tri = massless_polygon_a_config(n=3)
bms = bms_simplex_a_config(n=3)
res = tri.is_unimodular_equivalent_to(bms)
print(res.equivalent)   # True, BMS_3 is unimodularly equivalent to the triangle
```

The `det = 2` map between conformal families exists only for `n = 3`. For `n >= 4` no such map
exists; `finite_index_map` confirms this by returning `found=False`.

---

## The database

`FeynkitDatabase` is a SQLite-backed store for GKZ analysis results. It lets you:

- Cache toric ideal computations so they are never re-run for the same Newton polytope.
- Store metadata (label, loop count, A-matrix shape) alongside results.
- Retrieve all integrals equivalent to a given one.
- Resume interrupted computations without re-computing completed entries.

### Opening and closing

```python
from feynkit import FeynkitDatabase

# Context manager (preferred)
with FeynkitDatabase("analysis.db") as db:
    ...

# Or manually
db = FeynkitDatabase("analysis.db")
# ... do work ...
db.close()

# In-memory database (for tests or throwaway work)
with FeynkitDatabase(":memory:") as db:
    ...
```

### Storing integrals

`db.store()` computes everything (if not already done), writes to the database, and returns an
`IntegralRecord`:

```python
with FeynkitDatabase("analysis.db") as db:
    rec = db.store(fi, label="massless triangle")
    print(rec.n_toric_gens)      # number of toric generators
    print(rec.is_binomial)       # whether the toric ideal is binomial
    print(rec.n_rows, rec.n_cols)  # A-matrix shape

    # Optionally compute and store automorphism data (slower)
    rec = db.store(fi, label="massless triangle", compute_automorphisms=True)
    print(rec.poly_aut_order)    # |Aut(P)|
    print(rec.graph_aut_order)   # |Aut(Gamma)|
    print(rec.coeff_pres_order)  # coefficient-preserving subgroup order
    print(rec.vertex_orbits)     # list of orbit lists
```

If the same integral (same Newton polytope fingerprint) was stored before, `store()` updates the
existing record and returns it. Automorphism data is stored alongside toric data and displayed in
`db.summary()` when any record has been computed with `compute_automorphisms=True`.

### Looking up a stored result

```python
rec = db.lookup(fi)
if rec is not None:
    print("Already computed:", rec.n_toric_gens, "generators")
else:
    print("Not yet in database")
```

### IntegralRecord fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Primary key |
| `fingerprint` | `str` | SHA-256 of sorted Newton points |
| `a_matrix` | `sp.Matrix` | GKZ A-matrix |
| `newton_points` | `list[tuple]` | Support exponent vectors |
| `n_rows`, `n_cols` | `int` | A-matrix dimensions |
| `loop_count` | `int` | Number of loops |
| `n_props` | `int` | Number of internal propagators |
| `n_ext` | `int` | Number of external legs |
| `n_toric_gens` | `int` | Count of toric generators (0 is distinct from `None`) |
| `toric_generators` | `list[Expr]` | The generator polynomials |
| `is_binomial` | `bool` | Whether all generators are binomials |
| `cnickel` | `str` | Canonical CNickel index |
| `label` | `str` | User-supplied label |
| `poly_aut_order` | `int \| None` | \|Aut(P)\| (set when `compute_automorphisms=True`) |
| `graph_aut_order` | `int \| None` | \|Aut($\Gamma$)\| (vertex permutations) |
| `coeff_pres_order` | `int \| None` | Coefficient-preserving subgroup order |
| `vertex_orbits` | `list[list[int]] \| None` | Vertex orbits under Aut(P) |
| `stored_at` | `str` | ISO timestamp |

Note: `n_toric_gens is None` means the toric ideal has not yet been computed for this record (it
may have been inserted by an earlier partial run). `n_toric_gens == 0` is a valid result (trivial
toric ideal). Automorphism fields are `None` unless `store(..., compute_automorphisms=True)` was
called.

### Finding equivalent integrals

```python
# Find all integrals in the database unimodularly equivalent to fi
matches = db.find_equivalent(fi, relation="unimodular")
for m in matches:
    print(m.label, "is equivalent to", fi)

# Same, but starting from an already-retrieved IntegralRecord
matches = db.find_equivalent_record(rec, relation="unimodular")
```

Equivalence results are cached in the database: each pair is only tested once.

### Resumable survey script pattern

This pattern handles interruptions safely: each integral is committed atomically before the next
one starts.

```python
from feynkit import FeynkitDatabase, FeynmanIntegral
import time

integrals = [
    ("massless triangle",  fi_tri),
    ("massive sunrise",    fi_sun),
    # ...
]

with FeynkitDatabase("survey.db") as db:
    for label, fi in integrals:
        rec = db.lookup(fi)
        if rec is not None and rec.n_toric_gens is not None:
            print(f"[cached]  {label}  ({rec.n_toric_gens} gens)")
            continue

        t0  = time.perf_counter()
        rec = db.store(fi, label=label, compute_automorphisms=True)
        print(
            f"[stored]  {label}  {rec.n_toric_gens} gens"
            f"  |Aut(P)|={rec.poly_aut_order}"
            f"  {time.perf_counter()-t0:.2f}s"
        )
```

### Database summary

```python
print(db.summary())
# Feynkit database  survey.db
# -----------------------------------------------------
# Integrals stored   : 42
# Toric computed     : 42
# ...
```

---

## Visualisation and export

### TikZ diagram

```python
tikz_code = fi.tikz()
print(tikz_code)
```

### Newton polytope

```python
tikz_code = fi.visualise_polytope()
```

### LaTeX analysis document

Produces a self-contained `.tex` file with all polynomials, the GKZ system, parametrisations,
and toric generators formatted for publication:

```python
latex = fi.to_latex()
with open("analysis.tex", "w") as f:
    f.write(latex)
```

### Plain text report

```python
report = fi.to_text()
print(report)
```

---

## Standard diagram library

The following code snippets construct the most common Feynman diagrams. All return a
`FeynmanIntegral` with symbolic propagator exponents.

### Massless bubble (1-loop, 1-point)

```python
import sympy as sp
from feynkit import Edge, Graph, FeynmanIntegral

nu = sp.symbols("nu1:3", positive=True)
z  = sp.Integer(0)

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z, nu=nu[0]),
    Edge(idx=2, v1=1, v2=2, is_internal=True, mass=z, nu=nu[1]),
    Edge(idx=3, v1=1, v2=3, is_internal=False),
    Edge(idx=4, v1=2, v2=4, is_internal=False),
]
g      = Graph(internal_vertices=2, external_legs=2, edges=edges)
bubble = FeynmanIntegral(g, propagator_exponents={1: nu[0], 2: nu[1]})
```

### Massless triangle (1-loop, 3-point)

```python
nu = sp.symbols("nu1:4", positive=True)
z  = sp.Integer(0)

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z, nu=nu[0]),
    Edge(idx=2, v1=2, v2=3, is_internal=True, mass=z, nu=nu[1]),
    Edge(idx=3, v1=3, v2=1, is_internal=True, mass=z, nu=nu[2]),
    Edge(idx=4, v1=1, v2=4, is_internal=False),
    Edge(idx=5, v1=2, v2=5, is_internal=False),
    Edge(idx=6, v1=3, v2=6, is_internal=False),
]
g        = Graph(internal_vertices=3, external_legs=3, edges=edges)
triangle = FeynmanIntegral(g, propagator_exponents={i+1: nu[i] for i in range(3)})
```

### Massive triangle (all propagators massive)

```python
m  = sp.symbols("m1:4", nonnegative=True)
nu = sp.symbols("nu1:4", positive=True)

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m[0], nu=nu[0]),
    Edge(idx=2, v1=2, v2=3, is_internal=True, mass=m[1], nu=nu[1]),
    Edge(idx=3, v1=3, v2=1, is_internal=True, mass=m[2], nu=nu[2]),
    Edge(idx=4, v1=1, v2=4, is_internal=False),
    Edge(idx=5, v1=2, v2=5, is_internal=False),
    Edge(idx=6, v1=3, v2=6, is_internal=False),
]
g                = Graph(internal_vertices=3, external_legs=3, edges=edges)
massive_triangle = FeynmanIntegral(g, propagator_exponents={i+1: nu[i] for i in range(3)})
```

### Massless box (1-loop, 4-point)

```python
nu = sp.symbols("nu1:5", positive=True)
z  = sp.Integer(0)

internal = [
    Edge(idx=i+1, v1=(i % 4)+1, v2=((i+1) % 4)+1,
         is_internal=True, mass=z, nu=nu[i])
    for i in range(4)
]
external = [Edge(idx=5+i, v1=i+1, v2=5+i, is_internal=False) for i in range(4)]
g   = Graph(internal_vertices=4, external_legs=4, edges=internal+external)
box = FeynmanIntegral(g, propagator_exponents={i+1: nu[i] for i in range(4)})
```

### Massless n-gon (1-loop, n-point)

```python
def massless_polygon(n: int) -> FeynmanIntegral:
    nu = sp.symbols(f"nu1:{n+1}", positive=True)
    z  = sp.Integer(0)
    if n == 2:
        edges = [
            Edge(idx=1, v1=1, v2=2, is_internal=True, mass=z, nu=nu[0]),
            Edge(idx=2, v1=1, v2=2, is_internal=True, mass=z, nu=nu[1]),
            Edge(idx=3, v1=1, v2=3, is_internal=False),
            Edge(idx=4, v1=2, v2=4, is_internal=False),
        ]
        g = Graph(internal_vertices=2, external_legs=2, edges=edges)
    else:
        internal = [
            Edge(idx=i+1, v1=(i%n)+1, v2=((i+1)%n)+1,
                 is_internal=True, mass=z, nu=nu[i])
            for i in range(n)
        ]
        external = [
            Edge(idx=n+1+i, v1=i+1, v2=n+1+i, is_internal=False)
            for i in range(n)
        ]
        g = Graph(internal_vertices=n, external_legs=n, edges=internal+external)
    return FeynmanIntegral(g, propagator_exponents={i+1: nu[i] for i in range(n)})
```

### Sunrise / banana (multi-loop, 2-point)

```python
def banana(n_props: int, *, massive: bool = True) -> FeynmanIntegral:
    """(n_props - 1)-loop banana: n_props parallel propagators."""
    nu = sp.symbols(f"nu1:{n_props+1}", positive=True)
    ms = sp.symbols(f"m1:{n_props+1}", nonnegative=True)

    def mass(i):
        return ms[i] if massive else sp.Integer(0)

    edges = [
        Edge(idx=i+1, v1=1, v2=2, is_internal=True, mass=mass(i), nu=nu[i])
        for i in range(n_props)
    ] + [
        Edge(idx=n_props+1, v1=1, v2=3, is_internal=False),
        Edge(idx=n_props+2, v1=2, v2=4, is_internal=False),
    ]
    g = Graph(internal_vertices=2, external_legs=2, edges=edges)
    return FeynmanIntegral(g, propagator_exponents={i+1: nu[i] for i in range(n_props)})

sunrise      = banana(3, massive=True)   # 2-loop, 3 massive propagators
kite         = banana(4, massive=True)   # 3-loop, 4 massive propagators
```

### Planar double box (2-loop, 4-point)

```python
m  = sp.symbols("m1:8", nonnegative=True)
nu = sp.symbols("nu1:8", positive=True)

#  v1 -[1]- v2
#   |         |
#  [2]       [3]
#   |         |
#   v3 -[4]- v4
#   |         |
#  [5]       [6]
#   |         |
#  v5 -[7]- v6

edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m[0], nu=nu[0]),
    Edge(idx=2, v1=1, v2=3, is_internal=True, mass=m[1], nu=nu[1]),
    Edge(idx=3, v1=2, v2=4, is_internal=True, mass=m[2], nu=nu[2]),
    Edge(idx=4, v1=3, v2=4, is_internal=True, mass=m[3], nu=nu[3]),
    Edge(idx=5, v1=3, v2=5, is_internal=True, mass=m[4], nu=nu[4]),
    Edge(idx=6, v1=4, v2=6, is_internal=True, mass=m[5], nu=nu[5]),
    Edge(idx=7, v1=5, v2=6, is_internal=True, mass=m[6], nu=nu[6]),
    Edge(idx=8,  v1=1, v2=7,  is_internal=False),
    Edge(idx=9,  v1=2, v2=8,  is_internal=False),
    Edge(idx=10, v1=5, v2=9,  is_internal=False),
    Edge(idx=11, v1=6, v2=10, is_internal=False),
]
g          = Graph(internal_vertices=6, external_legs=4, edges=edges)
double_box = FeynmanIntegral(g, propagator_exponents={i+1: nu[i] for i in range(7)})
```

### Tetrahedron / K4 (3-loop, 4-point)

```python
m  = sp.symbols("m1:7", nonnegative=True)
nu = sp.symbols("nu1:7", positive=True)

# Complete graph on 4 internal vertices: all C(4,2) = 6 edges
edges = [
    Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m[0], nu=nu[0]),
    Edge(idx=2, v1=1, v2=3, is_internal=True, mass=m[1], nu=nu[1]),
    Edge(idx=3, v1=1, v2=4, is_internal=True, mass=m[2], nu=nu[2]),
    Edge(idx=4, v1=2, v2=3, is_internal=True, mass=m[3], nu=nu[3]),
    Edge(idx=5, v1=2, v2=4, is_internal=True, mass=m[4], nu=nu[4]),
    Edge(idx=6, v1=3, v2=4, is_internal=True, mass=m[5], nu=nu[5]),
    Edge(idx=7,  v1=1, v2=5, is_internal=False),
    Edge(idx=8,  v1=2, v2=6, is_internal=False),
    Edge(idx=9,  v1=3, v2=7, is_internal=False),
    Edge(idx=10, v1=4, v2=8, is_internal=False),
]
g           = Graph(internal_vertices=4, external_legs=4, edges=edges)
tetrahedron = FeynmanIntegral(g, propagator_exponents={i+1: nu[i] for i in range(6)})
```

---

## References

1. Weinzierl, S. (2022). *Feynman Integrals: A Comprehensive Treatment for Students and
   Researchers.* Springer. [arXiv:2201.03593](https://arxiv.org/abs/2201.03593)

2. de la Cruz, L. (2019). Feynman integrals as A-hypergeometric functions. *JHEP* **12**, 123.
   [arXiv:1907.01007](https://arxiv.org/abs/1907.01007)

3. Lee, R.N. and Pomeransky, A.A. (2013). Critical points and master integrals. *JHEP* **11**,
   165. [arXiv:1308.6676](https://arxiv.org/abs/1308.6676)

4. Gelfand, I.M., Kapranov, M.M., Zelevinsky, A.V. (1994). *Discriminants, Resultants and
   Multidimensional Determinants.* Birkhäuser.

5. Sturmfels, B. (1996). *Gröbner Bases and Convex Polytopes.* American Mathematical Society.

6. Bogner, C. et al. (2017). Loopedia, a database for loop integrals.
   [arXiv:1709.01266](https://arxiv.org/abs/1709.01266)

7. Liu, Y. and Cai, B. (2025). Unimodular isomorphism of lattice polytopes.
   [arXiv:2506.23846](https://arxiv.org/abs/2506.23846)
