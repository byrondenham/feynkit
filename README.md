# Feynkit

**Symbolic Feynman integral computations in Python**

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Feynkit computes every standard representation of a Feynman integral from a single graph
description: Symanzik polynomials, Schwinger / Feynman / Lee-Pomeransky parametrisations, the
GKZ hypergeometric system, the Newton polytope, and the toric ideal. It assigns
a canonical **Nickel / CNickel index** to every graph and can construct diagrams directly from
that index. It can compare integrals by unimodular, affine, or point-configuration equivalence,
compute the principal A-determinant whose factors are the candidate Landau singularities, and cache results in a
persistent SQLite database.

---

## Installation

```bash
git clone https://github.com/byrondenham/feynkit.git
cd feynkit
pip install -e .
```

With [uv](https://github.com/astral-sh/uv):

```bash
uv sync
uv run python examples/complete_analysis.py
```

### Optional: 4ti2 (recommended for toric ideals)

The built-in SymPy backend for toric ideal computation is exact but slow above box level.
Install 4ti2 to get ~15 x  faster results:

```bash
sudo pacman -S 4ti2      # Arch Linux
brew install 4ti2         # macOS
sudo apt install 4ti2     # Ubuntu / Debian
```

Feynkit detects 4ti2 automatically (`backend="auto"`, the default).

---

## CLI: `fk`

After installation, the `fk` command analyses any diagram given by its CNickel string. Quote the
string: an unquoted `|` is a shell pipe.

```bash
fk analyse "12e|2e|e|:zzz"                     # every section of the massless triangle
fk analyse "12e|2e|e|:zzz" -g -n               # GKZ system and Newton polytope only
fk analyse "12e|2e|e|"                         # bare topology: every propagator massless
fk analyse "12e|2e|e|:nzz" --latex triangle.tex --json --no-db
fk compare "12e|2e|e|:nzz" "12e|2e|e|:znz"     # the mass on two different propagators
```

`fk analyse` prints the Symanzik polynomials, parametrisations, GKZ system, toric ideal, Newton
polytope and symmetries, or those its section flags choose. It also writes the analysis report
with `--latex FILE` and `--text FILE`, and prints a JSON summary with `--json`. `fk compare` tests
four equivalences between the A-configurations of two diagrams: unimodular, affine,
point-configuration and finite-index. It prints each map it finds, and for a point-configuration
or finite-index map, which sends every column of one A-matrix to a column of the other, the
identity between the two integrals. Results are stored in `feynkit.db` in the working directory;
use `--db PATH` for another file or `--no-db` for none. The bare forms `fk CNICKEL` and `fk A B`
still work. See [the guide](docs/guide.md#cli-fk) for every option and the exit status.

---

## Quick start (Python API)

```python
import sympy as sp
from feynkit import Edge, Graph, FeynmanIntegral

# 1-loop massless triangle, explicit edge construction
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
fi    = FeynmanIntegral(graph, propagator_exponents={i+1: nu[i] for i in range(3)})

# Canonical graph identifier (Nickel / CNickel)
print(fi.nickel_index)    # "12e|2e|e|"
print(fi.cnickel)         # "12e|2e|e|:zzz"

# Symanzik polynomials
sym = fi.symanzik
print("U =", sym.u)          # a_1 + a_2 + a_3  (degree L=1 for 1-loop)
print("G =", sym.g)          # Lee-Pomeransky G = U + F

# GKZ A-matrix
r, m = fi.gkz.a_matrix.shape
print(f"A is {r} x {m}")       # A is 4 x 6 for the triangle

# Schwinger-representation (Cayley) GKZ system and its F-block reduction
cay = fi.schwinger_gkz
print(cay.a_matrix.shape, cay.beta_parameters)
print(cay.restrict_to_f_block().a_matrix)

# Newton polytope
pts = fi.newton_polytope.points
print(f"G has {len(pts)} monomials")

# Toric ideal of the A-matrix
ti = fi.toric_ideal
print(f"{len(ti.generators)} generator(s)")
for gen in ti.generators:
    print(" ", gen)

# Derive a related integral with unit dimension
fi4 = fi.with_(dimension=sp.Integer(4))
```

Or skip the boilerplate entirely using CNickel:

```python
fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")   # massless triangle
fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")      # massive sunrise
```

`fi.to_latex()` writes these results up as an analysis report, a LaTeX article that also covers
the convergence region, the symmetries and the Landau surfaces, states its conventions, cites a
source for each claim and says what feynkit does not compute. It compiles with pdflatex.
`fi.to_text()` gives the same report as plain text, and
[`docs/triangle_analysis.pdf`](docs/triangle_analysis.pdf) is the report of the massive triangle.

```python
from pathlib import Path
fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")   # massive triangle
Path("triangle.tex").write_text(fi.to_latex())
```

---

## Nickel and CNickel index

Every graph has a canonical **Nickel index**, a string that uniquely identifies its topology
up to vertex relabelling. The **CNickel** (Colored Nickel) index extends this with a mass
colouring (`z` = massless, `n` = massive) for each propagator, so the string fully specifies
the integral family.

```python
# Diagrams can be constructed directly from a CNickel string
fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")  # one-mass triangle
fi.cnickel   # -> "12e|2e|e|:nzz"  (round-trips for canonical input)

fi = FeynmanIntegral.from_nickel("111e|e|")          # massless 3-prop banana
fi.cnickel   # -> "111e|e|:zzz"

# Read the index from any FeynmanIntegral or Graph
print(fi.nickel_index)   # topology only
print(fi.cnickel)        # topology + mass colouring

print(graph.nickel_index())
print(graph.cnickel())
```

Common examples:

| Diagram | CNickel |
|---------|---------|
| Massless bubble | `11e\|e\|:zz` |
| Massless triangle | `12e\|2e\|e\|:zzz` |
| One-mass triangle | `12e\|2e\|e\|:nzz` |
| Massless box | `12e\|3e\|3e\|e\|:zzzz` |
| Massless 3-prop banana | `111e\|e\|:zzz` |

All integrals related by graph automorphisms (e.g. all three one-mass triangles) share the
same CNickel string, the canonical form minimises the mass colouring lexicographically.

---

## Features

| Module | What it computes |
|--------|-----------------|
| `feynkit.core` | Graph data structures, Nickel/CNickel index, `from_cnickel` constructor |
| `feynkit.polynomials` | Symanzik U and F via spanning-tree enumeration |
| `feynkit.parametrisations` | Schwinger, Feynman, Lee-Pomeransky representations |
| `feynkit.systems` | GKZ A-matrix, beta parameters, Euler operators; Schwinger-representation Cayley system |
| `feynkit.algebra` | Toric ideal generators (SymPy or 4ti2 backend), Gröbner bases, ideal quotients and intersections, syzygies |
| `feynkit.normal_forms` | Unimodular equivalence (Liu-Cai), affine equivalence, polytope automorphism groups |
| `feynkit.polytope` | Face lattice, facet inequalities and normalised volume of a lattice polytope (`polytope_data`) |
| `feynkit.a_configuration` | Arbitrary GKZ A-configurations: equivalence, finite-index maps, Smith invariants, symmetry pairs |
| `feynkit.landau` | Principal A-determinant over all polytope faces; one-loop closed form |
| `feynkit.artifacts` | Conformal simplex, BMS simplex, complete-graph, and massless-polygon A-configurations |
| `feynkit.database` | SQLite cache for GKZ analysis results, CNickel, and automorphism data |
| `feynkit.visualisation` | TikZ diagrams and Newton polytope plots |
| `feynkit.io` | The analysis report (`AnalysisReport`) and its LaTeX and plain-text renderers |
| `feynkit.cli` | `fk` command-line tool |

All representations are computed lazily and cached on the `FeynmanIntegral` object.

---

## Polytope equivalence

Two integrals whose monomial supports are unimodularly equivalent, every point and not only the
hull vertices, have isomorphic GKZ systems and belong to the same analytic family.
`is_unimodular_equivalent_to()` compares the hull vertices, which settles it when every monomial
is a vertex, as for the one-mass triangles:

```python
fi_a = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")   # one-mass triangle, edge 0-1 massive
fi_b = FeynmanIntegral.from_cnickel("12e|2e|e|:znz")   # one-mass triangle, edge 0-2 massive

result = fi_a.is_unimodular_equivalent_to(fi_b)
print(result.equivalent)     # True
print(result.witness_map)    # GL_n(Z) matrix
```

For broader comparison, `is_affinely_equivalent_to()` tests equivalence over the rationals.

---

## AConfiguration: arbitrary GKZ inputs

`AConfiguration` wraps any integer A-matrix (not just those from Feynman graphs) and exposes
the same equivalence, polytope, and symmetry machinery:

```python
from feynkit import AConfiguration
import sympy as sp

A = sp.Matrix([
    [1, 1, 1, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
])
cfg = AConfiguration(A)

print(cfg.smith_invariants)           # Smith normal-form diagonal
print(cfg.normalized_volume)          # normalised polytope volume
print(cfg.n_points)                   # number of A-columns
print(cfg.newton_polytope_points)     # hull vertices

# Equivalence
cfg2 = AConfiguration(other_A)
res = cfg.is_unimodular_equivalent_to(cfg2)
print(res.equivalent, res.witness_map)

# Symmetry pairs: integer affine self-maps, each with |det M| = 1
from feynkit import symmetry_pairs
pairs = symmetry_pairs(cfg)
```

This is also the entry point for conformal simplex and BMS families:

```python
from feynkit import bms_simplex_a_config, complete_graph_a_config, conformal_companion_a_config

bms3 = bms_simplex_a_config(n=3)      # BMS_3 / massless triangle A-matrix
bms4 = bms_simplex_a_config(n=4)      # BMS_4
k4   = complete_graph_a_config(n=4)   # K_4 complete-graph A-matrix
comp = conformal_companion_a_config(n=4)  # conformal companion for n=4
```

---

## Landau singularities

`feynkit.landau` computes the reduced principal A-determinant of the Lee-Pomeransky polynomial:
the product over every face of the Newton polytope of the discriminant of $G$ restricted to that
face. Its irreducible factors are the candidate Landau surfaces:

```python
from feynkit import FeynmanIntegral, landau_analysis

fi = FeynmanIntegral.from_cnickel("11e|e|:nn")   # massive bubble
la = landau_analysis(fi)

print(la.landau_surfaces)          # m_1, m_2, s, s - (m_1 + m_2)^2, s - (m_1 - m_2)^2
print(la.face_discriminants)       # one FaceDiscriminant per face, all dimensions
```

For one-loop graphs `one_loop_landau_surfaces(fi)` gives the same factors in closed form from the
modified Cayley matrix (Dlapa, Helmer, Papathanasiou, Tellander 2023). Faces of dimension two or
more use a Gröbner elimination; install Singular for speed.

---

## Automorphism groups

The unimodular automorphism group Aut(P) of a Newton polytope encodes the symmetries of the GKZ
system. Each symmetry pair (M, t, P) gives the identity $I_A(\beta, z_P) = I_A(T\beta, z)$ for the
integral without Gamma prefactors; for a coefficient-preserving automorphism $z_P = z$, so it
relates the integral at $\beta$ and at $T\beta$:

```python
from feynkit.normal_forms.polytope_automorphisms import coefficient_preserving_indices

fi = FeynmanIntegral.from_cnickel("111e|e|:zzz")  # massless 3-banana

auts = fi.polytope_automorphisms       # PolytopeAutomorphisms
print(auts.order)                      # 24  (S_4: Newton polytope is a 3-simplex)
print(auts.vertex_orbits)             # [[0,1,2,3]]  (single orbit)

gauts = fi.graph_automorphisms         # vertex permutations of the Feynman graph
print(len(gauts))                      # 2  (Z/2: swap the two vertices)

cp = coefficient_preserving_indices(fi, auts)
print(len(cp))                         # 6  (= 3! edge perms, all monomials have coeff 1)

# Symmetry pairs: integer affine self-maps of the point configuration
pairs = fi.symmetry_pairs
uni  = [p for p in pairs if p.is_unimodular]     # all of them: a self-map has |det| = 1
```

---

## Database

Store and retrieve GKZ analysis results in a persistent SQLite database. Computations are never
repeated for the same Newton polytope fingerprint. The CNickel index is stored automatically:

```python
from feynkit import FeynkitDatabase

with FeynkitDatabase("survey.db") as db:
    rec = db.store(fi, label="massless triangle", compute_automorphisms=True)
    print(rec.n_toric_gens, "generators")
    print(rec.cnickel)            # "12e|2e|e|:zzz"
    print(rec.poly_aut_order)     # |Aut(P)|
    print(rec.graph_aut_order)    # |Aut(Gamma)|
    print(rec.coeff_pres_order)   # coefficient-preserving subgroup order

    # Find all unimodularly equivalent integrals in the database
    matches = db.find_equivalent(fi, relation="unimodular")
```

The `examples/toric_survey.py` script runs a 56-diagram survey across massless and massive
polygon and banana families and is safe to interrupt and resume.

---

## Examples

| Script | Description |
|--------|-------------|
| `examples/complete_analysis.py` | Full analysis of 8 diagrams: triangle, box, sunrise, double box, tetrahedron |
| `examples/toric_scaling.py` | Timing survey: massless polygons L=2...8 plus masses |
| `examples/toric_survey.py` | Resumable 56-diagram survey with SQLite storage |
| `examples/equivalence_survey.py` | Equivalence-rich survey: all C(n,k) mass placements |
| `examples/automorphism_survey.py` | Automorphism group survey: polytope, graph, and coefficient-preserving groups |
| `examples/bms_g_polynomial_analysis.py` | BMS_n G-polynomial analysis and conformal families |
| `examples/higher_analogues_search.py` | Search for higher-dimensional BMS/triangle analogues |
| `examples/landau_analysis.py` | Landau singularity analysis of standard diagrams |
| `examples/bubble_example.py` | Minimal working example (massless bubble) |
| `examples/gkz_example.py` | GKZ system and Newton polytope in detail |
| `examples/parametrisations_example.py` | All three parametrisations side by side |
| `examples/affine_equivalence_example.py` | Polytope equivalence examples |

Run any example with:

```bash
uv run python examples/complete_analysis.py
uv run python examples/toric_survey.py survey.db
```

---

## Documentation

See [`docs/guide.md`](docs/guide.md) for the full user guide, including:

- Complete `fk` CLI reference
- Detailed API reference with parameter tables
- Nickel/CNickel index format and construction from strings
- `AConfiguration` for arbitrary GKZ systems
- Landau singularity analysis
- Conformal and BMS artifact factories
- Runnable examples for every feature
- Standard diagram library (bubble, triangle, box, n-gon, banana, double box, tetrahedron)
- Database patterns for large-scale surveys

---

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=feynkit --cov-report=html
```

---

## References

- Weinzierl, S. (2022). *Feynman Integrals.* Springer. [arXiv:2201.03593](https://arxiv.org/abs/2201.03593)
- de la Cruz, L. (2019). Feynman integrals as A-hypergeometric functions. *JHEP* **12**, 123.
- Lee, R.N. and Pomeransky, A.A. (2013). Critical points and number of master integrals. *JHEP* **11**, 165.
- Gelfand, Kapranov, Zelevinsky (1994). *Discriminants, Resultants and Multidimensional Determinants.*
- Bogner, C. et al. (2017). Loopedia, a database for loop integrals. [arXiv:1709.01266](https://arxiv.org/abs/1709.01266)
- Liu, Y. and Cai, B. (2025). Unimodular isomorphism of lattice polytopes. [arXiv:2506.23846](https://arxiv.org/abs/2506.23846)

---

## License

MIT, see [LICENSE](LICENSE).
