# Feynkit User Guide

Feynkit is a Python library for symbolic computation of Feynman integrals. Starting from a graph
description it computes Symanzik polynomials, parametric representations, the GKZ hypergeometric
system, the Newton polytope, and the toric ideal. It can compare polytopes by unimodular, affine,
or point-configuration equivalence, detect Landau singularities, write the results up as an
analysis report in LaTeX or plain text, and provides the `fk` command line, which analyses a
diagram given by its CNickel string.

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
10. [Schwinger-representation GKZ system](#schwinger-representation-gkz-system)
11. [Newton polytope](#newton-polytope)
12. [Toric ideal](#toric-ideal)
13. [Polytope equivalence](#polytope-equivalence)
14. [Automorphism groups and symmetry pairs](#automorphism-groups-and-symmetry-pairs)
15. [Deriving modified integrals](#deriving-modified-integrals)
16. [AConfiguration: arbitrary GKZ inputs](#aconfiguration-arbitrary-gkz-inputs)
17. [Landau singularities](#landau-singularities)
18. [Torus point counts](#torus-point-counts)
19. [Conformal and BMS artifact factories](#conformal-and-bms-artifact-factories)
20. [The database](#the-database)
21. [Visualisation and export](#visualisation-and-export)
22. [Standard diagram library](#standard-diagram-library)
23. [References](#references)

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

After `pip install -e .` or `uv sync`, the `fk` command is on the PATH (run it as `uv run fk`
under uv). It has two subcommands:

```
fk analyse CNICKEL [--kinematics CLASS] [section flags] [--latex FILE] [--text FILE] [--json]
           [--sections NAMES] [--seed N] [--torus-budget N] [--db PATH | --no-db] [--verbose]
fk compare A B [--db PATH | --no-db] [--verbose]
fk --version
```

Quote every CNickel string: an unquoted `|` is a shell pipe. The bare forms `fk CNICKEL` and
`fk A B` of earlier versions still work and run `fk analyse` and `fk compare`; a first argument
without a `|` is read as a command, never as a CNickel string. `fk analyze` is the same as
`fk analyse`. The options of `fk analyse` and `fk compare` follow the command, as in
`fk analyse "12e|2e|e|" --no-db`.

### Analysing one diagram

With no section flags, `fk analyse` prints every section except the point counts of
`--torus-count`, which can take minutes. Pass one or more flags to choose.

```bash
fk analyse "12e|2e|e|:zzz"              # every section of the massless triangle
fk analyse "12e|2e|e|:zzz" -g -n        # GKZ system and Newton polytope only
fk analyse "111e|e|:zzz"                # massless banana with three propagators
fk analyse "12e|2e|e|"                  # bare topology: every propagator massless
fk analyse "12e|2e|e|:nzz" -S           # symmetries of the one-mass triangle
fk analyse "11e|e|:nn" --torus-count    # candidate master count of the massive bubble
```

| Flag | Long form | Section |
|------|-----------|---------|
| `-s` | `--symanzik` | Symanzik polynomials U, F, G |
| `-p` | `--params` | Schwinger, Feynman and Lee-Pomeransky parametrisations |
| `-g` | `--gkz` | GKZ A-matrix and Euler equations |
| `-t` | `--toric` | Toric ideal of the A-matrix |
| `-n` | `--newton` | Newton polytope: vertices, whether the integral is scaleless, normalised volume (the holonomic rank for generic $\beta$), Smith invariants |
| `-S` | `--symmetries` | Polytope automorphisms and symmetry pairs |
| | `--torus-count` | Candidate Euler characteristic from finite-field point counts; left out when no flag is given |

The graph header (CNickel string, Nickel index, loop count, propagators, external legs,
kinematic class) is printed whatever the section flags, and so is the database record unless
`--no-db` is given. The header shows the string as typed, with its canonical form beside it when
the two differ.

#### Imposing a kinematic class

A CNickel string fixes the masses of the propagators but says nothing of the legs, and `fk`
analyses it with generic invariants. `--kinematics CLASS` imposes a kinematic class first (see
[Kinematic classes](#kinematic-classes)):

| Class | Effect |
|-------|--------|
| `generic` | none; the masses must be distinct symbols (codes `n` and `z`) |
| `massless_off_shell` | none; every propagator must be massless |
| `massless_on_shell` | sets every $p_i^2$ to 0; every propagator must be massless |
| `equal_masses` | gives every propagator the mass $m_a$ of the code `a`; every propagator must be massive |

```bash
fk analyse "12e|3e|3e|e|:zzzz" --kinematics massless_on_shell -n
fk analyse "12e|3e|3e|e|:nnnn" --kinematics equal_masses --json --no-db
```

An unknown class is a usage error (status 2). A class the integral cannot take, such as
`massless_on_shell` on a graph with a massive propagator, or on one with fewer than two legs,
stops with one line on stderr and status 1. `equal_masses` changes the CNickel string, to
`12e|3e|3e|e|:aaaa` in the second example, and the header shows the new form beside the one
typed. The database records the class with the polytope.

#### Point counts

`--torus-count` counts the points of $G = 0$ in the torus over finite fields $\mathbb{F}_p$ at one
kinematic point and fits a polynomial in $p$ (see [Torus point counts](#torus-point-counts)). It
prints the point, the excluded primes and the counts, then either the candidate polynomial, Euler
characteristic and master count or the reason the counts give no candidate. These are candidates,
not proofs.

| Option | Effect |
|--------|--------|
| `--seed N` | seed for the kinematic point, 0 or more; 0 by default |
| `--torus-budget N` | maximum evaluations of $G$, 1 or more; $2 \times 10^9$ by default, enough for six propagators unless many small primes are left out, and never for seven |

Both need `--torus-count`, or `torus` among the report sections of `--sections`. `--json` does not
take `--torus-count`: name `torus` in `--sections` instead, and the summary gains
`candidate_master_count`, which is `null` when the counts give no candidate. Given
`--torus-count` and a report with the `torus` section, `fk analyse` counts once for both. A
count that needs more evaluations than `--torus-budget` allows stops with an error that names the
option.

```bash
fk analyse "11e|e|:nn" --torus-count --seed 1
fk analyse "12e|2e|e|:zzz" --json --sections polytope,torus --no-db
```

#### Reports and JSON

With these options, `fk analyse` also writes the analysis report of `FeynmanIntegral.to_latex`
and `to_text` (section 21), or summarises it as JSON. It builds the report once, however many of
the options are given, and checks that it can write each file before the analysis starts.

| Option | Effect |
|--------|--------|
| `--latex FILE` | write the report as a LaTeX document |
| `--text FILE` | write the report as plain text |
| `--sections NAMES` | comma-separated report sections from `identity`, `conventions`, `polynomials`, `representations`, `polytope`, `torus`, `gkz`, `symmetries`, `landau` and `schwinger`; all but `torus` by default |
| `--json` | print a JSON summary of the report on stdout, and nothing else |

`--sections` chooses what the report holds, and so what `--json` summarises; it does not change
the sections printed on the terminal, which the section flags choose. The report always has
`identity`, `conventions` and `polynomials`.

```bash
fk analyse "12e|2e|e|:nnn" --latex triangle.tex --text triangle.txt
fk analyse "12e|2e|e|:nnn" --text triangle.txt --sections polytope,gkz
fk analyse "12e|2e|e|:nzz" --json --no-db
```

The JSON object holds the string as typed (`input`), its canonical form (`cnickel`), the report
sections asked for (`sections`) and the numbers of `AnalysisReport.summary()` for the sections
built (`summary`), keyed in snake case, with `scaleless` as a JSON boolean and the kinematic
class as a string:

```
$ fk analyse "12e|2e|e|:nzz" --json --sections gkz --no-db
{
  "input": "12e|2e|e|:nzz",
  "cnickel": "12e|2e|e|:nzz",
  "sections": [
    "gkz"
  ],
  "summary": {
    "loops": 1,
    "propagators": 3,
    "external_legs": 3,
    "kinematic_class": "generic",
    "monomials_of_f": 4,
    "monomials_of_g": 7,
    "independent_invariants": 4,
    "codimension": 3,
    "scaleless": false,
    "toric_generators": 5
  }
}
```

`--json` still writes the files of `--latex` and `--text` and stores the integral in the
database, but does not report either on stdout. It cannot be combined with section flags, and
`--sections` needs `--latex`, `--text` or `--json`.

### Comparing two diagrams

```bash
fk compare "12e|2e|e|:zzz" "11e|e|:zz"        # triangle against bubble
fk compare "12e|2e|e|:nzz" "12e|2e|e|:znz"    # the mass on two different propagators
```

`fk compare`:
1. prints a summary of each diagram: Symanzik polynomials, A-matrix, normalised volume and Smith
   invariants;
2. tests four equivalences between the two A-configurations: `unimodular`, `affine_polytope`,
   `point_config` and `finite_index`. When the ambient dimensions differ, as for the triangle and
   the bubble, it reports the mismatch and stops. Below full dimension, as for `011e|e|:znn`, the
   `unimodular` and `affine_polytope` checks work in the lattice charts of the two vertex sets and
   `point_config` in those of all the points, `finite_index` is skipped, and the identity of a
   `point_config` map holds only trivially, since for generic $\beta$ the GKZ systems have no
   non-zero solutions;
3. prints each map it finds. A `point_config` or `finite_index` map sends every column of one
   A-matrix to a column of the other, and for such a map it also prints the column permutation
   $P$, the substitution $u_i = \prod_k v_k^{M_{ki}}$ and the identity
   $I_A(\beta, z_P) = |\det M|\, I_B(T\beta, z)$ between the two integrals without Gamma
   prefactors. A `unimodular` or `affine_polytope` map relates only the hull vertices and gives
   no identity. The `finite_index` search stops at the first map it finds, and when that map is
   singular, with $\det M = 0$, the check reports `no`, although a map with $\det M \neq 0$ may
   exist.

`fk compare` exits with status 0 when a check finds a map and with 3 when none does, including
after an ambient dimension mismatch, so a script can test the verdict. Status 0 does not mean that
a GKZ identity follows: a `unimodular` or `affine_polytope` map alone gives 0.

### Database

Both commands store their results in `feynkit.db` in the working directory. Choose another file
with `--db PATH`, or use no database with `--no-db`:

```bash
fk analyse "12e|2e|e|:zzz" --db my_survey.db
fk compare "12e|2e|e|:nzz" "12e|2e|e|:znz" --no-db
```

### Progress, errors and exit status

Output is flushed after each section, so a long analysis shows its progress. `--verbose` (`-v`)
also prints the time of each stage on stderr. `fk --version` prints the version.

| Exit status | Meaning |
|-------------|---------|
| 0 | success |
| 1 | a CNickel string that does not parse, a feynkit error, a database error or a report file that cannot be written; one line on stderr, no traceback |
| 2 | a usage error, such as a missing argument or an unknown option |
| 3 | `fk compare` found no equivalence between the two diagrams, including when their ambient dimensions differ |
| 141 | the reader of the output closed the pipe early, as `head` does; nothing on stderr. 141 is 128 + SIGPIPE, the status the shell gives `cat` or `grep` in the same place |

A string that does not parse is reported with the grammar:

```
$ fk analyse "12e|2e|e|:zz"
fk: error: cannot parse CNickel '12e|2e|e|:zz': Mass-colour length 2 does not match internal edge count 3 in '12e|2e|e|:zz'; expected TOPOLOGY or TOPOLOGY:COLOURS, where TOPOLOGY has one '|'-terminated entry per vertex naming the vertices it joins (digits) and its external legs (e), and COLOURS one mass code per propagator (z massless, n massive), as in fk analyse "12e|2e|e|:nzz"
```

### Example output (single diagram)

```
$ fk analyse "12e|2e|e|:zzz" -g -n --no-db
====================================================================
  Feynman integral  12e|2e|e|:zzz
====================================================================
  Nickel index                 12e|2e|e|
  Loop count                   1
  Propagators                  3
  External legs                3
  Kinematic class              massless_off_shell

--------------------------------------------------------------------
  GKZ hypergeometric system
--------------------------------------------------------------------
  A-matrix  (4 x 6)  [rows = coordinates; cols = monomials of G]
    [ 1  1  1  1  1  1 ]
    [ 1  1  0  1  0  0 ]
    [ 1  0  1  0  1  0 ]
    [ 0  1  1  0  0  1 ]

  beta-parameters              [-D/2, -nu_1, -nu_2, -nu_3]
  z-variables                  [z_1, z_2, z_3, z_4, z_5, z_6]

  Euler equations  (sum_j A_rj z_j d_j = beta_r):
    [0]  z_1 d_1 + z_2 d_2 + z_3 d_3 + z_4 d_4 + z_5 d_5 + z_6 d_6  =  -D/2
    [1]  z_1 d_1 + z_2 d_2 + z_4 d_4  =  -nu_1
    [2]  z_1 d_1 + z_3 d_3 + z_5 d_5  =  -nu_2
    [3]  z_2 d_2 + z_3 d_3 + z_6 d_6  =  -nu_3

--------------------------------------------------------------------
  Newton polytope
--------------------------------------------------------------------
  Monomials (A-columns)        6
  Hull vertices                6
  Ambient dimension            3
  Affine dimension             3
  Scaleless                    no
  Normalised volume            4  (the holonomic rank for generic beta)
  Smith invariants             [1, 1, 1]
  Lattice base point           (1, 1, 0)
====================================================================
  Done in 0.1s
====================================================================
```

---

## Core concepts

Feynkit works with Feynman integrals in the **Lee-Pomeransky representation**:

```
I ~ int [du] prod u_e^(nu_e - 1) G(u)^(-D/2)
```

where `G = U + F` is the Lee-Pomeransky polynomial, `U` is the first Symanzik polynomial
(spanning trees), `F` is the second (spanning 2-forests weighted by momenta), `nu_e` is the
exponent of propagator `e`, and `D` is the spacetime dimension.

The monomial support of `G` defines the **Newton polytope**. Its column-homogenised form is the
**GKZ A-matrix**. The kernel of the monomial map `z -> u^A` is the **toric ideal**. Each of its
binomials gives a differential operator in the coefficients `z_j` that annihilates the integral
(section 12).

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
order, list its higher-numbered internal neighbours (sorted ascending), then one `e` per
external leg, and end the entry with `|`. An entry can be empty: in the sunrise with one leg,
`111e||`, vertex 1 has no higher neighbours and no legs. The canonical form is the
lexicographically smallest string over all `V!` vertex labellings; an exact branch and bound
finds it without trying them all. Labels are single digits, so graphs have at most 10 vertices.
The search takes longer the larger the automorphism group of the graph, so highly symmetric
graphs with 10 vertices, such as the star and the complete graph, take from seconds to minutes,
while the graphs of the example under [Generating graphs](#generating-graphs) take milliseconds.

**CNickel format:** `<nickel>:<colours>`, with one mass code per internal edge in the order the
edges appear left to right in `<nickel>`:

| Code | Mass |
|------|------|
| `z`, `0` | zero |
| `n` | `m_<idx>`, a mass of its own |
| `a` to `y` except `n` and `s` | `m_<letter>`, shared by every edge with that letter |
| `s` | `m_s`, shared in the same way |

Digits other than `0` are rejected, since `m_1` is also the mass that `n` gives edge 1. The
canonical form jointly minimises topology and mass colouring, over the names of the shared
masses as well: it names them `a`, `b`, `c` in order of first appearance, and writes a mass
that one edge alone carries as `n`. So `12e|2e|e|:bbn` and `12e|2e|e|:ssn` both become
`12e|2e|e|:aan`, and `12e|2e|e|:nan` becomes `12e|2e|e|:nnn`.

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

For the canonical CNickel string `s` of any graph in which each vertex has a propagator or a
leg, including strings with empty entries such as `"111e||:zzz"`:

```python
Graph.from_cnickel(s).cnickel() == s   # True
```

For non-canonical input the resulting graph's `cnickel()` returns the canonical form. This
makes CNickel suitable as a stable, human-readable identifier for integral families. The parser
strips only the terminating `|`, so a superfluous one, as in `"12e|2e|e||"`, describes a vertex
with no edges and no legs and raises `ValueError`.

### Generating graphs

`generate_graphs` yields the one-particle-irreducible graphs with given numbers of loops and
legs as canonical CNickel strings, each once with every mass colouring up to its automorphisms.
A graph is kept when it is connected, bridgeless (deleting one propagator leaves the
propagators connected; legs do not count) and of degree at least 3 at every vertex, counting a
self-loop twice and each leg once. A vertex of degree 2 without a leg would only raise the
power of a propagator, or split into two graphs by partial fractions.

```python
from feynkit import generate_graphs
from feynkit.generate import mass_colourings

one_loop = list(generate_graphs(1, range(2, 7)))   # the bubble to the hexagon
two_loop = list(generate_graphs(2, range(2, 5)))   # two loops, 2 to 4 legs
print(len(one_loop), len(two_loop))
print(two_loop[:4])
print(len(mass_colourings("12e|23|3|e|")))       # the kite, from any labelling
```

prints

```
34 675
['111e|e|:nnn', '111e|e|:nnz', '111e|e|:nzz', '111e|e|:zzz']
14
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `loops`, `legs` | | One number or several. Vacuum graphs (`legs=0`) and one-leg graphs come only when asked for |
| `edges` | `None` | The numbers of propagators $E$; `None` for every $E$ the rules allow, $L \le E \le n + 3(L - 1)$ from two loops and $2 \le E \le n$ at one loop, or $1 \le E \le n$ with self-loops |
| `masses` | `"zn"` | `"zn"`: each propagator massless or with a mass of its own; `"z"`: massless only; `"shared"`: equal masses as well, as letters |
| `self_loops` | `False` | Allow propagators from a vertex to itself; massless ones are scaleless and still kept |
| `max_legs_per_vertex` | `1` | The most legs at one vertex, `None` for no limit |
| `one_vertex_irreducible` | `False` | Drop graphs whose propagators split into two sets sharing one vertex, whose integrals factorise |

The strings come in blocks of loops, propagators and legs in increasing order, sorted within a
block, and the same arguments always give the same strings. With `"zn"` the colourings are
enough for anything that depends only on the support of $G$, such as the Newton polytope and its
equivalence class, since at generic (symbolic) kinematics equal masses change coefficients but
not the support. `"shared"` matters for the Landau analysis and the point counts.

Legs carry no labels, so graphs that differ only in which momentum enters where are one string,
and by default a vertex has at most one leg: at generic kinematics several legs at a vertex give
the Newton polytope of one. `FeynmanIntegral.from_cnickel(s)` builds an integral from a string;
with fewer than two legs pass `use_mandelstam=False`. Arguments are checked when
`generate_graphs` is called: a negative number of legs, a value that is not an integer, an
unknown alphabet or a block of more than 10 vertices raises `ValidationError` before any string
is produced. `mass_colourings(topology, masses=...)` lists the colourings of one topology,
canonical or not.

The counts are checked against an independent enumeration, Burnside's lemma, closed forms at one
and two loops and OEIS A000029: the one-loop `"zn"` colourings of the $n$-gon are the binary
bracelets with $n$ beads.

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
fi.schwinger_gkz        # CayleyGKZSystem (Schwinger representation)
fi.newton_polytope      # NewtonPolytope
fi.toric_ideal          # ToricIdeal
fi.polytope_automorphisms   # PolytopeAutomorphisms
fi.graph_automorphisms      # list[list[int]], vertex permutations
fi.symmetry_pairs           # list[SymmetryPair], all integer affine maps
fi.is_scaleless             # bool, Lee's criterion of a zero sector
```

The polytope data, the point counts and the analysis report are built on request and not cached:

```python
from feynkit import polytope_data
from feynkit.io import AnalysisReport

polytope_data(fi.newton_polytope.points)   # PolytopeData: faces, facets, volume (section 11)
fi.torus_count()                           # TorusCount: point counts and candidates (section 18)
AnalysisReport.from_integral(fi)           # every fact the analysis report states (section 21)
fi.to_latex()                              # the report as a LaTeX document
fi.to_text()                               # the report as plain text
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
homogenised exponent vectors of the monomials of G. A monomial whose coefficient cancels at the
integral's kinematics, as some do on shell, is not a column, so the columns are the points of
`fi.newton_polytope`.

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

## Schwinger-representation GKZ system

`fi.schwinger_gkz` is the GKZ system obtained from the Schwinger representation instead of the
Lee-Pomeransky polynomial (Jimenez-Santacruz, Lopez-Arcos, Quintero Velez 2026). Setting the last
Schwinger parameter to one gives two polynomials, $\tilde U$ and $\tilde F$, and a two-block
Cayley A-matrix whose first two rows mark the block of each polynomial.

```python
from feynkit import FeynmanIntegral

fi = FeynmanIntegral.from_cnickel("11e|e|:nn")   # massive bubble
sys_ = fi.schwinger_gkz

print(sys_.a_matrix)          # 3 x 5: two block rows above the u exponents
print(sys_.beta_parameters)   # (nu - D, D/2 - nu, -nu_1) with nu = nu_1 + nu_2
print(sys_.w_variables, sys_.z_variables)   # coefficients of U~ and of F~
print(sys_.toric_ideal())

reduced = sys_.restrict_to_f_block()   # the paper's eq. 54: an ordinary GKZSystem
print(reduced.a_matrix, reduced.beta_parameters)
```

The parameter convention is the one used for `fi.gkz`; section 4.6 of the mathematics
reference gives the derivation, the relation to the Lee-Pomeransky system and the limits of the
reduction. In short: Britto, Grimm and Hoefnagels (arXiv:2606.09978) show that the reduced
system's solutions solve the full one when the parameter vector lies in the span of the face's
columns, which here means the exponent of $\tilde U$ vanishes; away from that point, and off cut
contours, the relation between the reduced and full systems is not established.

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

The points live in $\mathbb{R}^N$, one coordinate per Lee-Pomeransky parameter in internal-edge
order, and the polytope is usually full-dimensional: the massless triangle's has dimension 3 and
the massless box's dimension 4.

### Faces, facets and convergence

`polytope_data` takes the points and returns a `PolytopeData` with the face lattice, the facet
inequalities and the normalised volume, all computed in integer arithmetic:

```python
from feynkit import FeynmanIntegral, polytope_data

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")   # massless triangle
data = polytope_data(fi.newton_polytope.points)
print(data.dimension, data.is_full_dimensional)      # 3 True
print(data.normalized_volume)                         # 4
print(data.f_vector)                                  # (6, 12, 8, 1), an octahedron
for facet in data.facets:
    print(facet.normal, facet.offset)                 # m . x <= b
```

`PolytopeData` fields:

| Attribute | Description |
|-----------|-------------|
| `points` | The points as given, not only the vertices |
| `ambient_dimension`, `dimension` | Number of coordinates and affine dimension |
| `is_full_dimensional` | Whether the two dimensions agree |
| `vertex_indices`, `vertices` | The vertices, as indices into `points` and as points |
| `faces` | Every face as (dimension, indices of its points), vertices and polytope included |
| `f_vector` | Number of faces of each dimension, from 0 up |
| `facets` | The facets as `Facet` inequalities; empty unless the polytope is full-dimensional |
| `normalized_volume` | Volume in the lattice the point differences span, computed exactly; a unimodular simplex has 1 |
| `relative_facets` | The facets relative to the affine hull, for every polytope of dimension at least 1; equal to `facets` when the polytope is full-dimensional |
| `affine_hull` | Rows $(h_0, h_1, \ldots, h_n)$ with $h_0 + h_1 x_1 + \cdots + h_n x_n = 0$ on the polytope; empty when it is full-dimensional |
| `chart` | The `LatticeChart`: origin $o$, basis $B$ and coordinates $c_j$ with $\alpha_j = o + B c_j$ |
| `smith_invariants`, `sublattice_index` | The non-zero Smith invariants of the difference matrix, and their product $[\mathrm{sat}(L) : L]$, for $L$ the lattice the point differences span |

A `Facet` holds the primitive integer outward normal `normal` ($m$), the right-hand side `offset`
($b$) and the indices `point_indices` of the points with $m \cdot x = b$. It also records the facet
in the lattice chart, `lattice_normal` $m'$ and `lattice_offset` $b'$, and its lattice index
`lattice_index` $g_F = \gcd_j (b - m \cdot \alpha_j)$. `homogenised_form` is
$(b, -m_1, \ldots, -m_n)$, and `lattice_form` is that divided by $g_F$: the facet form primitive with
respect to the lattice $\mathbb{Z}A$ spanned by the columns $(1, \alpha_j)$ of the A-matrix, with
non-negative integer values on the columns, zero exactly on the facet. $g_F = 1$ for every facet
when `sublattice_index` is 1, for instance when the point differences span $\mathbb{Z}^n$.

The facets give the convergence region of the Lee-Pomeransky integral (Klausen 2023,
arXiv:2302.13184, section 3.3). For Euclidean kinematics, where every coefficient of $G$ has
positive real part, and $\mathrm{Re}\, D > 0$, the integral converges absolutely when

$$b\, \mathrm{Re}(D/2) - m \cdot \mathrm{Re}(\nu) > 0$$

for every facet, with $\nu = (\nu_1, \ldots, \nu_N)$ in the same edge order. When the polytope is
not full-dimensional, the integral converges for no $D$ and $\nu$. `fi.is_scaleless` says whether
it is also scaleless by Lee's criterion (R. N. Lee, arXiv:1310.1145, section 3): whether the origin
lies outside the affine hull of the polytope, so that rescaling the $u_e$ multiplies the integral
by a power of $\lambda$ that involves $D$, and dimensional regularisation sets it to zero. A
massless self-loop makes an integral scaleless; `1ee|1|:zn`, whose massless line carries no
momentum, is not full-dimensional but does not satisfy the criterion either. The analysis report
(section 21) lists one such expression per facet:

```python
from feynkit.io import AnalysisReport

report = AnalysisReport.from_integral(fi, ["representations"])
print(report.representations.convergence)
# (-D/2 + nu_1 + nu_2 + nu_3, nu_3, nu_2, D/2 - nu_1, nu_1, D/2 - nu_2, D/2 - nu_3,
#  D - nu_1 - nu_2 - nu_3): each needs a positive real part
```

### Exact faces, relative facets and lattice forms

Every face, facet and volume is computed in integer arithmetic. The facets come from an integer
beneath-beyond construction, and the list is certified complete, from the face lattice it
generates, before anything is derived from it. On this default path a failed certificate would be
a bug, and it raises `ComputationError` rather than returning a wrong face lattice. The `backend`
keyword of `polytope_data`, and of `faces` and `normalized_volume` in `feynkit.polytope`, chooses
where facet candidates come from: `"python"` (beneath-beyond, the reference, which the default
`"auto"` uses), `"qhull"` (scipy's Qhull, which can be faster for configurations with hundreds of
facets) or `"normaliz"` (PyNormaliz, if installed with `pip install PyNormaliz`; requesting it
otherwise raises `ComputationError`). Every candidate is verified exactly and the certificate
applies to every backend. When Qhull or Normaliz fails, or its list fails the certificate,
beneath-beyond takes over, so all backends return the same `PolytopeData`.

A lower-dimensional polytope is handled in its lattice chart $x = o + Bc$, in which its points
generate $\mathbb{Z}^d$ affinely. Its facets relative to the affine hull are lifted to primitive
ambient inequalities, which together with the affine-hull equations cut out the polytope. A lifted
inequality is unique only up to adding integer multiples of those equations, and feynkit picks
one:

```python
from feynkit.polytope import polytope_data

# Four points on the plane x + y + z = 1: a parallelogram of dimension 2 in R^3.
data = polytope_data([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, -1)])
print(data.dimension, data.is_full_dimensional)  # 2 False
print(data.facets)                               # (): not full-dimensional
print(data.affine_hull)                          # ((-1, 1, 1, 1),): x + y + z = 1
for facet in data.relative_facets:
    print(facet.normal, facet.offset, facet.point_indices)
# (0, -1, 0) 0 (0, 2)
# (1, 0, 0) 1 (0, 3)
# (-1, 0, 0) 0 (1, 2)
# (0, 1, 0) 1 (1, 3)
print(data.chart.coordinates)                    # ((0, 1), (1, 1), (0, 2), (1, 0))
print(data.normalized_volume)                    # 2

# (0, 0), (2, 0), (0, 1) span a sublattice of index 2 in Z^2.
triangle = polytope_data([(0, 0), (2, 0), (0, 1)])
print(triangle.sublattice_index)                 # 2
for facet in triangle.facets:
    print(facet.normal, facet.offset, facet.lattice_index, facet.lattice_form)
# (0, -1) 0 1 (Fraction(0, 1), Fraction(0, 1), Fraction(1, 1))
# (-1, 0) 0 2 (Fraction(0, 1), Fraction(1, 2), Fraction(0, 1))
# (1, 2) 2 2 (Fraction(1, 1), Fraction(-1, 2), Fraction(-1, 1))
```

The lattice index differs per facet: $y \ge 0$ has $g_F = 1$, while $x \ge 0$ and $x + 2y \le 2$ have
$g_F = 2$. A Smith invariant above 1 is necessary for this but not sufficient: the facets of
$(0,0), (4,0), (2,2), (2,1)$ all have $g_F = 1$ although the differences of those points span a
sublattice of index 2. The convergence inequalities above are unchanged by positive scaling and
keep using $(m, b)$.

---

## Toric ideal

The toric ideal is the ideal of polynomial relations among the monomials of G. Each binomial
generator $z^k - z^l$ gives the operator $\partial^k - \partial^l$ in the coefficients $z_j$, which
annihilates the integral with the $z_j$ treated as independent. These operators are an analogue of
integration-by-parts (IBP) relations, not IBP relations (Chestnov et al. 2022,
arXiv:2204.12983); see section 6.3 of the mathematics reference.

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
columns of $A$ are linearly independent. That says nothing about the number of master integrals.

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

Below full dimension the two Newton polytopes must have the same dimension, and the tests run in
their lattice charts. The unimodular witness $U \in GL_N(\mathbb{Z})$ maps the direction space of
one affine hull onto that of the other, and a complement of the first onto a complement of the
second; the affine witness is invertible, and its determinant depends on those complements. The
affine tests accept rational points in every dimension: each configuration is scaled by the least
common denominator of its coordinates, and the witness is scaled back. The unimodular test needs
integers. `012e|2e|e|:znnn` and `12e|12e|e|:nnzn`, one graph with its massless self-loop on two
different vertices, are unimodularly equivalent.

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
subgroup, and the full set of symmetry pairs, the integer affine self-maps of the
point configuration.

See [`docs/automorphism_groups.md`](automorphism_groups.md) for the mathematical
background, physical interpretation, and full results for standard diagrams.

### Three levels of symmetry

| Object | API | Description |
|--------|-----|-------------|
| `PolytopeAutomorphisms` | `fi.polytope_automorphisms` | Aut(P): all (U,t) with U in GL_n($\mathbb{Z}$), \|det U\|=1, permuting the vertices of P; below full dimension, one such map per automorphism of P in its affine hull |
| `list[list[int]]` | `fi.graph_automorphisms` | Vertex permutations preserving topology and mass colouring |
| `list[int]` | `coefficient_preserving_indices(fi, auts)` | Indices into auts.maps whose (U,t) also preserves G's coefficients |

### Symmetry pairs

`fi.symmetry_pairs` returns all integer affine maps between the point configuration and itself.
Each has $|\det M| = 1$: $P$ has finite order $k$, so $M^k = I$. Maps with $|\det M| > 1$ relate two
different configurations (`finite_index_map`). Each pair gives the identity
$I_A(\beta, z_P) = I_A(T\beta, z)$ for the integral without Gamma prefactors, with
$z_P = (z_{P(1)}, \ldots, z_{P(N)})$; section 8 of the mathematics reference states it with its
conventions. Below full dimension the pairs are found in the lattice chart of the points, and each
is one extension of a map of the affine hull to $\mathbb{Z}^N$; their identities then hold only
trivially, since for generic $D$ and $\nu$ the GKZ system has no non-zero solutions. Each
`SymmetryPair` records:

| Field | Type | Description |
|-------|------|-------------|
| `linear_map` | `sp.ImmutableMatrix` | The integer linear map M |
| `translation` | `sp.ImmutableMatrix` | Translation vector t |
| `determinant` | `int` | \|det(M)\|, always 1 for a self-map |
| `is_unimodular` | `bool` | Shorthand for `determinant == 1` |
| `column_permutation` | `tuple[int, ...]` | Induced permutation on A-matrix columns |

```python
pairs = fi.symmetry_pairs
print(len(pairs), all(p.is_unimodular for p in pairs))   # 48 True for the massless triangle
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

The vertices and edges come from the certified face lattice of `polytope_data`. A Newton polytope
that is not full-dimensional, such as that of a graph with a massless self-loop, is searched in
the lattice chart of its vertices, where it is full-dimensional. Its group is that of the polytope
in its affine hull: the affine maps of the affine hull that preserve the integer points on it and
the polytope. Each is returned as one unimodular map $(U, t)$ of $\mathbb{Z}^N$ that extends it,
whose linear part $U$ fixes a complement of the direction space of the affine hull:

```python
znnn = FeynmanIntegral.from_cnickel("012e|2e|e|:znnn")   # massive triangle, massless self-loop
print(znnn.polytope_automorphisms.order)                   # 6, as for 12e|2e|e|:nnn
```

### PolytopeAutomorphisms fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `maps` | `list[tuple[ImmutableMatrix, ImmutableMatrix]]` | All (U, t) pairs |
| `order` | `int` | Group order \|Aut(P)\|, the number of vertex permutations |
| `vertex_permutations` | `list[list[int]]` | Induced permutation of the vertices per automorphism, in the order of `polytope_data(points).vertices` |
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
| `cfg.normalized_volume` | Normalised volume of the Newton polytope in the lattice the point differences span, computed exactly; for a full-dimensional configuration, the holonomic rank for generic $\beta$; raises `ValidationError` on an empty configuration |
| `cfg.newton_polytope_points` | Hull vertex coordinates |
| `cfg.intrinsic_model()` | `IntrinsicModel`, the points in the Hermite normal form basis of the lattice their differences span, relative to the first point |

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

pairs = symmetry_pairs(cfg)   # every pair has |det M| = 1
```

For maps with $|\det M| > 1$ between two different configurations use `finite_index_map`
(see Equivalence above).

### Intrinsic lattice model

`intrinsic_model()` writes the points in a basis of the lattice $L$ spanned by their differences
from the first point: the Hermite normal form basis of `feynkit.polytope.lattice_chart`. Point $i$
is $\alpha_0 + \sum_t c_{i,t} b_t$, where $\alpha_0$ is `base_point`, $c_i$ is `intrinsic_coords[i]`
and $b_t$ is `basis[t]`. The coordinates are integers and those of the first point are 0, whatever
the dimension of the points:

```python
from feynkit import AConfiguration
import sympy as sp

# (0, 0), (2, 4) and (3, 6): three points on a line in Z^2.
line = AConfiguration(sp.Matrix([
    [1, 1, 1],
    [0, 2, 3],
    [0, 4, 6],
]), is_homogenized=True)
model = line.intrinsic_model()
print(model.base_point)        # (0, 0): the first point
print(model.basis)             # ((1, 2),): a basis of L
print(model.intrinsic_coords)  # ((0,), (2,), (3,))
print(model.intrinsic_rank)    # 1: the affine dimension
print(model.smith_invariants)  # [1]
```

`intrinsic_rank` is the affine dimension of the points, not the holonomic rank of the GKZ system.
These points are not full-dimensional, so the rows of the homogenised $A$ are linearly dependent,
and for generic $\beta$ the system has no non-zero solutions. The lattice chart itself shifts its
coordinates to be non-negative, so its origin need not be one of the points; the model keeps the
first point as origin.

---

## Landau singularities

`feynkit.landau` computes the reduced principal A-determinant of the Lee-Pomeransky polynomial
$G$: the product, over every face of the Newton polytope, of the A-discriminant of $G$ restricted
to that face (GKZ 1994, chapter 10). Its irreducible kinematic factors are candidate singular
surfaces of the integral. See section 10 of the mathematics reference for what each kind of face
contributes and for the caveats.

### From a FeynmanIntegral

```python
from feynkit import FeynmanIntegral, landau_analysis

fi = FeynmanIntegral.from_cnickel("11e|e|:nn")   # massive bubble
la = landau_analysis(fi)

print(la.landau_surfaces)          # m_1, m_2, s, s - (m_1 + m_2)^2, s - (m_1 - m_2)^2
print(la.principal_a_determinant)  # their product
for face in la.face_discriminants:
    print(face.dimension, face.is_simplex, face.discriminant)
```

### From a polynomial directly

```python
from feynkit import landau_analysis_from_polynomial
import sympy as sp

u1, u2, u3 = sp.symbols("u1:4")
p1, p2, p3 = sp.symbols("p1^2 p2^2 p3^2")
G = u1 + u2 + u3 - p1*u1*u2 - p2*u1*u3 - p3*u2*u3   # massless off-shell triangle
la = landau_analysis_from_polynomial(G, [u1, u2, u3])
```

### One-loop closed form

For one-loop graphs `one_loop_landau_surfaces(fi)` returns the same factors, for generic
kinematics and when no face is skipped, from the principal minors of the modified Cayley matrix
(Dlapa, Helmer, Papathanasiou, Tellander 2023). With special kinematics it can keep a factor the
faces miss (see [Backends](#backends)), and it keeps those of skipped faces: at the default
`max_face_points` the faces miss 1 of the 32 factors of the massless pentagon and 8 of the
hexagon's 79. It is fast, needs no Gröbner basis, and is what the test-suite checks the face
computation against. The massless pentagon takes about half a second, the massless hexagon about
7 s and the all-massive hexagon about 80 s. With Singular on the path the minors are factored
there, in one run. Without it SymPy factors them, which usually takes about as long but now and
then far longer: SymPy's factorisation draws evaluation points from a random generator the whole
process shares, and from some of its states a single minor takes minutes.

`one_loop_landau_surfaces_by_type(fi)` splits the same factors by kind of minor. Principal minors
that leave out the bordering first row and column of the modified Cayley matrix give first-type
(Cayley) factors, and those that keep it give second-type (Gram) factors. A factor that arises
from both kinds is listed in both.

```python
from feynkit import one_loop_landau_surfaces_by_type

first, second = one_loop_landau_surfaces_by_type(fi)   # massive bubble
print(first)    # m_1, m_2, s - (m_1 + m_2)^2, s - (m_1 - m_2)^2
print(second)   # s
```

A graph with bridges, internal edges on no cycle, is its cycle with trees attached. Its matrix is
that of the cycle, with the legs of each tree moved to the vertex where the tree meets the cycle,
and each bridge $b$ adds the pole $m_b^2 = q_b^2$ of its propagator, $q_b$ being the momentum
through it. `one_loop_bridge_poles(fi)` returns these factors; `one_loop_landau_surfaces(fi)`
includes them and `one_loop_landau_surfaces_by_type(fi)` does not.

```python
from feynkit import one_loop_bridge_poles

fi = FeynmanIntegral.from_cnickel("11e|2|e|:nnn")   # massive bubble with a massive bridge
print(one_loop_bridge_poles(fi))                     # (-m_3**2 + s,)
```

### Backends

Faces of dimension two or more that are not simplices need an elimination ideal. It is taken at
$\mu = 1$ when every coefficient of $G$ is $\mu^k$ times a factor free of $\mu$, with $k$ an affine
function of the exponent, as when the kinematics are free of $\mu$, and with $\mu$ as a variable
otherwise. When the face's coefficients are independent linear forms in the invariants and squared
masses, it is taken in fresh symbols for them. A face whose elimination ideal has several
generators contributes the factors of their greatest common divisor, the codimension-one part of
its locus. A component that arises only as a limit under special kinematics can then be missing,
while the one-loop closed form keeps it: with $p_1^2 = 0$ the massive triangle `12e|2e|e|:nnn`
loses $p_2^2 - p_3^2$, since the locus of its top face is $p_2^2 = p_3^2 = 0$, of codimension two.
The massless pentagon and hexagon take about 3 s and 15 s. feynkit eliminates and factors with
Singular when the `Singular` binary is on the path, except for the discriminants of edges, small
polynomials that SymPy factors to write them, and falls back to SymPy otherwise, which is much
slower and whose factorisation now and then takes minutes. Faces with more monomials than
`max_face_points` (default 12) are skipped and listed in `la.skipped_faces`.
`landau_analysis(fi, timeout=60)` gives Singular at most 60 s for each face, with no limit by
default. An elimination that runs past it, fails or prints output feynkit cannot read raises
`ComputationError`; a factorisation that fails is left to SymPy.

### LandauAnalysis fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `la.face_discriminants` | `tuple[FaceDiscriminant, ...]` | One per face of the Newton polytope |
| `la.principal_a_determinant` | `sp.Expr` | Product of the distinct kinematic factors |
| `la.landau_surfaces` | `tuple[sp.Expr, ...]` | The factors themselves |
| `la.skipped_faces` | `tuple` | Faces too large to eliminate |

### FaceDiscriminant fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `face.dimension` | `int` | Dimension of the face |
| `face.exponents` | `tuple[tuple[int, ...], ...]` | Exponent vectors on the face |
| `face.coefficients` | `tuple[sp.Expr, ...]` | Kinematic coefficients |
| `face.discriminant` | `sp.Expr` | Discriminant of the restriction, 1 if trivial |
| `face.is_simplex` | `bool` | Lattice points affinely independent |
| `face.principal` | `bool` | Elimination ideal had a single generator; with several, the discriminant holds the factors of their greatest common divisor |

---

## Torus point counts

`FeynmanIntegral.torus_count` counts the points of $V = \{G = 0\}$ in the torus $(\mathbb{F}_p^*)^N$
for finitely many primes $p$, at one rational kinematic point and with $\mu = 1$, and fits a
polynomial $P(q)$ to the counts. Let $X$ be the complement of $V$ in the complex torus
$(\mathbb{C}^*)^N$. The number of master integrals, subsectors included, symmetries unused and $D$
symbolic, is $C = (-1)^N \chi(X)$ (Bitoun, Bogner, Klausen and Panzer 2019, arXiv:1712.09215,
Corollary 37). When the fit passes the tests below, $\chi(X) = -P(1)$ is a candidate Euler
characteristic and $C$ a candidate master count. The normalised volume counts master integrals only
for generic coefficients and when the exponent differences span $\mathbb{Z}^N$; the point counts
look at the physical coefficients.

Every result is a candidate, not a proof. Katz's theorem (appendix to Hausel and Rodriguez-Villegas
2008, arXiv:math/0612668, Theorem 6.1.2(3)) gives $\chi(V) = P(1)$ when $\#V(\mathbb{F}_q) = P(q)$
for every finite field $\mathbb{F}_q$ whose characteristic avoids a finite set, and counts over
finitely many prime fields cannot establish that. Section 4.5 of the mathematics reference states
the master count and its bound, and section 4.7 the point counts.

```python
import sympy as sp

from feynkit import FeynmanIntegral

fi = FeynmanIntegral.from_cnickel("11e|e|:nn")  # the massive bubble
count = fi.torus_count(seed=0)
print(", ".join(f"{key} = {value}" for key, value in count.point))
print(count.fit_primes, count.verification_primes)
print(count.candidate_polynomial, count.candidate_master_count)

# lambda(1, 1, 1) = -3 is not a square: p - 3 - (-3/p) points, not a polynomial in p.
s = sp.Symbol("s", real=True)
m1, m2 = (edge.get_mass() for edge in fi.graph.get_internal_edges())
generic = fi.torus_count(point={s: 1, m1**2: 1, m2**2: 1})
print(generic.candidate_master_count, generic.reason)

# On the threshold s = (m_1 + m_2)^2 the count is p - 3, and C drops from 3 to 2.
threshold = fi.torus_count(point={s: 9, m1**2: 1, m2**2: 4}, allow_singular=True)
print(threshold.on_landau_surface, threshold.candidate_master_count)
```

prints

```
m_1**2 = 3, m_2**2 = 18, s = 6
(7, 11, 13) (17, 19, 23, 29)
(-4, 1) 3
None the polynomial through the fit counts has non-integer coefficients
True 2
```

For non-zero $s$, $m_1$ and $m_2$, the massive bubble has $p - 3 - (\lambda/p)$ points for every
prime not left out, with $\lambda$ the Källén function of $s$, $m_1^2$ and $m_2^2$: a polynomial in
$p$ only when $\lambda$ is the square of a rational number. At seed 0 the draw makes $\lambda = 9$,
and the counts fit $P(q) = q - 4$ (`candidate_polynomial` lists the coefficients from the constant
term up), so $C = 3$. At $s = m_1^2 = m_2^2 = 1$ the master count is 3 as well, but the counts give
no candidate: a missing candidate says nothing about $C$.

### How the point is chosen

Without `point`, the kinematic symbols of $G$ are drawn with `random.Random(seed)` in the order of
their names: each invariant from the non-zero integers of $[-20, 20]$, and the square $m_e^2$ of
each mass, or of any symbol that occurs in $G$ only to even powers, from 1 to 20. A draw is
admissible when every coefficient of $G$ and every face discriminant of the Landau analysis is
non-zero; 10,000 draws without an admissible one raise `ValidationError`. Of the first 200
admissible draws, the first at which every square-test factor is a non-zero rational square is used,
and otherwise the first admissible draw. The square-test factors are the irreducible factors of the
principal discriminants of the faces of dimension at least 1. When the Landau analysis has
eliminated at $\mu = 1$ (see Backends, under Landau singularities), a face with monomials of both
$U$ and $F$ whose monomials of $F$ are affinely dependent is left out; on every graph tried the test
then takes the faces it took when the analysis kept $\mu$ as a variable. The factors are written in
the squared masses: the Landau analysis factorises in the masses, and a factor $f$ odd in a mass $m$
is replaced by its norm $f(m) f(-m)$.

A draw where the factors are all squares gives polynomial counts far more often. Massive graphs and
off-shell legs often give no candidate, since many factors must be squares at once. The off-shell
massless box `12e|3e|3e|e|:zzzz` gives no candidate at any seed from 0 to 149, while at
$p_i^2 = (-12, -6, 5, -4)$, $s_{12} = 20$ and $s_{23} = -41$, where its five square-test factors are
squares, it gives $C = 11$.

`point` gives the point instead, as a rational value for each kinematic symbol of $G$. A symbol that
occurs only to even powers, such as a mass, can be keyed by its square instead, so `{m1**2: 4}` and
`{m1: 2}` give the same point. A point on a Landau surface, where a coefficient of $G$ or a face
discriminant vanishes, raises `ValidationError` unless `allow_singular=True`; there $C$ may be
smaller than for generic kinematics, and the result has `on_landau_surface` set.

`on_shell` substitutes into the momentum products before anything else, so that legs can be put on
shell. Its keys are symbols of the momentum products with their assumptions, such as
`sp.Symbol("p1^2", real=True)`, not the string `"p1^2"`. Its values are exact: ints, `Fraction`s,
strings or SymPy expressions without floats. They may not contain a key, and a symbol in them that
shares its name with one of the integral's symbols must be that symbol, so the string `"s12"` is
refused for the box below. SymPy parses string values, so `"p2^2"` means `p2**2`, the square of a
new symbol `p2` that is then drawn like the others; pass SymPy symbols for invariants.

```python
import sympy as sp

from feynkit import FeynmanIntegral

box = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")        # the massless box
legs = {sp.Symbol(f"p{i}^2", real=True): 0 for i in range(1, 5)}
print(box.torus_count(on_shell=legs).candidate_master_count)  # 3
```

The Landau analysis runs on the integral with `on_shell` applied and $\mu$ symbolic, before $\mu$ is
set to 1; `landau` passes one already computed for that integral. An integral with
`kinematic_constraints` raises `ValidationError`, since `torus_count` does not apply them;
substitute them with `on_shell` instead.

### Primes, the fit and its check

The primes left out are 2 and, up to `max_prime` (1000), those dividing the numerator or the
denominator of a non-zero value at the point of a coefficient of $G$, a face discriminant, an
irreducible factor of one, or the discriminant of $G$ on an edge of its Newton polytope. The rule is
heuristic: it misses the integer content of eliminants, skipped faces and components the Landau
analysis does not list. A bad prime it misses would, provided some check prime is good, make the
fit or its check fail rather than give a wrong candidate.

$G$ is solved for the first variable of lowest degree, which is at most 2 for a Feynman graph, so
each count is a sum, over the other variables, of numbers of roots in $\mathbb{F}_p^*$ of
quadratics. The polynomial of degree at most $N$ through the counts at the first $N + 1$ primes not
left out is computed exactly. It is then tested in this order, and the first test it fails leaves
the candidates None and gives the `reason`:

1. Its coefficients are integers, as those of a counting polynomial must be.
2. Its $q^N$ term is zero, since $V$ has dimension $N - 1$.
3. $0 \le C \le N!\,\mathrm{Vol}(\mathrm{Newt}\,G)$, where $\mathrm{Newt}\,G$ is the Newton polytope
   of $G$ at the point and $\mathrm{Vol}$ its Euclidean volume (Bitoun et al. 2019,
   arXiv:1712.09215, Theorem 44, after Kouchnirenko, Invent. Math. 32 (1976) 1). The bound is
   `normalized_volume * sublattice_index` of its `polytope_data`, or 0 when the polytope is not
   full-dimensional.
4. No edge of $\mathrm{Newt}\,G$ has lattice length 3 or more, and no face of dimension 2 or more
   has a quotient $L_\mathrm{sat}/L$ of exponent 3 or more, where $L$ is the lattice spanned by the
   differences of the face's points, $L_\mathrm{sat}$ the integer points of their span, and the
   exponent the largest Smith invariant of $L_\mathrm{sat}/L$, not its index. Otherwise the counts
   may depend on characters of order above 2, which the check does not cover, and only the fit
   primes are counted. When $G$ has degree at most 2 in every variable, as for a Feynman graph,
   every edge has lattice length at most 2.
5. It agrees with the counts at the next `verification` primes (4 by default), continued until every
   non-trivial product of the quadratic characters the check covers has taken both signs on the fit
   and check primes together.

The characters covered are $(d/p)$ for $d$ in the group generated by $-1$ and the non-zero values at
the point of the vertex coefficients of $G$, the irreducible factors of every face discriminant,
principal or not, written in the squared masses, and the discriminants of $G$ on its edges, each in
the edge's lattice coordinate. Characters outside that group are not covered. They can come from the
constant factors of the discriminants of faces of dimension 2 or more, or from the discriminants of
skipped faces, and a count that depends on one passes the check if the primes counted agree on it.
Running out of primes up to `max_prime` before the check is done raises `ValidationError`.

### Cost

A prime $p$ costs $(p - 1)^{N - 1}$ evaluations of $G$, at about $10^7$ a second, and a run that
reaches a candidate counts at least $N + 1 +$ `verification` primes, $N + 5$ with the default
`verification` of 4. With that default, and besides the Landau analysis, five propagators take
seconds and six about a minute. Seven need a larger `max_evaluations`: at least $7.6 \times 10^9$
evaluations, about 13 minutes, when only 2 is left out, and about two hours when 3, 5, 7 and 19 are
left out as well. `max_evaluations`, $2 \times 10^9$ by default, is compared with the cost of the
fit primes and the first `verification` check primes before anything is counted, and with the cost
of each further check prime before it is counted; going over raises `ValidationError`. With the
default `verification`, the default budget is enough for six propagators unless many small primes
are left out, and never for seven; with `verification` of 1 or 2, seven can pass the comparison made
before counting.

### TorusCount fields

| Attribute | Description |
|-----------|-------------|
| `variables`, `eliminated` | The variables of $G$, and the one solved for |
| `point` | The kinematic point as (symbol, value) pairs in the order of the symbol names; a symbol that occurs only to even powers, such as a mass, appears as its square |
| `on_shell` | The substitutions made first, as (symbol, value) pairs; empty from `count_torus_points` |
| `seed` | The seed of the draw, or None for a given point |
| `on_landau_surface` | Whether a coefficient of $G$ or a face discriminant vanishes at the point |
| `excluded_primes`, `max_prime` | The primes left out, and the largest prime the count could use, up to which they are listed |
| `fit_primes`, `verification_primes` | The primes fitted and checked, in order |
| `counts` | $(p, \#V(\mathbb{F}_p))$ for every prime counted |
| `candidate_polynomial` | The $N$ coefficients of $P(q)$, constant term first, or None |
| `candidate_euler_characteristic`, `candidate_master_count` | $\chi(X) = -P(1)$ and $C = (-1)^N \chi(X)$, or None |
| `reason` | Why there is no candidate, or None |
| `skipped_faces` | The number of faces the Landau analysis skipped; their discriminants take no part in the draw, the excluded primes or the check |
| `backend` | `"numpy"` or `"flint"` |

### Any polynomial, and a cross-check

`count_torus_points` in `feynkit.point_count` counts for any polynomial $G$ of degree at most 2 in
one of the given variables, and takes the energy scale as `scale` if $G$ has one. It has the
keywords of `torus_count` except `on_shell` and `max_face_points`, and two more: `volume_bound`,
which replaces $N!\,\mathrm{Vol}(\mathrm{Newt}\,G)$, and `max_prime`, the largest prime counted,
1000 by default and at most about $10^9$, where the int64 counts stop being exact. In both functions
`backend="flint"` counts with python-flint instead of numpy, point by point, at 5 to $8 \times 10^4$
evaluations a second: a cross-check 75 to 180 times slower. It raises `RuntimeError` unless
python-flint is installed.

`critical_point_count(g, variables, point)` counts the critical points of
$\sum_e \nu_e \log u_e - (D/2) \log G$ on $X$ at random rational exponents drawn with `seed`; for
generic exponents they number $|\chi(X)|$ (Fevola, Mizera and Telen 2024, arXiv:2311.16219, proof of
Theorem 3.1, after Huh 2013, arXiv:1207.0553). $G$ must have $\mu = 1$ already, and `point` gives
every other symbol, keyed as for `count_torus_points`. Singular computes the number modulo the two
largest primes below $2^{31}$ that divide no numerator or denominator of the coefficients and
exponents. A count modulo a prime can differ from the count over $\mathbb{Q}$, which it equals for
all but finitely many primes, so this is a cross-check, not a certificate; requiring the two results
to agree guards against an unlucky prime. It raises `RuntimeError` without Singular, and
`ComputationError` when the two results differ, when the critical points are not finite at the
exponents drawn, or when Singular runs past `timeout` seconds (300 by default).

```python
import sympy as sp

from feynkit.point_count import count_torus_points, critical_point_count

u1, u2, s = sp.symbols("u1 u2 s")
g = u1 + u2 - s * u1 * u2                                     # the massless bubble at mu = 1
print(count_torus_points(g, [u1, u2]).candidate_polynomial)  # (-2, 1)
print(critical_point_count(g, [u1, u2], {s: 3}))             # 1
```

The massless bubble has $p - 2$ points, so $\chi(X) = -P(1) = 1$ and $C = 1$, and its log-likelihood
function has one critical point.

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

A file written by feynkit 0.4.0 or earlier is repaired when it is first opened for writing; a
read-only or locked file opens as it is and is repaired at a later open. Those releases kept
monomials of G whose coefficients cancel, so a row can have more A-matrix columns than Newton
points; it gets the matrix of its points, and `n_toric_gens` is `None` until the toric ideal is
next computed. They also took the vertices of each Newton polytope from a floating convex hull, so
the repair deletes every cached equivalence verdict, to be recomputed by the next search, and sets
the automorphism fields to `None` until the next `store(..., compute_automorphisms=True)`.

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

### Analysis report

`AnalysisReport` (in `feynkit.io`) holds everything feynkit computes for one integral, section by
section, as a frozen dataclass. `render_latex` turns it into a LaTeX `article` and `render_text`
into plain ASCII text stating the same facts in the same order. `fi.to_latex()` and `fi.to_text()`
build the report and render it in one call:

```python
from pathlib import Path
from feynkit import FeynmanIntegral

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")   # massive triangle
Path("triangle.tex").write_text(fi.to_latex())        # compile with pdflatex
print(fi.to_text())
```

Both methods take the same arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `sections` | all but `torus` | Names of the sections to build, from `SECTION_NAMES` |
| `title` | "Feynman integral" and the CNickel string | Document title; `to_latex` escapes it |
| `max_face_points` | 12 | Faces of the Newton polytope with more monomials are left out of the Landau analysis and listed as skipped |

The section names, in `feynkit.io.report.SECTION_NAMES`, are `identity`, `conventions`,
`polynomials`, `representations`, `polytope`, `torus`, `gkz`, `symmetries`, `landau` and
`schwinger`. The first three are always built. The rest are built only when named, so a survey can
ask for a short report without the automorphism and Landau computations. Without `sections` every
section but `torus` is built, the tuple `DEFAULT_SECTIONS`. The point counts of `torus` take seconds
for five propagators, and seven exceed the default budget (see
[Torus point counts](#torus-point-counts)); `to_latex` and `to_text` count with seed 0 and the
default budget, and raise `ValidationError` when the count cannot run, as for an integral with
kinematic constraints.

```python
latex = fi.to_latex(["polytope", "gkz"], title="Massive triangle")
```

The Landau section dominates the build time: the kite `12e|23|3|e|:zzzzz` takes about 5 s in all,
3.4 s of it in the Landau analysis, and the massive box `12e|3e|3e|e|:nnnn` about 4 s, 2 s of it
in the analysis. A survey over many graphs can leave it out and keep everything else:

```python
from feynkit.io.report import DEFAULT_SECTIONS

text = fi.to_text([name for name in DEFAULT_SECTIONS if name != "landau"])
```

From the command line, `fk analyse "12e|2e|e|:nnn" --latex triangle.tex --text triangle.txt`
builds the report once and writes both documents; see [CLI: fk](#cli-fk).

To render one report twice, or to read its facts directly, build it once and pass it to the
renderers. `AnalysisReport.from_integral` takes the same `sections` and `max_face_points`,
`figure_max_vertices` (default 12): a polytope with more vertices gets no figure, and `torus_seed`
and `torus_budget` (defaults 0 and $2 \times 10^9$), the `seed` and `max_evaluations` of the point
counts. The `torus` and `landau` sections share one Landau analysis.

```python
from feynkit.io import AnalysisReport, render_latex, render_text

report = AnalysisReport.from_integral(fi)
print(report.summary())            # the summary table as (label, value) rows
latex = render_latex(report)
text = render_text(report, title="Massive triangle")
```

The document has up to thirteen parts:

1. The title; no author, date or abstract.
2. A summary table: loops, propagators, external legs, the monomial counts of $F$ and $G$,
   independent invariants, codimension, whether the integral is scaleless, polytope vertices,
   normalised volume, the candidate master count (with `torus`; "none" when the counts give no
   candidate), $|\mathrm{Aut}(P)|$, toric generators and Landau surfaces.
3. The graph: a TikZ figure and a table of the propagators with their endpoints, exponents and
   masses.
4. Conventions: the momentum-space integral and its normalisation, $D = D_0 - 2\epsilon$, the
   metric, the kinematic invariants (or, for a vacuum graph, that there are no external momenta)
   and the definition of $F$.
5. The Symanzik polynomials $U$, $F$ (with its $1/\mu^2$ factored out) and $G$, their degrees and
   monomial counts, the coefficients $z_j$ at their physical values, and the codimension against
   the number of independent invariants.
6. The Schwinger, Feynman and Lee-Pomeransky representations, written in $U$, $F$ and $G$, and the
   convergence region of the last as one inequality per facet (section 11).
7. The Newton polytope: vertices, dimension, normalised volume, face counts, a figure, and the
   conditions under which the holonomic rank equals the volume or, when the polytope is not
   full-dimensional, why the rank is 0 for generic $\beta$ and whether the integral is scaleless
   by Lee's criterion (R. N. Lee, arXiv:1310.1145, section 3).
8. The candidate Euler characteristic from point counts, only when `torus` is named: the master
   count and Katz's theorem, the kinematic point, the excluded primes, the characters the check
   covers, a table of the counts, and the candidate polynomial, Euler characteristic and master
   count, or why the counts give no candidate.
9. The GKZ system: $A$, $\beta = (-D/2, -\nu_1, \ldots, -\nu_N)$, one Euler operator per row of $A$
   and the toric generators.
10. Symmetries: the order and vertex orbits of $\mathrm{Aut}(P)$, the graph automorphisms, the
    coefficient-preserving subgroup, and the symmetry pairs with the identity each gives. For a
    Newton polytope that is not full-dimensional, $\mathrm{Aut}(P)$ is its group in its affine
    hull, and the section counts the symmetry pairs and says why their identities hold only
    trivially.
11. Landau surfaces: the factors of the reduced principal A-determinant by face dimension, split
    into first and second type for one-loop graphs, with the skipped faces, each named by its
    dimension and number of points, and the caveats.
12. The Schwinger-representation system (section 10), its equivalence to the Lee-Pomeransky
    configuration and its reduction to the $\tilde F$ block.
13. References, the works cited in order of first citation.

Parts 6 to 12 appear when their sections are built. When the `polytope` section is built and the
polytope is full-dimensional, the document also states what feynkit does not compute: the
holonomic rank at the physical point, the Euler characteristic that counts the master integrals,
series solutions, a Pfaffian system and the restriction of the GKZ system to physical kinematics.
With the `torus` section, that sentence points to the candidate Euler characteristic instead, or
says that the counts give no candidate.

The LaTeX source is ASCII and needs only standard TeX Live packages (amsmath, booktabs, longtable,
geometry, lmodern, TikZ with tikz-3dplot, hyperref). The test `tests/io/test_report_compile.py`
compiles the reports of the massive bubble, the massless triangle and the massless box with
pdflatex and checks that they have no errors, overfull lines or undefined references. It is
skipped when pdflatex is not installed, and the box also needs Singular. The files
`docs/triangle_analysis.tex`, `.pdf` and `.txt` are the report of the massive triangle.

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
   [arXiv:1907.00507](https://arxiv.org/abs/1907.00507)

3. Lee, R.N. and Pomeransky, A.A. (2013). Critical points and number of master integrals.
   *JHEP* **11**, 165. [arXiv:1308.6676](https://arxiv.org/abs/1308.6676)

4. Gelfand, I.M., Kapranov, M.M., Zelevinsky, A.V. (1994). *Discriminants, Resultants and
   Multidimensional Determinants.* Birkhäuser.

5. Sturmfels, B. (1996). *Gröbner Bases and Convex Polytopes.* American Mathematical Society.

6. Bogner, C. et al. (2017). Loopedia, a database for loop integrals.
   [arXiv:1709.01266](https://arxiv.org/abs/1709.01266)

7. Liu, Y. and Cai, B. (2025). Unimodular isomorphism of lattice polytopes.
   [arXiv:2506.23846](https://arxiv.org/abs/2506.23846)
