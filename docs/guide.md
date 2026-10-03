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
19. [Degenerate faces](#degenerate-faces)
20. [Conformal and BMS artifact factories](#conformal-and-bms-artifact-factories)
21. [The database](#the-database)
22. [Visualisation and export](#visualisation-and-export)
23. [Standard diagram library](#standard-diagram-library)
24. [References](#references)

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

### Optional: PyNormaliz and python-flint

```bash
pip install -e ".[backends]"    # from source; pip install "feynkit[backends]" from an index
```

The `backends` extra installs two Python packages, which `uv sync` also installs for development:

- PyNormaliz, on Linux and macOS only. The lattice invariants use it to decide the integer
  decomposition property, and past their work budget to give the Ehrhart polynomial, which in
  pure Python take minutes or hours from about dimension 8 (section on lattice invariants below);
  `polytope_data(backend="normaliz")` takes facet candidates from it.
- python-flint. `count_torus_points(backend="flint")` counts the points of $G = 0$ with it, point
  by point: slower than the default numpy backend, it is an independent cross-check of the counts.
  Installing it also makes SymPy use FLINT for its integers and polynomials everywhere, which is
  faster; set `SYMPY_GROUND_TYPES=python` to keep SymPy's own. The test suite runs both ways. The
  Landau analysis factors its polynomials with python-flint when Singular is not on the path,
  which is far quicker than SymPy's factorisation and does not depend on the state of its random
  generator.

feynkit detects both at run time; without them the pure-Python paths, the reference, are used.

---

## CLI: fk

After `pip install -e .` or `uv sync`, the `fk` command is on the PATH (run it as `uv run fk`
under uv). It has two subcommands:

```
fk analyse CNICKEL [--kinematics CLASS] [--set SYMBOL=VALUE ...] [section flags] [--d0 VALUE] [--latex FILE] [--text FILE]
           [--json] [--sections NAMES] [--limits] [--seed N] [--torus-budget N] [--degeneracy-mode MODE]
           [--hasse FILE [--hasse-view VIEW] [--hasse-codim K] [--hasse-face POINTS] [--hasse-filter NAME] [--hasse-highlight NAME] [--hasse-max-faces N]]
           [--db PATH | --no-db] [--verbose]
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
`--torus-count`, which can take minutes, and the degenerate faces of `-D`, which need the Landau
analysis. Pass one or more flags to choose.

```bash
fk analyse "12e|2e|e|:zzz"              # every section of the massless triangle
fk analyse "12e|2e|e|:zzz" -g -n        # GKZ system and Newton polytope only
fk analyse "111e|e|:zzz"                # massless banana with three propagators
fk analyse "12e|2e|e|"                  # bare topology: every propagator massless
fk analyse "12e|2e|e|:nzz" -S           # symmetries of the one-mass triangle
fk analyse "11e|e|:nn" -r --d0 3        # resonant facets of the massive bubble, D = 3 - 2 eps
fk analyse "12e|22e|e|:nnnn" -f         # the graphs of the faces of the massive parachute
fk analyse "111e|e|:nzz" -L             # resonance centres of the sunrise with one mass
fk analyse "11e|e|:nn" --torus-count    # candidate master count of the massive bubble
fk analyse "111e|e|:nnn" -D             # degenerate faces of the massive sunrise
```

| Flag | Long form | Section |
|------|-----------|---------|
| `-s` | `--symanzik` | Symanzik polynomials U, F, G |
| `-p` | `--params` | Schwinger, Feynman and Lee-Pomeransky parametrisations |
| `-g` | `--gkz` | GKZ A-matrix and Euler equations |
| `-t` | `--toric` | Toric ideal of the A-matrix |
| `-n` | `--newton` | Newton polytope: vertices, whether the integral is scaleless, normalised volume (the holonomic rank for generic $\beta$), Smith invariants, lattice invariants and whether $\mathbb{N}A$ is normal |
| `-r` | `--resonance` | For each facet of the Newton polytope, its inequality, $l_F(\beta)$, where it is resonant and admissible as $D = D_0 - 2\varepsilon$ varies, and whether it makes the GKZ system reducible (see [Resonant and admissible facets](#resonant-and-admissible-facets)) |
| `-f` | `--faces` | For each facet of the Newton polytope, its inequality and its graph, a product of Symanzik polynomials of minors, then the number of faces of each class up to codimension 2 (see [Graphs of the faces](#graphs-of-the-faces)). It also adds the `faces` section to a report whose `--sections` leave it out |
| `-L` | `--face-lattice` | The faces of each dimension by where they are resonant, the resonance centres at generic $\varepsilon$ and at $\varepsilon = 0$, and whether the GKZ system is reducible there (see [Resonance of every face](#resonance-of-every-face)). It also compares the Cayley configuration of the Schwinger representation with the Lee-Pomeransky one face by face, and adds the `face_lattice` section, with that comparison, to the reports |
| `-S` | `--symmetries` | Polytope automorphisms and symmetry pairs |
| | `--torus-count` | Candidate Euler characteristic from finite-field point counts; left out when no flag is given |
| `-D` | `--degeneracy` | The faces of the Newton polytope on which $G$ has a singular point in the torus, decided exactly at the kinematic point of `--torus-count` (see [Degenerate faces](#degenerate-faces)); left out when no flag is given. It needs Singular, and adds the `degeneracy` section to a report whose `--sections` leave it out |

`--d0 VALUE` sets $D_0$, an integer or a fraction such as `7/2`, for `-r`, `-L` and the report's
`resonance` and `face_lattice` sections; it is 4 by default, since `fk` builds $D$ as a symbol. The
powers are the integral's when they are integers, and 1 on every edge otherwise, which is what `fk`
builds. `--d0` needs `-r`, `-L`, no section flag at all, or a report.

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

#### Setting invariants

`--set SYMBOL=VALUE` sets one kinematic symbol of the integral, after `--kinematics`. The option
repeats, and each use applies to the integral the earlier ones left. The value is 0, a rational
number such as `-3/2`, another symbol, or a linear combination of them with `+`, `-`, `*` by a
number, `/` by a number and brackets, read exactly and never as a float. The symbols are those of
the momentum products, such as `p1^2`, `s12` and `s23` of the box. The three-mass box is the box
with $p_4^2 = 0$:

```bash
fk analyse "12e|3e|3e|e|:zzzz" --set 'p4^2=0' --json --no-db
fk analyse "12e|3e|3e|e|:zzzz" --kinematics massless_on_shell --set 's12=s23' -n
```

A symbol the integral does not have, including one that an earlier `--set` or `--kinematics`
removed, a value that is not linear (a product or quotient of symbols, or a power), a decimal or
function that cannot be read exactly, and a value that contains the symbol it sets, stop with one
line on stderr and status 1. A `--set` without `=` or with an empty side is a usage error (status
2). The header of `fk analyse` shows a `Substitutions` row, the LaTeX report states them in its
date, the text report in a line under its title, and the JSON lists them under `substitutions`
when there are any. The kinematic class of the result is usually `other`.

#### Hasse diagrams

`--hasse FILE` writes the Hasse diagram of the face lattice as a standalone LaTeX document, at the
$D_0$ and powers of the face-lattice section (see [Hasse diagrams of the face
lattice](#hasse-diagrams-of-the-face-lattice)). Without a section flag the usual printout comes
too, as with `--latex`.

| Option | Effect |
|--------|--------|
| `--hasse FILE` | write the diagram to FILE |
| `--hasse-view VIEW` | the faces drawn: `all`, `codim` (the default), `upset` or `downset` (of `--hasse-face`), or `filter` (those of `--hasse-filter`) |
| `--hasse-codim K` | largest codimension of the `codim` view; 2 by default |
| `--hasse-face POINTS` | the face of `upset` and `downset`, as comma-separated point indices |
| `--hasse-filter NAME` | the faces of the `filter` view: `resonant`, `degenerate`, `contraction` or `ir` |
| `--hasse-highlight NAME` | draw the faces of that filter with a heavy outline |
| `--hasse-max-faces N` | the most faces to draw; 200 by default |

These need `--hasse`. A view with more faces than the limit stops with status 1 and a message that
gives its size and the codimension views that fit.

```bash
fk analyse "12e|3e|3e|e|:nnnn" --hasse box.tex --hasse-highlight contraction --no-db
```

#### Point counts

`--torus-count` counts the points of $G = 0$ in the torus over finite fields $\mathbb{F}_p$ at one
kinematic point and fits a polynomial in $p$ (see [Torus point counts](#torus-point-counts)). It
prints the point, the excluded primes and the counts, then either the candidate polynomial, Euler
characteristic and master count or the reason the counts give no candidate. These are candidates,
not proofs.

| Option | Effect |
|--------|--------|
| `--seed N` | seed for the kinematic point, 0 or more; 0 by default; `-D` and the `degeneracy` section use the same point |
| `--torus-budget N` | maximum evaluations of $G$, 1 or more; $2 \times 10^9$ by default, enough for six propagators unless many small primes are left out, and never for seven |
| `--degeneracy-mode MODE` | where `-D` decides the faces: `point`, the default, at the point of `--seed`, or `generic`, over the field of rational functions in the kinematic symbols, which can take minutes; needs `-D` or the `degeneracy` section |

Both need `--torus-count`, or `torus` among the report sections of `--sections`; `--seed` also
works with `-D` or the `degeneracy` section. `--json` does not
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
and `to_text` (section 22), or summarises it as JSON. It builds the report once, however many of
the options are given, and checks that it can write each file before the analysis starts.

| Option | Effect |
|--------|--------|
| `--latex FILE` | write the report as a LaTeX document |
| `--text FILE` | write the report as plain text |
| `--sections NAMES` | comma-separated report sections from `identity`, `conventions`, `polynomials`, `representations`, `polytope`, `torus`, `gkz`, `resonance`, `faces`, `face_lattice`, `hasse`, `symmetries`, `landau`, `degeneracy` and `schwinger`; all but `torus`, `degeneracy` and `hasse` by default |
| `--limits` | look for limit surfaces in the Landau section, which analyses the parent family as well (see [Specialised kinematics](#specialised-kinematics)) |
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

With the `degeneracy` section the summary has `degenerate_faces`, the number of degenerate faces.
When Singular leaves a face undecided it is `null` if no face is degenerate, and a string such as
`"at least 2"` otherwise.

With the `landau` section and a kinematic class that specialises the legs, as `massless_on_shell`
does and `--limits` is given, the summary also has `limit_surfaces`, `limit_candidates` and
`parent_skipped_faces` (see
[Specialised kinematics](#specialised-kinematics)), and the report lists them apart from the Landau
surfaces.

`--json` still writes the files of `--latex` and `--text` and stores the integral in the
database, but does not report either on stdout. It cannot be combined with section flags, and
`--sections` and `--limits` need `--latex`, `--text` or `--json`.

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
   `point_config` and `finite_index` in those of all the points, and the identity of a
   `point_config` or `finite_index` map holds only trivially, since for generic $\beta$ the GKZ
   systems have no non-zero solutions;
3. prints each map it finds. A `point_config` or `finite_index` map sends every column of one
   A-matrix to a column of the other, and for such a map it also prints the column permutation
   $P$, the substitution $u_i = \prod_k v_k^{M_{ki}}$ and the identity
   $I_A(\beta, z_P) = |\det M|\, I_B(T\beta, z)$ between the two integrals without Gamma
   prefactors. A `unimodular` or `affine_polytope` map relates only the hull vertices and gives
   no identity. A `finite_index` map is an integer matrix $M$ with $\det M \neq 0$ and an integer
   translation that together send the columns of one A-matrix bijectively onto those of the
   other.

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
  Lattice points               6  (0 interior)
  h*-vector                    (1, 2, 1, 0)
  Gorenstein index             2
  Lattice width                1
  Integer decomposition (IDP)  yes
  Normal configuration         yes
  NA is normal, so C[NA] is Cohen-Macaulay (Hochster 1972)
  and there are no rank jumps (Matusevich, Miller and Walther 2005).
====================================================================
  Done in 0.3s
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
| `one_vertex_irreducible` | `False` | Drop graphs whose propagators split into two sets sharing one vertex, whose integrals factorise (the polynomial $\mathcal G$ itself need not factor) |

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
fi.kinematic_axes       # (internal, external), e.g. ('zero', 'on_shell')
fi.kinematic_class      # e.g. 'massless_on_shell'; see Kinematic classes below
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
fi.facet_resonance(nu={1: 1, 2: 1, 3: 1})  # FacetResonance per facet, D = 4 - 2 eps (section 11)
fi.face_identification()                   # FaceIdentification per face up to codim 2 (section 11)
AnalysisReport.from_integral(fi)           # every fact the analysis report states (section 22)
fi.to_latex()                              # the report as a LaTeX document
fi.to_text()                               # the report as plain text
```

### Kinematic classes

`fi.kinematic_class` names the kinematics of an integral. It is read from two axes, which
`fi.kinematic_axes` returns. The internal axis describes the masses $m_e$ of the propagators:
`zero` when every one is 0, `equal` when all are the same symbol, `generic` when the nonzero ones
are distinct symbols, and `other` otherwise, as for a number, an expression or a mass that some
propagators share but not all. The external axis describes the momentum products: `off_shell`
when no relation among the invariants is imposed, `on_shell` when every $p_i^2 = 0$ and nothing
else is, `equal` when every $p_i^2$ is the same nonzero expression and nothing else is, and
`other` otherwise, as with kinematic constraints, a vanishing $s_{12}$ or products that are not
linear in their symbols with rational coefficients. With fewer than two legs the external axis is
`off_shell`.

| Internal axis | External axis | Class |
|---------------|---------------|-------|
| `generic` | `off_shell` | `generic` |
| `zero` | `off_shell` | `massless_off_shell` |
| `zero` | `on_shell` | `massless_on_shell` |
| `equal` | `off_shell` | `equal_masses` |

Every other pair is `other`, massive propagators with on-shell legs for instance. The four classes
are the kinematics of the entries `generic_generic`, `zero_generic`, `zero_zero` and
`equal_generic` of the principal Landau determinant database of Fevola, Mizera and Telen, and the
axes take its names, external `on_shell` and `off_shell` standing for its `zero` and `generic`.

The class is derived from the integral as it stands, so it follows every change made with
`with_`. `with_kinematics`, and the `kinematics` keyword of `from_cnickel` and `from_nickel`,
impose a class by substitution:

```python
box = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
box.kinematic_class                      # 'massless_off_shell'
on_shell = box.with_kinematics("massless_on_shell")
on_shell.kinematic_axes                  # ('zero', 'on_shell')
len(on_shell.newton_polytope.points)     # 6, against 10 off shell

equal = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn", kinematics="equal_masses")
equal.cnickel                            # '12e|2e|e|:aaa'
```

- `massless_on_shell` sets each $p_i^2$ to 0 in every momentum product. It needs massless
  propagators, two or more legs, and $p_i^2$ that are symbols, as with `use_mandelstam=True`.
- `equal_masses` gives every propagator the mass $m_a$ of the shared code `a` (see
  [Format](#format)), so `:nnn` and `:aab` both become `:aaa`. It needs massive propagators.
- `generic` and `massless_off_shell` make no substitution.

A class never changes which propagators are massless. An integral that already has the class is
returned as it is. A class the integral cannot take, `other` and unknown names raise
`ValidationError`, and so does a result without the class, as when the integral has kinematic
constraints. The functions `kinematic_axes`, `kinematic_class` and `impose_kinematics` of
`feynkit.kinematics` do the same for any integral.

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

reduced = sys_.restrict_to_f_block()   # the paper's eq. 4.7: an ordinary GKZSystem
print(reduced.a_matrix, reduced.beta_parameters)
```

The parameter convention is the one used for `fi.gkz`; section 4.6 of the mathematics
reference gives the derivation, the relation to the Lee-Pomeransky system and the limits of the
reduction. In short: Britto, Grimm and Hoefnagels (arXiv:2606.09978) show that the reduced
system's solutions solve the full one when the parameter vector lies in the span of the face's
columns, which, when the block is a facet, means that the exponent of $\tilde U$ vanishes; away
from that span, and off cut contours, the relation between the reduced and full systems is not
established.
`restrict_to_f_block` warns, with a `UserWarning`, when the parameter vector does not lie in that
span; with $D$ a symbol that is unless $\nu - (L+1)D/2$ vanishes identically, so the call above
warns. `lp_to_cayley(N, L)` in `feynkit.systems.cayley` gives the unimodular matrix $T$ with
$T A_{\text{LP}} = A_{\text{Cayley}}$ up to column order and $T\beta_{\text{LP}} = \beta_{\text{Cayley}}$.

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
(section 22) lists one such expression per facet:

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
facets) or `"normaliz"` (PyNormaliz, installed with `pip install "feynkit[backends]"`; requesting it
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

### Resonant and admissible facets

A face $F$ of the A-matrix, a set of its columns $A_F$, is resonant for $\beta$ when $\beta$ lies
in $\operatorname{span}_{\mathbb{C}} A_F + \mathbb{Z}A$ (Britto, Grimm and Hoefnagels,
arXiv:2606.09978, Eq. 23, p. 12). feynkit calls it admissible when $\beta$ lies in
$\operatorname{span}_{\mathbb{C}} A_F$ itself; the paper notes that the face system $(A_F, \beta)$ is
then a true subsystem, its solutions solving the full system (pp. 10-11). For a facet the lattice form decides both: the facet is resonant
exactly when $l_F(\beta) \in \mathbb{Z}$ and admissible exactly when $l_F(\beta) = 0$ (section 4.8
of the mathematics reference). With integer powers and $D = D_0 - 2\varepsilon$, `classify_facets`
says at which $\varepsilon$ this happens:

```python
from feynkit import FeynmanIntegral, classify_facets, polytope_data

fi = FeynmanIntegral.from_cnickel("11e|e|:nn")        # massive bubble
data = polytope_data(fi.newton_polytope.points)
for r in classify_facets(data, [1, 1]):               # nu = (1, 1), D_0 = 4
    print(r.facet.normal, r.facet.offset, r.kind, r.resonant.offset, r.resonant.period)
# (-1, -1) -1 progression 0 1       F_F: resonant for eps in Z, admissible at eps = 0
# (0, -1) 0 all None None           the edge facet of u_2
# (-1, 0) 0 all None None           the edge facet of u_1
# (1, 1) 2 progression 1 1/2        F_U: eps in 1 + Z/2, admissible at eps = 1
print([r.resonant_at_zero for r in classify_facets(data, [1, 1], d0=3)])
# [False, True, True, True]: at D_0 = 3, F_F (b = -1) is not resonant at eps = 0
print(*classify_facets(data, [1, 1])[3].resonant.window(-1, 1))
# -1 -1/2 0 1/2 1
```

These are the four facets Britto, Grimm and Hoefnagels give for the bubble (Eqs. 67-68, p. 22):
the edge facets are resonant for every $D$, $F_{\mathcal F}$ when $D/2$ is an integer and
$F_{\mathcal U}$ when $D$ is. Each `FacetResonance` holds:

| Attribute | Description |
|-----------|-------------|
| `facet` | The `Facet`; its `point_indices` are the columns on it |
| `functional` | $l_F$, zero on the facet, positive off it, mapping $\mathbb{Z}A$ onto $\mathbb{Z}$; `facet.lattice_form` for `classify_facets` |
| `form` | $l_F(\beta)$ as a `ParameterForm` in $D$ and the $\nu_e$; `form.expression(D, nus)` gives it in SymPy |
| `resonant` | The $\varepsilon$ at which the facet is resonant, an `EpsilonSet`: `"all"`, `"progression"` (`offset` + `period` $\mathbb{Z}$), `"point"` or `"never"` |
| `admissible` | The $\varepsilon$ at which it is admissible: `"all"`, `"point"` or `"never"`; for a progression, the point at its offset $\varepsilon_F$ |
| `columns_off` | The number of columns off the facet; `pyramid` is whether it is 1 |
| `reducible` | True when the GKZ system is reducible wherever the facet is resonant (Schulze and Walther, arXiv:1009.3569, Theorem 4.1), None when the facet does not decide it |

`resonant_at_zero`, `kind` and `lattice_index` are shorthands. A facet with $b = 0$ is resonant for
every $\varepsilon$ or for none, and one with $b \ne 0$ on the progression
$\varepsilon_F + (g_F/|b|)\mathbb{Z}$, $\varepsilon_F = D_0/2 - m \cdot \nu / b$. The "never" class
needs $g_F > 1$: on $(0,0), (2,0), (0,1)$ with $\nu = (1, 0)$ the facet $x \ge 0$ is never resonant.
`reducible` is True only for a resonant facet of a full-dimensional polytope with at least two
columns off it; it is never False, since irreducibility is a question about faces of every
dimension, which [Resonance of every face](#resonance-of-every-face) answers. Below full dimension
$\beta$ must first lie in the span of $A$, and
`span_epsilons(data, nu, d0)` says for which $\varepsilon$ it does: for every value, for one, or for
none. The facets' sets are then intersected with it.

`classify_configuration(a_matrix, beta, nu, d0)` does the same for any homogeneous configuration,
computing the facets from the columns, with $\beta$ one `ParameterForm` per row
(`lee_pomeransky_beta(n)` is the Lee-Pomeransky one), and `admissible(a_matrix, face, beta)` tests
any set of columns, with numbers or SymPy expressions in $\beta$:

```python
from feynkit.resonance import admissible

a, beta = fi.gkz.a_matrix, fi.gkz.beta_parameters   # beta = (-D/2, -nu_1, -nu_2), symbolic
print(admissible(a, range(a.cols), beta))            # True: every column
print(admissible(a, [0], beta))                      # False: not for all D and nu
```

`fi.facet_resonance(d0=None, nu=None, system="gkz")` classifies the facets of `fi.gkz`, with point
indices into its columns, the $z_j$. `d0=None` reads $D_0$ from the dimension of the integral when
that is $D_0 - 2\varepsilon$, with $D_0$ a number and $\varepsilon$ the symbol named `epsilon`, as
for `fi.with_(dimension=6 - 2 * sp.Symbol("epsilon"))`, and takes 4 otherwise;
`feynkit.resonance.choose_d0` makes that choice. The powers are the integral's, which must then be integers, or
`nu`, a mapping from edge index to integer. With `system="schwinger"` it classifies the facets of
`fi.schwinger_gkz`, computed from the Cayley columns (the $w$ then the $z$), with
$\beta_{\text{Cayley}} = T\beta_{\text{LP}}$ written in $D$ and $\nu$; since $T$ is unimodular,
corresponding facets get the same classification. The $\tilde F$ block is $F_{\mathcal U}$:

```python
import sympy as sp

fi = FeynmanIntegral.from_cnickel("111e|e|:nnn")      # sunrise, three masses
unit = dict.fromkeys(fi.propagator_exponents, 1)
f_u = [r for r in fi.facet_resonance(nu=unit) if r.facet.offset == 3][0]
print(f_u.form.expression(sp.Symbol("D"), sp.symbols("nu_1:4")))   # -3*D/2 + nu_1 + nu_2 + nu_3
print(f_u.resonant.offset, f_u.resonant.period)       # 1 1/3: resonant for eps in 1 + Z/3
print(len(fi.facet_resonance(nu=unit, system="schwinger")))   # 8, as for fi.gkz
```

At $\varepsilon = 1$, $D = 2$, where $\nu = (L+1)D/2$, `restrict_to_f_block` is a true subsystem
and does not warn.

The analysis report has a `resonance` section built from the same data, and `fk analyse -r` prints
it on the terminal, both with `--d0` for $D_0$.

### Graphs of the faces

For a face $F$ of the Newton polytope $P$ of $\mathcal G$, the restriction $\mathcal G|_F$ keeps the
terms of $\mathcal G$ whose exponents lie on $F$. It is the initial form of $\mathcal G$ for any
weight $w$ that $F$ minimises (Fevola, Mizera and Telen, arXiv:2311.16219, p. 28), and it is often
a product of Symanzik polynomials of minors of the graph. For the weight that is 1 on the edges of
a connected subgraph $\gamma$ and 0 elsewhere, with every mass non-zero, it is
$\mathcal U_\gamma\,\mathcal G_{\Gamma/\gamma}$ (their Eqs. 3.13 and 3.15, pp. 27-28), and the face of
the terms free of $u_e$ gives $\mathcal G_{\Gamma/e}$ (Britto, Grimm and Hoefnagels,
arXiv:2606.09978, Eq. 21, p. 11, for a one-particle-irreducible graph). `identify_faces` names every face in this way:

```python
from feynkit import FeynmanIntegral, identify_faces

fi = FeynmanIntegral.from_cnickel("12e|22e|e|:nnnn")    # the parachute with four masses
for face in identify_faces(fi):
    if face.facet is not None:
        print(face.facet.normal, face.facet.offset, face.kind, face.name())
# (-1, -1, -1, -1) -2 u_layer U(Gamma)
# (0, 0, -1, -1) -1 product_uv U({3,4}) G(Gamma/{3,4})
# (0, -1, 0, 0) 0 contraction G(Gamma/{2})
# (-1, -1, 0, -1) -1 product_uv U({1,2,4}) G(Gamma/{1,2,4})
# (0, 0, 0, -1) 0 contraction G(Gamma/{4})
# (-1, -1, -1, 0) -1 product_uv U({1,2,3}) G(Gamma/{1,2,3})
# (0, 0, -1, 0) 0 contraction G(Gamma/{3})
# (-1, 0, 0, 0) 0 contraction G(Gamma/{1})
# (1, 1, 1, 1) 3 f_layer F(Gamma)
```

These are the nine rays Fevola, Mizera and Telen give for the parachute (p. 29), each inward
normal $-m$ a ray, with $\mathcal U_\gamma\,\mathcal G_{\Gamma/\gamma}$ on each $w_\gamma$.

The weight of a face is $w = -\sum m$ over the facets $m \cdot x \le b$ containing it, which the
face minimises. Its distinct values $t_1 > \dots > t_k$ on the edges give a flag of subgraphs
$\sigma_1 \subset \dots \subset \sigma_k$, $\sigma_j$ the edges with $w_e \ge t_j$, and the minors
$H_j = \sigma_j/\sigma_{j-1}$: the edges of weight $t_j$, with those of greater weight contracted
and the others deleted. The terms of $\mathcal U$ of least $w$-degree are
$\prod_j \mathcal U_{H_j}$ (section 5.7 of the mathematics reference). The prediction for
$\mathcal G|_F$ is that product with the factor of the last level whose $\mathcal F_{H_j}$ is not
zero replaced by $\mathcal G_{H_j}$ when $t_j = 0$ and by $\mathcal F_{H_j}$ when $t_j < 0$. A face
is identified only when $\mathcal G|_F$ equals the prediction exactly. Otherwise its class is
`support_product` when the exponents of $\mathcal G|_F$ are exactly the sums of one exponent of
each factor, and `unidentified` when they are not; both polynomials are kept. `name()` writes each factor as `U`, `F` or `G` of
its minor, `Gamma` for the whole graph, `{1,2}` for the subgraph on edges 1 and 2 and
`{1,2,3}/{1}` for it with edge 1 contracted, and leaves out the factors equal to 1.

| Class | $\mathcal G\vert_F$ |
|-------|--------------------|
| `whole` | $\mathcal G_\Gamma$, the polytope itself |
| `contraction` | $\mathcal G_{\Gamma/S}$ alone |
| `product_uv` | a product whose $\mathcal G$ factor is on a quotient $\Gamma/S$, as $\mathcal U_\gamma\,\mathcal G_{\Gamma/\gamma}$ |
| `product_ir` | a product whose $\mathcal G$ factor is on a minor with edges deleted, as $\mathcal G_\gamma\,\mathcal U_{\Gamma/\gamma}$ |
| `u_layer` | a product of $\mathcal U$'s alone, a face of $\mathrm{Newt}(\mathcal U)$ |
| `f_layer` | a product with one $\mathcal F$ factor, a face of $\mathrm{Newt}(\mathcal F)$ |
| `support_product` | not the prediction, but with exactly its exponents |
| `unidentified` | not the prediction, and with other exponents |

A support product is a face whose point set is the product of the supports of the flag's factors,
the sums of one exponent from each; since the factors are in disjoint variables, these are the
exponents of the predicted product, each once. Its Newton polytope is then the product of the
factors' Newton polytopes, although $\mathcal G|_F$ is not the product of the factors. The point
sets are compared exactly, and `support_verified` records the outcome: true for every identified
face and every support product.

On the bubble with $m_2 = 0$, where the edge face $F_1$ is no longer a facet, Britto, Grimm and
Hoefnagels find the new facet $F_{2,(1,2)}$ (Eq. 105, p. 29). It is $x_2 \le 1$, of the kind
`product_ir`, and
$\mathcal G|_F = \mathcal G_{\{1\}}\,\mathcal U_{\Gamma/\{1\}} = u_2(1 + (m_1^2 - s)u_1/\mu^2)$:

```python
bubble = FeynmanIntegral.from_cnickel("11e|e|:nz")
(face,) = [f for f in identify_faces(bubble) if f.kind == "product_ir"]
print(face.facet.normal, face.facet.offset, face.name())   # (0, 1) 1 G({1}) U(Gamma/{1})
print(face.polynomial)                    # u_1*u_2*(m_1**2/mu**2 - s/mu**2) + u_2
print(face.levels[0])
# FlagLevel(edges=(1,), contracted=(), deleted=(2,), weight=0, kind='G', loops=0)
```

At generic kinematics, Arkani-Hamed, Hillman and Mizera label the facets of the Feynman polytope
by subgraphs $\gamma$, ultraviolet when $\mathcal F_{\Gamma/\gamma} \ne 0$ and infrared when it is
zero (arXiv:2202.12296, Eqs. 7-8, p. 3); there $\mathrm{Newt}(\mathcal F)$ is the polytope their
subgraph inequalities cut out (Borinsky, Munch and Tellander, arXiv:2302.08955, Theorem 3.6,
p. 13). At exceptional kinematics it need not be: on Borinsky, Munch and Tellander's QED
triangle, $\mathrm{Newt}(\mathcal F)$ is a segment while the inequalities cut out a polygon
(Fig. 3, p. 15). The massive sunrise has, besides the two layers, six facets,
three of them the ultraviolet bubbles $\mathcal U_{\{i,j\}}\,\mathcal G_{\Gamma/\{i,j\}}$, as in
their App. B (Eq. B14, p. 9). On their three-mass box the facet of $\gamma_{14}$ (App. B, p. 10)
carries the restricted $\mathcal U$ and $\mathcal F$ they write down. Its flag names $\gamma_{14}$ and
predicts $\mathcal G_{\gamma_{14}}\,\mathcal U_{\Gamma/\gamma_{14}}$. The $\mathcal F$ part has
exactly the exponents of that product, but its coefficients form a matrix of rank 2, so
$\mathcal G|_F$ is irreducible, and the face is a `support_product`. That box has four support
products in all, the facet and three faces of codimension 2; the massless double box with
$p_i^2 = 0$ has 22, among 932 faces that are not identified.

Each `FaceIdentification` holds:

| Attribute | Description |
|-----------|-------------|
| `point_indices`, `dimension`, `codimension` | The face, as indices into `fi.newton_polytope.points` |
| `facet` | The `Facet`, when the face is one |
| `weight` | $w$, in internal-edge order |
| `levels` | The flag, one `FlagLevel` per value of $w$, from the largest: `edges`, `contracted`, `deleted`, `weight` $t_j$, `kind` (`"U"`, `"F"` or `"G"`) and `loops` of $H_j$ |
| `kind`, `contracted` | The class, and $S$ for a contraction |
| `verified` | Whether $\mathcal G\vert_F$ equals the prediction |
| `support_verified` | Whether the exponents of $\mathcal G\vert_F$ are exactly the sums of one exponent of each factor |
| `polynomial`, `prediction` | $\mathcal G\vert_F$ and the product of the level polynomials, in the $u_e$ |
| `reason` | Why the face is not identified as a product; None when it is |

`identify_faces(fi, max_codimension=2)` takes the polytope, its facets and the faces of codimension
2; `max_codimension=None` takes every face, vertices included. The massless double box has 154
faces up to codimension 2, identified in about 0.3 s, and 2245 in all, in about 1.5 s.
`fi.face_identification(max_codimension=2)` caches the result for each argument. The faces of a
Newton polytope that is not full-dimensional, as that of a scaleless integral, are not identified:
a relative facet's normal is fixed only modulo the equations of the affine hull, and so is the
flag. They come back as `unidentified` with that reason.

The minors come from `Graph.contract` and `Graph.delete`, which keep the edge indices, move the legs
with their vertices and remove a vertex left without propagators, with its legs.
`minor_polynomials` in `feynkit.polynomials` gives their $\mathcal U$ and $\mathcal F$ in the
parameters and kinematics of the graph, a minor that is not connected taking the product of the
$\mathcal U_c$ of its components and $\sum_c \mathcal F_c \prod_{c' \ne c} \mathcal U_{c'}$:

```python
from feynkit.polynomials import minor_polynomials

print(fi.graph.contract([3, 4]))         # the bubble on edges 1 and 2
u, f = minor_polynomials(fi.graph, fi.momentum_products, contract=[3, 4])
print(u)                                 # a_1 + a_2
```

The analysis report has a `faces` section, built by default, and `fk analyse` prints the facets and
the counts on the terminal, as `-f` does alone.

### Resonance of every face

The facets alone decide resonance only for themselves, and reducibility only one way: a resonant
facet with two columns off it makes the system reducible, but irreducibility depends on the
resonance centres, faces of any dimension, the empty face included (Schulze and Walther,
arXiv:1009.3569, Def. 3.2, p. 5). `fi.face_lattice()` decorates every face of the Newton polytope:

```python
from feynkit import FeynmanIntegral

fi = FeynmanIntegral.from_cnickel("111e|e|:nnz")      # sunrise, m_3 = 0
lattice = fi.face_lattice(nu={1: 1, 2: 1, 3: 1})       # D = 4 - 2 eps
print(len(lattice.faces), len(lattice.facets))          # 26 6, the empty face included
points = fi.newton_polytope.points
f1 = lattice.face(j for j, p in enumerate(points) if p[0] == 0)
print(f1.codimension, f1.resonant)
# 2 EpsilonSet(kind='progression', offset=Fraction(0, 1), period=Fraction(1, 1))
print([face.identification.name() for face in lattice.centres("generic")])   # ['G(Gamma/{3})']
print(lattice.reducible("generic"), [face.dimension for face in lattice.centres(0)])
# True [-1]: at eps = 0 the empty face is the centre
print(lattice.check_schwinger().agrees)                 # True
```

The edge face $F_1$, the terms free of $u_1$, is no longer a facet and is resonant only for
integer $\varepsilon$; Britto, Grimm and Hoefnagels find it not resonant for generic $D$ (p. 43).
With $N_G$ the integer forms on $\mathbb{Z}A$ that vanish on the columns of a face $G$, $G$ is
resonant exactly when $\beta$ lies in the span of $A$ and a basis of $N_G$ takes integer values on
it, and admissible exactly when every form of the basis vanishes on it. The forms of the facets
containing $G$ span a subgroup of $N_G$ of finite index $|T_G|$, the lattice defect, and $G$ is
resonant wherever all its facets are, for every $\beta$, exactly when $|T_G| = 1$ (section 4.9 of
the mathematics reference). On every Feynman graph in feynkit's tests $|T_G| = 1$. On Schulze and
Walther's quadric cone it is 2 at the empty face (Ex. 3.3, p. 6), and
`decorate_configuration(a_matrix, beta0, beta1=None)` shows it for any homogeneous configuration
and $\beta_0 + \varepsilon\beta_1$:

```python
from fractions import Fraction
from feynkit.face_lattice import decorate_configuration

cone = decorate_configuration([[1, 1, 1], [0, 1, 2]], [Fraction(1, 2), 1])
for face in cone.faces:
    print(face.point_indices, face.resonant.kind, face.lattice_defect, face.pyramid)
# () never 2 False
# (0,) all 1 False
# (2,) all 1 False
# (0, 1, 2) all 1 True
print([face.point_indices for face in cone.centres(0)], cone.reducible(0))
# [(0,), (2,)] True: two centres, A a pyramid over neither
```

`centres(eps)` gives the minimal resonant faces at `eps`, an integer, a Fraction or `"generic"`,
where exactly the faces resonant for every $\varepsilon$ are resonant. `reducible(eps)` is True when
$A$ is a pyramid over no centre, so that the monodromy is reducible (Theorem 4.1), and False when it
is a pyramid over one, so that the monodromy is irreducible (Theorem 5.1); that centre is then the
only one (Prop. 3.8). It is None when $A$ does not have full rank, as below full
dimension, and where no face is resonant, off the span of $A$. On the massless bubble $A$ is a
pyramid over every face, so `fi.face_lattice(nu={1: 1, 2: 1}).reducible(0)` is False: at $D = 4$
every face is resonant, yet the system has irreducible monodromy (its D-module is reducible there,
of rank 1), which the facets leave open.

Each `DecoratedFace` holds:

| Attribute | Description |
|-----------|-------------|
| `point_indices`, `dimension`, `codimension` | The face, as indices into `fi.newton_polytope.points`; dimension $-1$ for the empty face |
| `facets` | The facets containing it, as positions in `lattice.facets`, the order of `classify_facets` |
| `resonant`, `admissible` | `EpsilonSet`s, as for the facets; a progression's offset is the admissible value when there is one |
| `lattice_defect` | $\vert T_G\vert$ |
| `columns_off`, `pyramid` | The columns off the face, and whether $A$ is a pyramid over it (Def. 3.4) |
| `through_origin` | Whether the origin lies in the affine hull of its points; for a facet, $b = 0$ |
| `identification` | Its `FaceIdentification` up to codimension `identify_codimension`, 2 by default; None beyond |

The lattice also holds `facet_resonance`, the facets as `classify_facets` gives them, and
`gkz_columns` and `cayley_columns`, the column of `fi.gkz.a_matrix` and of
`fi.schwinger_gkz.a_matrix` that each point gives, since `fi.gkz` orders its columns differently.
`check_schwinger()` decorates the Cayley configuration from its own columns and compares it with
this one face by face under $T$. `fi.face_lattice(d0=None, *, nu=None, identify_codimension=2)`
chooses $D_0$ and the powers as `fi.facet_resonance` does and caches the result;
`feynkit.decorate_faces` is the same without the cache. The massless double box has 2246 faces,
decorated in about 0.8 s.

The analysis report has a `face_lattice` section, built by default, and `fk analyse -L` prints it on
the terminal and adds the comparison with the Cayley side to it.

### Hasse diagrams of the face lattice

`lattice.covers(face)` gives the faces one dimension up that contain `face`, and
`lattice.lower_covers(face)` those one down; each takes a `DecoratedFace` or its point indices.
These are the edges of the Hasse diagram of the lattice, in the order of `lattice.faces`.
`feynkit.visualisation.hasse` draws it:

```python
from feynkit import FeynmanIntegral
from feynkit.visualisation.hasse import hasse_document, infrared_facets, soft_collinear_cones

box = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz")
symbols = {s.name: s for v in box.momentum_products.values() for s in v.free_symbols}
corners = {symbols[f"p{j}^2"]: 0 for j in (1, 2, 3)}      # three massless corners
box = box.with_(momentum_products={k: v.subs(corners) for k, v in box.momentum_products.items()})
lattice = box.face_lattice(nu={1: 1, 2: 1, 3: 1, 4: 1})
print(len(infrared_facets(lattice)), len(soft_collinear_cones(lattice)))   # 3 2
open("box.tex", "w").write(hasse_document(lattice, "codim", 2, highlight="ir"))
```

`lualatex box.tex` compiles the file. `hasse_tikz` returns the bare `tikzpicture`, and
`save_hasse_tikz` writes either. Each face is a node, one row to a dimension with the empty face at
the bottom, in the order of the lattice, so that a second run gives the same file byte for byte.
The label is the name of the face's identification, such as $G(\Gamma/\{4\})$, with a question mark
when the face is not verified, and otherwise its points. The colour is the face's resonance at the
lattice's $D_0$ and powers: red for every $\varepsilon$, yellow for a progression, green for none. With
`eps=0` the faces resonant at $\varepsilon = 0$ are in bold. A ring marks a degenerate face and a
dashed outline one that the analysis left undecided; they appear when the lattice carries verdicts,
from `fi.face_lattice(..., degeneracy=True)`, or when `degeneracy=True` is passed with an integral,
which decorates it.

| Argument | Description |
|----------|-------------|
| `lattice` | a `DecoratedFaceLattice`, or an integral, which is decorated at `d0` and `nu` |
| `view`, `k` | `"all"`; `"codim"`, the faces of codimension at most `k`, 2 by default; `"upset"` and `"downset"`, the faces containing or contained in `face`; `"filter"`, the faces of `filter` |
| `filter` | `"resonant"` (resonant for every $\varepsilon$), `"degenerate"`, `"contraction"` or `"ir"` |
| `highlight` | a filter whose faces get a heavy outline |
| `eps`, `degeneracy` | the bold faces, and whether to mark degeneracy; None marks it when the lattice has verdicts |
| `max_faces` | 200 by default; a larger view raises `ValidationError` with its size and the codimension views that fit |

The full lattice of the double box has 2246 faces, so `hasse_tikz(lattice, "all")` raises there,
and the default view, codimension 2, has under 200. A filtered view joins two faces when one
contains the other and no chosen face lies between them.

An infrared facet, in feynkit's term, is a facet whose flag deletes edges and has a later level
whose minor has $F = 0$ and at least two edges, as at a massless corner of the box; a level of a
single line is soft only, and does not count. $G$ on the facet need not be the product
$G(\gamma)\,U(\Gamma/\gamma)$ of the flag, and at the massless box it is not, so these facets
are reported as unidentified. A soft-collinear cone is a face of
codimension 2 on two infrared facets and on neither layer of the polytope. The massless box with two
neighbouring massless corners has 1 cone, with three has 2, and with four has 4, one for each pair of
neighbouring corners; two opposite corners have none, since their facets meet only in the layer of
$F$. The report's `hasse` section, off by default, puts the diagram in the LaTeX report, drawn to a
smaller codimension if the default would have more than 200 faces. The diagram is on the page at
the width of the text, so a wide one is small.

### Lattice invariants

`fi.lattice_invariants()` gives the classical lattice invariants of the Newton polytope $P$,
computed exactly in pure Python by `feynkit.lattice_invariants` (section 5.6 of the mathematics
reference):

```python
from feynkit import FeynmanIntegral

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")    # massless triangle
inv = fi.lattice_invariants()
print(inv.lattice_points, inv.interior_points)          # 6 0
print(inv.h_star)                                        # (1, 2, 1, 0)
print(inv.gorenstein_index, inv.reflexive)               # 2 False
print(inv.lattice_width, inv.width_direction)           # 1 (1, 0, 0)
print(inv.idp, inv.support_is_saturated, inv.normal)     # True True True
```

`LatticeInvariants` fields:

| Attribute | Description |
|-----------|-------------|
| `lattice` | `"support"` or `"ambient"`, the lattice of every field but `normal` |
| `dimension` | The dimension $d$ of $P$ |
| `lattice_points`, `interior_points` | The lattice points of $P$ and of its relative interior |
| `ehrhart` | The coefficients $c_0, \ldots, c_d$ of the Ehrhart polynomial, as `Fraction`s |
| `h_star` | $h^*_0, \ldots, h^*_d$, trailing zeros included |
| `gorenstein_index`, `reflexive` | The Gorenstein index, None when $P$ is not Gorenstein, and whether it is 1 |
| `lattice_width`, `width_direction` | The lattice width and a direction that attains it, in the coordinates of `invariant_chart` |
| `idp` | Whether $P$ has the integer decomposition property |
| `support_is_saturated` | Whether the points of $G$ are all the lattice points of $P$ |
| `normal` | Whether the monoid $\mathbb{N}A$ of the GKZ configuration is normal |

$\mathbb{N}A$, generated by the columns $(1, \alpha_j)$ of the A-matrix, is normal exactly when the
monomials of $G$ are all the lattice points of $P$ and $P$ has IDP. Then
$\mathbb{C}[\mathbb{N}A]$ is Cohen-Macaulay (Hochster 1972), and when $P$ is full-dimensional there
are no rank jumps: the holonomic rank is the normalised volume for every $\beta$ (Matusevich, Miller
and Walther 2005). This certifies, for the integral at hand and at any kinematics, what Klausen's
theorem gives for a class of graphs at generic kinematics (section 4.5 of the mathematics
reference). A monoid that is not normal can still be Cohen-Macaulay, so `normal=False` settles
nothing.

By default $P$ is measured in the lattice its points generate, that of the normalised volume and
of $\mathbb{Z}A$, in which normality is defined; `lattice="ambient"` measures it in
$\mathrm{aff}(P) \cap \mathbb{Z}^N$ instead, the usual lattice of Ehrhart theory. The two agree
when `sublattice_index` is 1, as for the graphs `generate_graphs` gives with one loop and up to
six legs, or two loops and up to four, at generic kinematics. `normal` always refers to the
support lattice. The functions of `feynkit.lattice_invariants` take any points `polytope_data`
takes:

```python
from feynkit.lattice_invariants import (
    ehrhart_polynomial, h_star_vector, is_idp, lattice_invariants, lattice_points,
)

reeve = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 2)]  # a Reeve tetrahedron
print(ehrhart_polynomial(reeve, lattice="ambient"))  # Poly(1/3*k**3 + k**2 + 5/3*k + 1, k, domain='QQ')
print(h_star_vector(reeve, lattice="ambient"))       # (1, 0, 1, 0)
print(is_idp(reeve, lattice="ambient"))              # False
print(h_star_vector(reeve))                          # (1, 0, 0, 0)
print(lattice_points(reeve, 2, lattice="ambient"))   # the 11 points of 2P, (1, 1, 1) among them

# The rank-jump configuration A = [[1, 1, 1, 1], [0, 1, 3, 4]]: the support misses 2.
print(lattice_invariants([(0,), (1,), (3,), (4,)]).normal)   # False
```

In the lattice of its vertices the Reeve tetrahedron is a unimodular simplex, so its
$\mathbb{N}A$ is normal although it lacks IDP in $\mathbb{Z}^3$. `count_lattice_points` counts
instead of listing, `gorenstein_index` and `polar_dual` give the Gorenstein index and the dual of
a reflexive polytope, `lattice_width` the width, and `invariant_chart` the chart
$x = o + Sc$ in which widths and duals are written.

`is_idp` and `lattice_invariants` take `backend`: `"python"`, the reference, checks that every
lattice point of $(k+1)P$ is one of $kP$ plus one of $P$ for $k \le d - 2$ (Bruns, Gubeladze and
Trung 1997, Corollary 1.3.4, stated for the lattice generated by the lattice points of $P$; the
ambient lattice needs the full triangulation argument in the proof of their Theorem 1.3.3, p. 133); `"normaliz"` asks PyNormaliz for the Hilbert basis of the cone over $P$; and `"auto"`,
the default, uses PyNormaliz when it is installed and Python otherwise. Both are exact.
All the invariants of the kite `12e|23|3|e|:nnnnn` take 0.16 s in pure Python, those of the
massless pentagon `12e|3e|4e|4e|e|:zzzzz` 0.04 s and those of the massless hexagon
`12e|3e|4e|5e|5e|e|:zzzzzz` 0.15 s, most of it the IDP check, which lists the dilates up to
$(d - 1)P$. That grows quickly with the dimension: for the three-loop graph
`123|24|e|45|5|e|:znzzzzzz`, of dimension 8, the check takes minutes in Python and a fraction of a
second in Normaliz, so install PyNormaliz (`pip install "feynkit[backends]"`) for three loops and
more.

`lattice_invariants` and `fi.lattice_invariants` take `budget`, the most steps of pure-Python work
for the Ehrhart polynomial and the IDP check, and `timeout`, the most seconds each run of Normaliz
may take. A step is one term of an inequality the enumerator evaluates or one set lookup of the
IDP check, and CPython takes about four million a second. Over budget, PyNormaliz supplies the
Ehrhart polynomial and IDP when it is installed; otherwise `ehrhart`, `h_star`, `idp` and `normal` are
None, except that `normal` is False whenever the support misses a lattice point. Without a budget,
the default, everything is computed. The report and `fk analyse` pass `LATTICE_BUDGET`,
$2 \times 10^7$ steps or about five seconds, and `NORMALIZ_TIMEOUT`, 60 s, both in
`feynkit.io.report`, and say "not computed" for what is left out. With a timeout PyNormaliz runs in
a child process, which is ended when the time is up. The result is cached on the
integral for each choice of the arguments, so the report and the CLI compute it once.

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
faces miss, which the limit surfaces then give (see
[Specialised kinematics](#specialised-kinematics)), and it keeps those of skipped faces: at the
default `max_face_points` the faces miss 1 of the 32 factors of the massless pentagon and 8 of the
hexagon's 79. It is fast, needs no Gröbner basis, and is what the test-suite checks the face
computation against. The massless pentagon takes about half a second, the massless hexagon about
7 s and the all-massive hexagon about 80 s. With Singular on the path the minors are factored
there, in one run. Without it python-flint factors them when it is installed, which is as quick
and gives the same factors; the closed form then needs neither Singular nor a Gröbner basis. With
neither, SymPy factors them, which usually takes about as long but now and then far longer:
SymPy's factorisation draws evaluation points from a random generator the whole process shares,
and from some of its states a single minor takes minutes.

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
its locus. A face whose elimination ideal is zero has a component that projects onto all of
kinematic space; Singular's `minAssChar` splits its ideal into minimal primes, and each prime that
projects onto a hypersurface contributes that hypersurface, as $bc = ad$ does in example 3.9 of
Fevola, Mizera and Telen (2024), while the dominant ones are left out and `face.dominant` is set.
Without Singular such a face contributes nothing. The result is, by definition, the principal
Landau determinant of Fevola, Mizera and Telen (2024, definition 3.5); see
[Specialised kinematics](#specialised-kinematics) for what it can leave out.
The massless pentagon and hexagon take about 3 s and 15 s. feynkit eliminates and factors with
Singular when the `Singular` binary is on the path, except for the discriminants of edges, small
polynomials that SymPy factors to write them. Without Singular it factors with python-flint when
that is installed, else with SymPy, whose factorisation now and then takes minutes, and it
eliminates with SymPy's Gröbner basis. That elimination is far too slow beyond small faces:
the massless pentagon had not finished after 300 s without Singular. The Landau analysis needs
Singular beyond small cases; python-flint speeds up only the factorisation. Faces with more monomials than
`max_face_points` (default 14) are skipped and listed in `la.skipped_faces`. Their factors are
missing from `la.landau_surfaces` and `la.principal_a_determinant`, which are then incomplete
whatever the kinematics, and the report names each skipped face. The default covers every
one-loop box, whose polytope has at most 14 points: the all-massive box `12e|3e|3e|e|:nnnn` takes
about 6 s, against 2 s without its polytope. The massless pentagon's polytope has 15 points and is
skipped; Singular had not eliminated it after 20 minutes. Singular gets at most `timeout` seconds
for each face, `DEFAULT_FACE_TIMEOUT` of `feynkit.landau`, 60 s, by default, and a face that runs
past it is skipped and listed in `la.skipped_faces` as a face with too many points is; the report
says which faces ran past it. No face of the graphs the tests and the report exercise at the
default `max_face_points` has taken more than about 8 s, so the default changes none of their
results. Faces past the time limit are skipped: the generic massive parachute `12ee|22e|e|:nnnn`
skips a face of 14 points, which Singular had not eliminated after 400 s, and its analysis takes
about a minute where it did not finish. `timeout=None` sets no limit. A large face whose
elimination ideal is zero can take far longer to decompose than to eliminate: with
`max_face_points=30` one of 26 points of the massive kite takes 18 s to eliminate and over two
minutes to decompose, past the default. An elimination that fails or prints output feynkit cannot read raises
`ComputationError`; a factorisation that fails is left to SymPy. The limit applies to each face,
its decomposition into minimal primes included, and is not a bound on the whole analysis, whose
time can reach it for every face eliminated. The SymPy fallback, the discriminants of edges and the
factorisations are not limited, and SymPy's factorisation of a large polynomial can take minutes
on its own.

### Specialised kinematics

The principal Landau determinant is the locus the faces see. Fevola, Mizera and Telen conjecture
that it lies in the Euler discriminant, where $|\chi|$ of the complement of $\{G = 0\}$ in the
torus drops (2024, definition 3.2 and conjecture 3.6), and show that it can be strictly smaller
(example 3.10). At special kinematics this happens already at one loop: a singular point of a
face can leave the torus as the kinematics specialise, and no face then sees its limit. With
$p_1^2 = 0$ the massive triangle `12e|2e|e|:nnn` has no $p_2^2 - p_3^2$ among its Landau surfaces:
where $p_2^2 = p_3^2$ its top face has no singular point in the torus, since it has moved onto the
facet $u_3 = 0$, yet $|\chi|$ drops from 6 to 5 there.

For a `FeynmanIntegral` whose momentum products are not the generic ones, `landau_analysis(fi,
limits=True)` therefore also analyses the parent family, the same graph and masses with the momentum products it
would have by default, whose invariants the integral's products determine. It restricts the
parent's surfaces to the integral's kinematics, factors them, and keeps the irreducible factors
that do not vanish and are not Landau surfaces already. Each is tested with
`critical_point_count` (see [Torus point counts](#torus-point-counts)): the number of critical
points, which is $|\chi|$ for generic exponents, is counted at a random rational point of the
family and at two random rational points of the factor, each off every other surface found. A
factor at whose points the count drops is a limit surface, in `la.limit_surfaces`; the others are
in `la.limit_candidates`, each with the reason. The drop is evidence, not proof: the points are
random, and the counts are taken modulo two primes.

```python
import sympy as sp
from feynkit.kinematics.mandelstam import standard_invariants

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nnn")
p1, p2, p3 = standard_invariants(3).external_masses
fi = fi.with_(momentum_products={k: sp.expand(v.subs(p1, 0)) for k, v in fi.momentum_products.items()})
la = landau_analysis(fi, limits=True)
(limit,) = la.limit_surfaces
print(limit.surface, limit.generic_count, limit.counts)   # p2^2 - p3^2  6  (5, 5)
```

`landau_analysis(fi, confirm=False)` lists every factor as a candidate without counting,
`confirm_timeout` (default 60 s) limits the counts of each factor, and `seed` fixes the points and
the exponents. `limits=False`, the default, leaves the parent out. `landau_analysis_from_polynomial` has no parent
unless one is passed, as `parent=` a polynomial in the same variables and `restriction=` the map
from its kinematic symbols to expressions in the polynomial's, under which it must restrict to the
polynomial. A restricted surface that vanishes identically gives nothing, and a skipped face of the
parent, listed in `la.parent.skipped_faces`, gives nothing either. The parent is analysed with the
same `max_face_points` and `timeout`, and when it skips faces the limit surfaces may be
incomplete; the report says so. Its analysis is kept for the next integral of the same family, so
that a sweep over the sets of massless legs of one graph analyses its generic family once.
Kinematics that are the generic ones in renamed invariants, related to them by an invertible
linear map, have no parent.

Whether `landau_analysis` looks for limit surfaces when `limits` is not given is set by
`DEFAULT_LIMITS` of `feynkit.landau`: True, False, or `"one-loop"` for one-loop integrals only. It
is False. At one loop the closed form `one_loop_landau_surfaces`, which the report prints, gives the
limit surfaces already, and beyond one loop the parent can cost more than the family: the massless
double box `15e|24|3e|4e|5|e|:zzzzzzz` on shell takes about 40 s without the parent and about two
minutes with `limits=True`, most of it the parent's analysis, which skips 57 faces. The report
takes `limits` too (`AnalysisReport.from_integral`, `FeynmanIntegral.to_latex` and `to_text`), and
`fk analyse --limits` asks for them.

On the one-loop bubbles, triangles and boxes, with every set of massless legs, the Landau surfaces
lie within the one-loop closed form and, with the limit surfaces, equal it; every factor tested
dropped. Two questions stay open. Whether every non-vanishing restricted factor lies in the Euler
discriminant is not known, which is why each is tested. Beyond one loop the two lists together
need not be the whole Euler discriminant: the parachute has a component of it outside the
principal Landau determinant at generic kinematics (Fevola, Mizera and Telen 2024, eq. (3.18)),
where there is no parent.

### LandauAnalysis fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `la.face_discriminants` | `tuple[FaceDiscriminant, ...]` | One per face of the Newton polytope |
| `la.principal_a_determinant` | `sp.Expr` | Product of the distinct kinematic factors |
| `la.landau_surfaces` | `tuple[sp.Expr, ...]` | The factors themselves |
| `la.skipped_faces` | `tuple` | Faces with more points than `max_face_points`, or past `timeout` |
| `la.limit_surfaces` | `tuple[LimitSurface, ...]` | Restricted factors of the parent's surfaces outside the Landau surfaces, confirmed by a drop of the count |
| `la.limit_candidates` | `tuple[LimitSurface, ...]` | The other such factors, not confirmed |
| `la.parent` | `LandauAnalysis \| None` | The analysis of the parent family, if there is one |

A `LimitSurface` has `surface`, the `parent_surfaces` whose restrictions it divides,
`generic_count`, the `counts` at its `points`, `reason`, None when the counts dropped, and
`confirmed`.

### FaceDiscriminant fields

| Attribute | Type | Description |
|-----------|------|-------------|
| `face.dimension` | `int` | Dimension of the face |
| `face.exponents` | `tuple[tuple[int, ...], ...]` | Exponent vectors on the face |
| `face.coefficients` | `tuple[sp.Expr, ...]` | Kinematic coefficients |
| `face.discriminant` | `sp.Expr` | Discriminant of the restriction, 1 if trivial |
| `face.is_simplex` | `bool` | Lattice points affinely independent |
| `face.principal` | `bool` | Elimination ideal had a single generator; with several, the discriminant holds the factors of their greatest common divisor |
| `face.dominant` | `bool` | Elimination ideal was zero: a component projects onto all of kinematic space and is left out, and the discriminant holds the hypersurfaces the other components project onto |

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

## Degenerate faces

A face $F$ of the Newton polytope of $G$, the polytope itself included, is degenerate when $G$
restricted to $F$, the sum of its terms with exponents on $F$, has a singular point in the torus:
$G|_F = u_1 \partial_1 G|_F = \dots = u_N \partial_N G|_F = 0$ has a solution in
$(\mathbb{C}^*)^N$. When the polytope is full-dimensional and no face is degenerate, the principal
A-determinant does not vanish and $|\chi(X)| = N!\,\mathrm{Vol}$ (section 10.6 of the mathematics
reference). `FeynmanIntegral.face_degeneracy(point)` decides every face exactly at a rational
kinematic point, keyed as `TorusCount.point` keys it, with $\mu = 1$; without a point it decides
them at the generic point of the integral's kinematics, over the field of rational functions in its
symbols, where a face is degenerate exactly when it is degenerate on a dense set of kinematic
points.

```python
import sympy as sp

from feynkit import FeynmanIntegral

fi = FeynmanIntegral.from_cnickel("11e|e|:nn")  # the massive bubble
s = sp.Symbol("s", real=True)
m1, m2 = (edge.get_mass() for edge in fi.graph.get_internal_edges())

# s = (m_1 + m_2)^2: G has the edge (u_1 - 2 u_2)^2 at mu = 1.
threshold = fi.face_degeneracy({s: 9, m1**2: 1, m2**2: 4})
for face in threshold.degenerate_faces:
    print(face.dimension, [threshold.points[i] for i in face.point_indices], face.tjurina)
print(threshold.volume, threshold.non_degenerate)

generic = fi.face_degeneracy()  # at the generic point of the kinematics
print(generic.mode, generic.non_degenerate, generic.volume)
```

prints

```
1 [(2, 0), (0, 2), (1, 1)] 1
3 False
generic True 3
```

so the bubble has $|\chi(X)| = 3$ for generic kinematics, and on the threshold the torus counts
above find 2.

Vertices and faces whose points are affinely independent are never degenerate. An edge is decided
by the discriminant of $G$ along it, and any other face in Singular, by whether the saturation
$(g_F, \partial_1 g_F, \ldots, \partial_d g_F) : (t_1 \cdots t_d)^\infty$ is the unit ideal, with
$g_F$ the face polynomial in the lattice coordinates $t$ of $F$. All faces go to Singular in one run
of at most `timeout` seconds, 120 by default; past it, the face Singular was on runs alone, and a
face that passes the limit alone is left undecided, with a reason. `check=True`, the default,
decides the same faces again over $\mathbb{F}_p$, $p$ the largest prime below $2^{29}$ that divides
no coefficient, as a probabilistic cross-check that never decides. Without Singular, faces that need
it raise `RuntimeError`.

Where coefficients vanish at the point, the faces are those of the smaller polytope of the point,
and the difference of the normalised volumes is reported as `support_loss`, not as degeneracy.
`face_degeneracy_from_polynomial(g, variables, point)` in `feynkit.degeneracy` does the same for
any polynomial. At the generic point the larger graphs take minutes: the hexagons and the off-shell
massless double box pass the default limit.

| Attribute | Description |
|-----------|-------------|
| `mode`, `point` | `"point"` with the point as (quantity, value) pairs, or `"generic"` with None |
| `points` | The exponents with non-zero coefficients, the points of the polytope |
| `volume`, `support_loss` | $N!\,\mathrm{Vol}$ of that polytope, and how much smaller it is than for generic kinematics |
| `faces` | One `FaceDegeneracy` per face, in the order of `PolytopeData.faces` |
| `prime` | The prime of the check over $\mathbb{F}_p$, or None |
| `degenerate_faces`, `undecided_faces` | The faces decided degenerate, and those left undecided |
| `non_degenerate` | True when no face is degenerate, False when one is, None when undecided faces leave it open |

A `FaceDegeneracy` has `point_indices`, `dimension`, `degenerate` (None when undecided), `method`
(`vertex`, `simplex`, `edge` or `groebner`), `singular_dimension`, the dimension of the singular
locus in the torus ($-1$ when empty), `tjurina`, its total Tjurina number when it is finite,
`modular`, the verdict over $\mathbb{F}_p$, and `reason`. The Tjurina number is taken in the lattice
coordinates of the face, those of the lattice spanned by the differences of its points.

`fi.face_lattice(..., degeneracy=True)` gives each `DecoratedFace` its verdict at the generic point
as `degenerate`. The report's `degeneracy` section, which `--sections` or `fk analyse -D` adds,
decides the faces at the point of the torus counts, drawn with the same seed;
`fk analyse -D --degeneracy-mode generic` and
`AnalysisReport.from_integral(fi, ["degeneracy"], degeneracy_mode="generic")` use the generic
point instead.

```python
sunrise = FeynmanIntegral.from_cnickel("111e|e|:nnn")
analysis = sunrise.face_degeneracy()
print([(f.dimension, len(f.point_indices), f.singular_dimension, f.tjurina)
       for f in analysis.degenerate_faces])
```

prints `[(2, 4, 0, 1), (2, 4, 0, 1), (2, 4, 0, 1)]`: the three faces of the fully massive sunrise
on which $G$ is $\mathcal{U}_{\{i,j\}} \mathcal{G}_{\Gamma/\{i,j\}}$ have a node for generic
kinematics. The discriminant of one of them vanishes identically on the kinematic space (Fevola,
Mizera and Telen 2024, example 2.5).

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
| `kinematics` | `tuple[StoredKinematics, ...]` | The graphs stored for the polytope, with their kinematic axes and class, oldest first |

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
The same first open gives such a file the table of kinematic classes below.

### Kinematic classes in the database

A record is a Newton polytope, and kinematics that leave the polytope unchanged share it: the
massive box with equal masses, or with on-shell legs, has the polytope of the generic massive
box. `store()` therefore also records the graph's CNickel string and kinematic axes for the
polytope, once each, and `rec.kinematics` lists them as `StoredKinematics(cnickel,
internal_axis, external_axis, kinematic_class)`, oldest first. `all_integrals` filters on them,
keeping the polytopes with a record that matches every filter given; an unknown class or axis
value raises `ValidationError`:

```python
with FeynkitDatabase("analysis.db") as db:
    massive = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:nnnn")
    db.store(massive)
    rec = db.store(massive.with_kinematics("equal_masses"))
    [k.kinematic_class for k in rec.kinematics]      # ['generic', 'equal_masses']
    db.all_integrals(kinematic_class="equal_masses")
    db.all_integrals(internal_axis="generic", external_axis="on_shell")   # in no class
```

`db.summary()` lists the classes of each polytope, or `?` when none is recorded. Polytopes stored
by earlier versions, or by releases up to 0.4.0 writing to the file later, have none, since a
record does not keep the momentum products the axes are read from; storing the integral again
adds its class.

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
| `max_face_points` | 14 | Faces of the Newton polytope with more monomials are left out of the Landau analysis and listed as skipped |
| `d0` | None | $D_0$ of the `resonance` section, which takes $D = D_0 - 2\varepsilon$: an integer or a `Fraction`; None reads it from the dimension of the integral when that is $D_0 - 2\varepsilon$ with $D_0$ a number, and takes 4 otherwise |

The section names, in `feynkit.io.report.SECTION_NAMES`, are `identity`, `conventions`,
`polynomials`, `representations`, `polytope`, `torus`, `gkz`, `resonance`, `faces`, `face_lattice`, `hasse`, `symmetries`,
`landau` and `schwinger`. The first three are always built. The rest are built only when named, so a
survey can ask for a short report without the automorphism and Landau computations. Without
`sections` every section but `torus`, `degeneracy` and `hasse` is built, the tuple `DEFAULT_SECTIONS`. The point counts of `torus` take seconds
for five propagators, and seven exceed the default budget (see
[Torus point counts](#torus-point-counts)); `to_latex` and `to_text` count with seed 0 and the
default budget, and raise `ValidationError` when the count cannot run, as for an integral with
kinematic constraints.

```python
latex = fi.to_latex(["polytope", "gkz"], title="Massive triangle")
```

The Landau section dominates the build time: the kite `12e|23|3|e|:zzzzz` takes about 7 s in all,
6 s of it in the Landau analysis, and the massive box `12e|3e|3e|e|:nnnn` about 7 s, 5 s of it in
the analysis. With `limits=True`, at kinematics that specialise the generic ones, the analysis of the parent
family and the counts of the limit surfaces come on top (see [Specialised kinematics](#specialised-kinematics)).
A survey over many graphs can leave the section out and keep everything else:

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

The document has up to fifteen parts:

1. The title; no author, date or abstract.
2. A summary table: loops, propagators, external legs, the monomial counts of $F$ and $G$,
   independent invariants, codimension, whether the integral is scaleless, polytope vertices,
   normalised volume, the candidate master count (with `torus`; "none" when the counts give no
   candidate), the numbers of unidentified and of support product faces (with `faces`),
   $|\mathrm{Aut}(P)|$, toric generators and Landau surfaces.
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
10. Resonance: with $D = D_0 - 2\varepsilon$, $D_0$ from `d0`, from the dimension of the integral
    or 4, and where it came from, and the integral's powers when they are integers, 1 on every edge otherwise, one row per facet of the Newton polytope with its
    inequality and $l_F(\beta)$, where it is resonant, whether at $\varepsilon = 0$, where it is
    admissible and whether it makes the system reducible, then the resonant values with
    $-1 \le \varepsilon \le 1$ (see
    [Resonant and admissible facets](#resonant-and-admissible-facets)). Below full dimension it
    also says where $\beta$ lies in the span of $A$.
11. Faces as graphs: how a face is compared with a product of Symanzik
    polynomials of minors, a table of the facets with their inequalities and graphs, the number of
    faces of each class up to codimension 2, and up to ten faces that are not identified, support
    products among them, with $G|_F$ and the prediction written out (see [Graphs of the faces](#graphs-of-the-faces)). Below full
    dimension it says that the faces are not identified.
12. Symmetries: the order and vertex orbits of $\mathrm{Aut}(P)$, the graph automorphisms, the
    coefficient-preserving subgroup, and the symmetry pairs with the identity each gives. For a
    Newton polytope that is not full-dimensional, $\mathrm{Aut}(P)$ is its group in its affine
    hull, and the section counts the symmetry pairs and says why their identities hold only
    trivially.
13. Landau surfaces: the factors of the reduced principal A-determinant by face dimension, split
    into first and second type for one-loop graphs, with the skipped faces, each named by its
    dimension and number of points and marked when it ran past the time limit, and the
    caveats. With `limits=True`, at kinematics that specialise the generic ones, the limit surfaces
    follow apart,
    labelled as confirmed by a drop of the count of critical points at random points, with the
    counts, then the candidates with the reason each is not confirmed, and a note when the
    analysis of the generic family skipped faces, so that the limit surfaces may be incomplete.
14. The Schwinger-representation system (section 10), its equivalence to the Lee-Pomeransky
    configuration and its reduction to the $\tilde F$ block.
15. References, the works cited in order of first citation.

Parts 6 to 14 appear when their sections are built. When the `polytope` section is built and the
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

7. Liu, Q. and Cai, Z. (2025). On the unimodular isomorphism problem of convex lattice polytopes.
   [arXiv:2506.23846](https://arxiv.org/abs/2506.23846)

8. Beck and Robins (2015). *Computing the Continuous Discretely.* 2nd ed., Undergraduate Texts in
   Mathematics, Springer. [doi:10.1007/978-1-4939-2969-6](https://doi.org/10.1007/978-1-4939-2969-6)

9. Bruns, Gubeladze and Trung (1997). Normal polytopes, triangulations, and Koszul algebras.
   *J. reine angew. Math.* **485**, 123-160.
   [doi:10.1515/crll.1997.485.123](https://doi.org/10.1515/crll.1997.485.123)

10. Hochster (1972). Rings of invariants of tori, Cohen-Macaulay rings generated by monomials, and
    polytopes. *Ann. of Math.* **96**, 318-337. [doi:10.2307/1970791](https://doi.org/10.2307/1970791)

11. Batyrev (1994). Dual polyhedra and mirror symmetry for Calabi-Yau hypersurfaces in toric
    varieties. *J. Algebraic Geom.* **3**, 493-545.
    [arXiv:alg-geom/9310003](https://arxiv.org/abs/alg-geom/9310003)

12. Britto, R., Grimm, T.W. and Hoefnagels, A. (2026). Resonance and differential reduction of
    Feynman integrals. *JHEP* **09**, 018. [arXiv:2606.09978](https://arxiv.org/abs/2606.09978)

13. Schulze, M. and Walther, U. (2012). Resonance equals reducibility for A-hypergeometric
    systems. *Algebra Number Theory* **6** (3), 527-537. [arXiv:1009.3569](https://arxiv.org/abs/1009.3569)

14. Arkani-Hamed, N., Hillman, A. and Mizera, S. (2022). Feynman polytopes and the tropical
    geometry of UV and IR divergences. *Phys. Rev. D* **105**, 125013.
    [arXiv:2202.12296](https://arxiv.org/abs/2202.12296)

15. Matusevich, L.F., Miller, E. and Walther, U. (2005). Homological methods for hypergeometric
    families. *J. Amer. Math. Soc.* **18**, 919. [arXiv:math/0406383](https://arxiv.org/abs/math/0406383)

16. Klausen, R.P. (2023). *Hypergeometric Feynman Integrals.* PhD thesis, Johannes Gutenberg
    University Mainz. [arXiv:2302.13184](https://arxiv.org/abs/2302.13184)

17. Bitoun, T., Bogner, C., Klausen, R.P. and Panzer, E. (2019). Feynman integral relations from
    parametric annihilators. *Lett. Math. Phys.* **109**, 497.
    [arXiv:1712.09215](https://arxiv.org/abs/1712.09215)

18. Huh, J. (2013). The maximum likelihood degree of a very affine variety. *Compositio Math.*
    **149**, 1245. [arXiv:1207.0553](https://arxiv.org/abs/1207.0553)

19. Kouchnirenko, A.G. (1976). Polyèdres de Newton et nombres de Milnor. *Invent. Math.* **32**, 1-31.

20. Hausel, T. and Rodriguez-Villegas, F. (2008), with an appendix by N.M. Katz. Mixed Hodge
    polynomials of character varieties. *Invent. Math.* **174**, 555-624.
    [arXiv:math/0612668](https://arxiv.org/abs/math/0612668)

21. Lee, R.N. (2014). LiteRed 1.4: a powerful tool for the reduction of the multiloop integrals.
    *J. Phys. Conf. Ser.* **523**, 012059. [arXiv:1310.1145](https://arxiv.org/abs/1310.1145)

22. Fevola, C., Mizera, S. and Telen, S. (2024). Principal Landau determinants.
    *Comput. Phys. Commun.* **303**, 109278. [arXiv:2311.16219](https://arxiv.org/abs/2311.16219)

23. Dlapa, C., Helmer, M., Papathanasiou, G. and Tellander, F. (2023). Symbol alphabets from the
    Landau singular locus. *JHEP* **10**, 161. [arXiv:2304.02629](https://arxiv.org/abs/2304.02629)

24. Jimenez-Santacruz, M., Lopez-Arcos, C. and Quintero Velez, A. (2026). Canonical differential
    equations for Feynman integrals from A-hypergeometric systems in the Schwinger
    representation. [arXiv:2609.16107](https://arxiv.org/abs/2609.16107)

25. Borinsky, M., Munch, H.J. and Tellander, F. (2023). Tropical Feynman integration in the
    Minkowski regime. *Comput. Phys. Commun.* **292**, 108874.
    [arXiv:2302.08955](https://arxiv.org/abs/2302.08955)

26. Chestnov, V., Matsubara-Heo, S.J., Munch, H.J. and Takayama, N. (2023). Restrictions of
    Pfaffian systems for Feynman integrals. *JHEP* **11**, 202.
    [arXiv:2305.01585](https://arxiv.org/abs/2305.01585)
