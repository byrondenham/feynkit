# feynkit

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

feynkit is a Python library for the parametric and polyhedral structure of Feynman integrals. You
give it a graph, usually as a CNickel string such as `12e|2e|e|:nzz`. It builds the Symanzik
polynomials, the Lee-Pomeransky polynomial $G = \mathcal U + \mathcal F$ and its GKZ system, and
then works with the Newton polytope of $G$: what each face is, where the system is resonant, which
faces are degenerate at a kinematic point, and how the faces line up with the sectors of an IBP
reduction. Arithmetic is exact, and the guide says where a result rests on counts over finite
fields.

## What it computes

- **Representations.** Symanzik polynomials; the Schwinger, Feynman and Lee-Pomeransky
  parametrisations; the GKZ A-matrix and $\beta$, and the Cayley system of the Schwinger
  representation; toric ideals.
- **The Newton polytope.** Facets and normalised volume; lattice invariants (Ehrhart polynomial,
  $h^*$-vector, integer decomposition property, normality); the decorated face lattice, with the
  graph of each face (a contraction, or a product of the polynomials of minors, checked exactly),
  where every face is resonant as $D = D_0 - 2\varepsilon$ varies, and whether the GKZ system is
  reducible; Hasse diagrams in TikZ.
- **Kinematics and singularities.** Kinematic classes (generic, massless on or off shell, equal
  masses); Landau analysis through the principal A-determinant; the faces on which $G$ is
  degenerate at a point; master counts from torus point counts or from critical points over finite
  fields.
- **Families and sectors.** Integral families with a momentum routing and numerators; the sector
  hierarchy, with the zero sectors, a count for each sector and the symmetries between sectors;
  export of a family to Kira.
- **Comparison and symmetry.** Unimodular and affine equivalence; polytope automorphisms and
  symmetry pairs; arbitrary A-configurations; conformal and BMS families.
- **Output.** An analysis report in LaTeX or plain text, a SQLite cache, the 1PI graphs with given
  loops, legs and propagators as CNickel strings, and the `fk` command.

## Installation

feynkit is not on PyPI yet. Install it from source:

```bash
git clone https://github.com/byrondenham/feynkit.git
cd feynkit
pip install -e .              # or: uv sync
pip install -e ".[backends]"  # optional: PyNormaliz and python-flint
```

Some computations use other programs when they are installed. Everything else works without them,
and a function that needs one says so.

| Program | Used for |
|---------|----------|
| [Singular](https://www.singular.uni-kl.de/) | degenerate faces, critical point counts, faster Landau eliminations |
| [4ti2](https://4ti2.github.io/) | toric ideals, much faster than the built-in SymPy backend |
| [msolve](https://msolve.lip6.fr/) 0.10.1 or later | an optional, faster backend for critical point counts |
| PyNormaliz, python-flint | lattice invariants of large polytopes; a cross-check of point counts |

feynkit writes the input of a Kira job and reads its results; running the job needs Kira itself.

## Command line

```bash
fk analyse "12e|2e|e|:nzz"                    # every section for the one-mass triangle
fk analyse "111e|e|:nnz" -L                   # where each face of the sunrise is resonant
fk analyse "111e|e|:nnn" -D                   # faces where G is singular in the torus
fk analyse "12e|3e|3e|e|:zzzz" --kinematics massless_on_shell --sectors
fk analyse "12e|2e|e|:nnn" --latex triangle.tex
fk compare "12e|2e|e|:nzz" "12e|2e|e|:znz"    # two placements of one mass
```

Quote the string, since an unquoted `|` is a shell pipe. Results are cached in `feynkit.db` in the
working directory unless you pass `--no-db`. The [guide](docs/guide.md#cli-fk) lists every option.

## Python

```python
from pathlib import Path
from feynkit import FeynmanIntegral, polytope_data

fi = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")   # triangle, one massive line
print(fi.symanzik.u)                                  # a_1 + a_2 + a_3
print(fi.gkz.a_matrix.shape)                          # (4, 7): G has 7 terms
print(polytope_data(fi.newton_polytope.points).normalized_volume)   # 5

lattice = fi.face_lattice(nu={1: 1, 2: 1, 3: 1})      # unit powers, D = 4 - 2 eps
print(len(lattice.faces))                             # 28, the empty face included
print(fi.sectors().totals().non_zero)                 # 5 non-zero sectors

Path("triangle.tex").write_text(fi.to_latex())        # the analysis report
```

The report is a LaTeX article that states its conventions, cites a source for each claim and says
what feynkit does not compute; [`docs/triangle_analysis.pdf`](docs/triangle_analysis.pdf) is the
one for the massive triangle. Graphs can also be built edge by edge with `Graph` and `Edge`.

## CNickel strings

A Nickel index names a graph up to relabelling of its vertices. CNickel adds a mass code for each
propagator, `z` for massless and `n` for massive, so the string fixes the integral family:

| Diagram | CNickel |
|---------|---------|
| Massless bubble | `11e\|e\|:zz` |
| One-mass triangle | `12e\|2e\|e\|:nzz` |
| Massless box | `12e\|3e\|3e\|e\|:zzzz` |
| Massive sunrise | `111e\|e\|:nnn` |
| Massless planar double box | `123\|45\|4e\|5e\|e\|e\|:zzzzzzz` |

Any labelling is accepted as input, and `fi.cnickel` gives the canonical string, so the three
one-mass triangles share one.

## Documentation

- [`docs/guide.md`](docs/guide.md): the user guide: the API, the CLI options and a library of
  standard diagrams.
- [`docs/mathematics-reference.md`](docs/mathematics-reference.md): the mathematics behind each
  computation, with sources.
- [`examples/`](examples/): runnable scripts; `bubble_example.py` is the shortest.

## Development

```bash
uv sync                   # the package and its development dependencies
uv run pytest             # in parallel, with coverage
uv run pytest -m slow     # the slow acceptance tests, left out by default
```

Every test has a wall-clock limit; `--timeout=0` removes it, and `-n 0` runs the tests in one
process.

## Citing

If you use feynkit in published work, please cite it; [`CITATION.cff`](CITATION.cff) has the
details.

## Licence

MIT; see [LICENSE](LICENSE).
