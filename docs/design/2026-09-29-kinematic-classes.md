# Kinematic classes

Design note for a label on `FeynmanIntegral`, in the report and in the database, that names the
kinematic class of an integral, and for storing the two axes the label is read from.

## Purpose

The label names one of four classes:

- generic masses and momenta;
- massless propagators with off-shell legs;
- massless propagators with on-shell legs, $p_i^2 = 0$;
- equal internal masses.

They are the kinematics of the entries `generic_generic`, `zero_generic`, `zero_zero` and
`equal_generic` of the principal Landau determinant database (Fevola, Mizera and Telen, Comput.
Phys. Commun. 303 (2024) 109278, arXiv:2311.16219), whose names give the internal masses, then the
external ones. The label supports systematic enumeration of graphs by kinematic class.

A CNickel string gives the mass of each propagator: `z` for massless, `n` for a mass of its own,
and a shared code such as `a` for masses set equal (mathematics reference, section 1). It says
nothing about the legs, so the on-shell and off-shell massless boxes are both
`12e|3e|3e|e|:zzzz`. The database keys a row by the Newton polytope alone, and some kinematics
leave the polytope unchanged: equal masses never change the support of $G$, and nor do on-shell
legs when every propagator is massive. Stored after the generic massive box, the equal-mass box
therefore lands in the same row, and the record returned keeps the CNickel string and label of
the first. The massive box on shell has the same polytope again, but $\chi = 11$ against 15 for
generic kinematics (entries `A4_generic_zero` and `A4_generic_generic`). It lies in none of the
four classes, which is why the database stores the two axes as well as the label.

## Current state

- `momentum_products` maps each pair $i < j$ to $p_i \cdot p_j$. With `use_mandelstam=True`, the
  default of `from_cnickel`, the values are written in the invariants of `standard_invariants`,
  $p_i^2$ and the planar $s_{i \ldots j-1}$ ($s$ alone for two legs); otherwise they are symbols
  `p1p2`, and so on. No $p_i^2$ is stored. With all momenta incoming, conservation gives
  $p_i^2 = -\sum_{j \ne i} p_i \cdot p_j$. A key given as $(j, i)$ is read like $(i, j)$ when
  $G$ is built, so $G$ does not depend on the order.
- Masses live on the edges: `Edge.get_mass()` is the mass, or $m_e$ when none is given.
  `FeynmanIntegral` has no `masses` argument, and `with_` passes unknown keywords to the
  constructor, so `with_(masses=...)` raises `TypeError`. Equal masses are written with shared
  codes (`from_cnickel("12e|2e|e|:aaa")`), with edges built on one symbol, or through
  `with_(graph=...)`.
- `Graph.cnickel()` is canonical over the vertex labellings and the names of the shared masses:
  `:sss` gives `:aaa`, and `:aab` gives `:aan`, since one propagator alone carries $m_b$. Digit
  mass codes other than `0` are refused.
- `Graph.cnickel()` and $G$ still disagree in two cases. Edges with `mass=None` give `:zz` while
  $G$ is massive, since `get_mass()` supplies $m_e$. A mass `Float(0.0)` gives `n`, since
  `Float(0.0) != 0` in SymPy. The axes below read `get_mass()`, as $G$ does; fixing the codes is a
  separate change.
- `kinematic_constraints` is stored and carried by `with_`, but never applied.
- `torus_count(on_shell=...)` substitutes into the products of a copy made with `with_`. Its keys
  must be symbols of the products, such as `Symbol("p1^2", real=True)`.
- `gkz` has one column per monomial of $G$ at every kinematic point, on shell included (see the
  note on GKZ support, `docs/design/2026-09-27-gkz-support.md`). The on-shell class relies on it.

## Classes

### Axes

The internal axis is read from the masses $m_e$ of the $N$ internal edges, compared as SymPy
expressions; the first rule that applies decides.

1. `zero`: every $m_e$ is exactly 0.
2. `equal`: $N \ge 2$ and every $m_e$ is the same symbol.
3. `generic`: the nonzero $m_e$ are distinct symbols. Massless lines are allowed, as in
   `12e|2e|e|:nzz`, which the database of Fevola, Mizera and Telen does not list, and so is a
   single massive line, as in `0|:n`.
4. `other`: a mass that is a number or an expression, `Float(0.0)` included, or a symbol shared
   by some lines but not all (`:aan`, `:aaz`).

The external axis is read from the $n$ legs and the products $P_{ij}$, $i < j$, each looked up
under $(i, j)$ or $(j, i)$, a missing product counting as 0. Put $k = n(n-1)/2$,
$p_i^2 = -\sum_{j \ne i} P_{ij}$, and $r$ the rank of the Jacobian of the $P_{ij}$ with respect
to their free symbols.

1. `other` if the integral has `kinematic_constraints`.
2. `off_shell` if $n < 2$: there is no invariant to constrain.
3. `other` unless every product is a polynomial of degree at most 1 in its free symbols, with
   rational coefficients, containing no mass symbol and not $\mu$. So $1/s_{23}$,
   $\sqrt{s_{23}}$, Float or irrational coefficients, $p_1^2 = m_1^2$ and $\mu$ in a product all
   give `other`. The rank below is then that of a rational matrix, and exact.
4. `off_shell` if $r = k$: no relation among the invariants is imposed.
5. `on_shell` if every $p_i^2 = 0$ and $r = \max(k - n, 0)$: the legs are on shell and nothing
   else is imposed. The $n$ conditions are independent for $n \ge 3$, since the incidence matrix
   of the complete graph $K_n$ then has rank $n$; for $n = 2$ they reduce to $s = 0$.
6. `equal` if $n \ge 3$, every $p_i^2$ is the same nonzero expression and $r = k - n + 1$: equal
   external masses, nothing else imposed.
7. `other` otherwise: some legs on shell and others not, a vanishing $s_{12}$, numerical values.

The derivation never raises. It runs in `store`, in `_store_toric` through `toric_ideal`, and in
the report, all of which must keep working for every integral they accept.

The axis values follow the database of Fevola, Mizera and Telen: internal `zero`, `equal` and
`generic` are its internal names, and external `on_shell`, `equal` and `off_shell` its external
`zero`, `equal` and `generic`. Imposing each of its nine kinematics on the box by hand gives the
published point set and the axes the entry names.

### Classes

| Internal axis | `off_shell` | `on_shell` | `equal` | `other` |
|---|---|---|---|---|
| `zero` | `massless_off_shell` | `massless_on_shell` | `other` | `other` |
| `equal` | `equal_masses` | `other` | `other` | `other` |
| `generic` | `generic` | `other` | `other` | `other` |
| `other` | `other` | `other` | `other` | `other` |

The columns are the external axis. Every integral has exactly one pair of axes and one class. With
two or three legs, `massless_on_shell` makes every product vanish, so $F = 0$ and the integral is
scaleless; the class is still assigned. A massless vacuum graph is `massless_off_shell`.

### Deriving the class

The axes and the class are computed from the integral as it stands. A declared label would have
to be carried by `with_`, and would go stale: `with_(momentum_products=...)` can undo an on-shell
substitution, and `with_(graph=...)` can change the masses. Deriving them needs no new state, so
`__init__` and `with_` do not change and nothing has to be propagated. The on-shell box
`.with_(momentum_products=box.momentum_products)` is `massless_off_shell` again, and
`.with_(dimension=4)` keeps its class. Constructors that impose a class do so by substitution,
and the label follows. The cost is the rank of a rational matrix of at most $15 \times 15$
entries for six legs.

## Interfaces

```python
# feynkit/kinematics/classes.py
InternalAxis = Literal["zero", "equal", "generic", "other"]
ExternalAxis = Literal["off_shell", "on_shell", "equal", "other"]
KinematicClass = Literal[
    "generic", "massless_off_shell", "massless_on_shell", "equal_masses", "other"
]
KINEMATIC_CLASSES: tuple[KinematicClass, ...]   # in the order above
IMPOSABLE_CLASSES: tuple[KinematicClass, ...]   # all but "other"
CLASS_OF_AXES: Mapping[tuple[InternalAxis, ExternalAxis], KinematicClass]   # the named cells
def kinematic_axes(integral: FeynmanIntegral) -> tuple[InternalAxis, ExternalAxis]: ...
def kinematic_class(integral: FeynmanIntegral) -> KinematicClass: ...
def impose_kinematics(integral: FeynmanIntegral,
                      kinematic_class: KinematicClass) -> FeynmanIntegral: ...
# FeynmanIntegral
@cached_property
def kinematic_axes(self) -> tuple[InternalAxis, ExternalAxis]: ...
@cached_property
def kinematic_class(self) -> KinematicClass: ...
def with_kinematics(self, kinematic_class: KinematicClass) -> FeynmanIntegral: ...
@classmethod
def from_cnickel(cls, cnickel: str, *, kinematics: KinematicClass | None = None,
                 **kwargs: Any) -> FeynmanIntegral: ...
```

`from_cnickel` builds the integral as before, then calls `with_kinematics` unless `kinematics`
is None; `from_nickel` forwards its keywords and so accepts it too. The argument takes class
names, not substitution names such as `"on_shell"`. The argument, the property, the report, the
CLI and the database then share one vocabulary, and
`from_cnickel(s, kinematics=k).kinematic_class == k` holds for every `k` accepted. Substitution
names could be combined, with equal masses and on-shell legs for instance, and such combinations
land in `other`.

### Imposing a class

`with_kinematics` returns the integral itself when it already has the class asked for, so
`from_cnickel("12e|2e|e|:sss", kinematics="equal_masses")` keeps $m_s$. Otherwise it makes the
substitutions below and raises `ValidationError` if the result is not of that class. A name
outside `KINEMATIC_CLASSES`, or `other`, raises `ValidationError` listing the four classes one
can impose.

- `generic`, `massless_off_shell`: no substitution, so these raise unless the class holds.
- `massless_on_shell`: every $m_e$ must be 0, else `ValidationError` naming the massive
  propagators. Each $p_i^2$ computed from the products must be 0 or a symbol, as for the standard
  invariants ($p_i^2$, or $s$ for two legs), and each such symbol is set to 0 in every product,
  which is then expanded. For the box this gives $p_1 \cdot p_2 = p_3 \cdot p_4 = s_{12}/2$,
  $p_1 \cdot p_4 = p_2 \cdot p_3 = s_{23}/2$ and
  $p_1 \cdot p_3 = p_2 \cdot p_4 = -(s_{12} + s_{23})/2$. With dot products,
  $p_1^2 = -p_1 \cdot p_2 - p_1 \cdot p_3 - p_1 \cdot p_4$ is not a symbol, and the method asks
  for `use_mandelstam=True` rather than choose which products to eliminate. Fewer than two legs
  raise, as there is nothing to put on shell.
- `equal_masses`: every $m_e$ must be nonzero. Every internal edge gets `mass=m_a`, where
  `m_a = Symbol("m_a", nonnegative=True, real=True)` is the mass of code `a`, through
  `dataclasses.replace`, in a new `Graph` with the same energy scale; the products are unchanged.
  `from_cnickel("12e|2e|e|:nnn", kinematics="equal_masses")` is then the integral of
  `from_cnickel("12e|2e|e|:aaa")`, with CNickel `12e|2e|e|:aaa`.
- `other`: refused.

`equal_masses` accepts `:aab`, and numerical masses, and turns them into `:aaa`. This is a
deliberate choice. A class never changes which lines are massless, so `massless_on_shell` refuses
massive lines rather than zeroing them. The masses of massive lines, like the $p_i^2$ of the legs,
are values, and setting values is what a class does.

`torus_count(on_shell=...)` stays for ad hoc substitutions. On the massless box it gives the same
counts, and $C = 3$, as `with_kinematics("massless_on_shell").torus_count()`.

## Database

### Schema

`integrals` is unchanged. The repair step from `user_version` 2 to 3 creates

```sql
CREATE TABLE IF NOT EXISTS kinematic_classes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint      TEXT    NOT NULL,
    cnickel          TEXT    NOT NULL,
    internal_axis    TEXT    NOT NULL,
    external_axis    TEXT    NOT NULL,
    kinematic_class  TEXT    NOT NULL,
    stored_at        TEXT    NOT NULL,
    UNIQUE(fingerprint, cnickel, internal_axis, external_axis)
);
```

A row of `integrals` is a polytope: its key `fingerprint` is the SHA-256 of the sorted Newton
points, and its toric generators, automorphism data and entries in `equivalences` depend on the
points alone. A row of `kinematic_classes` records one graph with one pair of axes that gives the
polytope. The class is a function of the axes, stored so that queries need no table in SQL; if a
later version names more classes, a repair step recomputes the column from the axes. A row whose
class is `other` keeps its axes, so the massive box on shell, `generic` and `on_shell`, can be
split out later without storing it again. `cnickel` is what `Graph.cnickel` returns, with the
inconsistencies noted under Current state, or the empty string when it raises, as it does above
ten vertices, because SQLite treats NULLs in a UNIQUE key as distinct.

### Migration and compatibility

- `_migrate` runs numbered repair steps under `PRAGMA user_version`; this is step 3, one method
  and one entry in its list. A new file runs every step and starts at version 3. A file written
  by 0.4.0 or earlier, at version 0, runs steps 1 to 3 once; a file at version 2 runs step 3
  once. Reopening a file at version 3 writes nothing.
- The table is created by the step rather than by `_SCHEMA`, which runs on every open. On a
  read-only file without the table, `CREATE TABLE` in `_SCHEMA` would raise and the file could not
  be opened, while the step is skipped on such a file, which opens unrepaired. Until the step has
  run, reads return no class rows and `store` records none.
- Existing polytopes get no class row, meaning "not recorded". No default is filled in: a stored
  row does not hold its momentum products, and equal masses, or on-shell legs with massive lines,
  leave the points unchanged, so the axes cannot be recovered. Storing the integral again adds
  its row.
- Releases up to 0.4.0 read and write a file carrying the table: lookup, store, the toric cache
  in both directions, `find_equivalent`, `all_integrals` and `summary` work. They ignore the table,
  and what they store gets no class row.

### Key

`integrals.fingerprint` stays UNIQUE, so the toric and equivalence caches stay shared between
kinematics: the equal-mass box reuses the toric ideal of the generic one. The same graph in two
kinematics gives two rows of `integrals` when they change the points (the massless box, off and
on shell: 10 and 6 points), and one row with two class rows when they do not (the massive box,
generic and with equal masses: 14 points each).

The alternative, `ALTER TABLE integrals ADD COLUMN kinematic_class TEXT` with the key widened to
`UNIQUE(fingerprint, kinematic_class)`, needs the table rebuilt (create, copy, drop, rename),
since SQLite cannot drop the column constraint on `fingerprint`. After the rebuild 0.4.0 can
still read the file, but both of its upserts fail with "ON CONFLICT clause does not match any
PRIMARY KEY or UNIQUE constraint". That breaks `store`, and also `toric_ideal` on any integral
with a database, through `_store_toric`. Old rows would hold NULL, which the key does not
constrain, and both caches would need a new lookup rule. A column without the wider key records
only the first class stored for a polytope, which is the fault this note sets out to fix.

### Interface

- `StoredKinematics`, a `NamedTuple` of `cnickel`, `internal_axis`, `external_axis` and
  `kinematic_class`, all strings.
- `IntegralRecord.kinematics: tuple[StoredKinematics, ...] = ()`: the rows stored for the
  polytope, oldest first, filled by `_to_record`; `__repr__` lists the classes. A value written by
  a later version with more axis values or classes is returned as it is.
- `store` and `_store_toric` insert a row for `fi`, with `ON CONFLICT DO NOTHING`.
- `lookup(fi)` still finds the polytope; comparing `fi.cnickel` and `fi.kinematic_axes` with
  `record.kinematics` tells whether this graph and these axes were stored.
- `all_integrals(*, kinematic_class=None, internal_axis=None, external_axis=None)` returns the
  polytopes with a class row that matches every filter given.
- `summary()` gains a column `classes`, before `label`, with the distinct classes of each
  polytope joined by commas, or `?`.

## Report, CLI and JSON

- `Conventions` gains `kinematic_axes` and `kinematic_class`, since the class describes the
  kinematics, which this section states; the graph section already lists each $m_e$. The
  paragraph gains one sentence after the kinematics sentence naming the class and what it
  imposes. For `massless_on_shell` it says that $p_i^2 = 0$, leaving the free symbols of the
  products ($s_{12}$ and $s_{23}$ for the box). For `other` it names the two axes, as in "the
  masses are generic and every leg is on shell". The kinematics sentence lists $p_i^2$ among the
  invariants even when they vanish; the new sentence says that they do.
- `summary()` gains `("Kinematic class", value)` after "External legs", in both renderers, and
  so the JSON summary of `fk analyse --json` gains `"kinematic_class": "massless_on_shell"` by
  the snake-case rule, with no change to `_summary_json`.
- `fk analyse` gains `--kinematics CLASS`, one of `generic`, `massless_off_shell`,
  `massless_on_shell` and `equal_masses`, parsed by a type function in the manner of
  `_section_list`. An unknown name is a usage error listing the choices. The option is added to
  `_VALUE_OPTIONS`, so that the bare form reads its value as a value, not a second diagram. It
  sits beside `CNICKEL`, not in a section or report group, since it changes the integral. Help:
  "kinematic class to impose; by default the kinematics of the CNickel string". `_load` applies
  `with_kinematics` and reports a `ValidationError` in one line with status 1. The graph header
  prints "Kinematic class" whether or not the option is given, and the database record stores
  the axes and the class.

```bash
fk analyse "12e|3e|3e|e|:zzzz" --kinematics massless_on_shell -n
fk analyse "12e|3e|3e|e|:nnnn" --kinematics equal_masses --json --no-db
```

## Testing

- Axes and classes: every cell of the table, and $p_4^2 = 0$ alone, equal external masses on the
  box and the triangle, $s_{12} = 0$, on-shell legs with massive lines, $p_1^2 = m_1^2$ on the
  one-mass triangle, `:aab`, `:aaz`, `:sss`, a numerical mass, `Float(0.0)`, `0|:n`, a vacuum
  graph and dot products.
- Never raising: $s_{12} \to 1/s_{23}$, $p_1^2 \to \sqrt{s_{23}}$, $\mu$ or $\mu^2$ in a product,
  Float and irrational coefficients, `kinematic_constraints`, all `other`; products given under
  reversed keys, the same axes as in order.
- Imposing: `from_cnickel(s, kinematics=k).kinematic_class == k` over a list of graphs and every
  class they admit; each refusal and its message; unknown names and `other`; the integral itself
  returned when the class holds; re-derivation after `with_`; `:aab` to `:aaa`; the equal-mass
  triangle from `:nnn` against `:aaa` (CNickel, masses, $G$).
- Newton polytopes, against the database of Fevola, Mizera and Telen: points equal up to a
  permutation of coordinates, f-vectors equal and axes as named. The entries below all match, and
  so do the other five kinematics of `A4`. Four entries are committed, `A4_zero_generic`,
  `A4_zero_zero`, `par_zero_generic` and `kite_generic_generic`, so that the on-shell check always
  runs; the rest run under `FEYNKIT_PLD_DATA`.

| Graph | Class | Points | f-vector | Volume | Entry |
|---|---|---|---|---|---|
| `12e\|3e\|3e\|e\|:zzzz` | `massless_off_shell` | 10 | (10, 30, 30, 10) | 11 | `A4_zero_generic` |
| `12e\|3e\|3e\|e\|:zzzz` | `massless_on_shell` | 6 | (6, 15, 18, 9) | 3 | `A4_zero_zero` |
| `12e\|3e\|3e\|e\|:nnnn` | `generic`, `equal_masses` | 14 | (8, 16, 14, 6) | 15 | `A4_generic_generic`, `A4_equal_generic` |
| `12ee\|22e\|e\|:zzzz` | `massless_off_shell` | 9 | (9, 24, 24, 9) | 8 | `par_zero_generic` |
| `12ee\|22e\|e\|:zzzz` | `massless_on_shell` | 7 | (7, 15, 14, 6) | 3 | `par_zero_zero` |
| `12ee\|22e\|e\|:nnnn` | `generic`, `equal_masses` | 19 | (15, 33, 27, 9) | 35 | `par_generic_generic`, `par_equal_generic` |
| `15e\|24\|3e\|4e\|5\|e\|:zzzzzzz` | `massless_off_shell` | 46 | (46, 282, 636, 706, 421, 133, 20) | 903 | `dbox_zero_generic` |
| `15e\|24\|3e\|4e\|5\|e\|:zzzzzzz` | `massless_on_shell` | 26 | (26, 151, 380, 491, 341, 123, 20) | 238 | `dbox_zero_zero` |
| `15e\|24\|3e\|4e\|5\|e\|:nnnnnnn` | `generic`, `equal_masses` | 78 | (45, 171, 291, 282, 167, 60, 12) | 1422 | `dbox_generic_generic`, `dbox_equal_generic` |

The on-shell massless box has 6 monomials against 10 off shell. The bubble, triangle, sunrise and
kite on shell have $G = U$, with 2, 3, 3 and 8 points.

- Point counts: the on-shell box through the class gives the counts of
  `torus_count(on_shell=...)` and $C = 3$, the $\chi$ of `A4_zero_zero`.
- Database: a file at version 0 or 2 reaches version 3 once, gains the table, keeps its rows, and
  its records have no kinematics; a read-only old file still opens and reads; the generic and
  equal-mass massive boxes give one polytope with two rows; the off-shell and on-shell massless
  boxes give two polytopes; the massive box on shell is stored as `generic`, `on_shell`, `other`;
  `all_integrals` filters by class and by axis; the 0.4.0 `store` statement still succeeds on a
  new file.
- Report and CLI: the summary labels gain the row; one golden sentence per class, and one for
  `other` naming the axes; the JSON key; `--kinematics` with an unknown name (status 2), a
  refused class (status 1, one line on stderr), and in the bare form.

## Changes to core modules

- `integral.py`: the properties `kinematic_axes` and `kinematic_class`, the method
  `with_kinematics`, the keyword `kinematics` of `from_cnickel`, and their docstrings. `__init__`
  and `with_` are unchanged.
- `database.py`: the repair step that creates the table; `StoredKinematics`;
  `IntegralRecord.kinematics` and its `__repr__`; `_to_record` reads the rows; `store` and
  `_store_toric` insert one; `all_integrals` takes the class and axes; `summary()` gains a column.
- `polytope.py`, `a_configuration.py`, `landau.py`: none.

Elsewhere: the new `kinematics/classes.py`, exported from `feynkit.kinematics` and `feynkit`;
`io/report.py`, `io/report_latex.py` and `io/report_text.py`; `cli.py`; the guide, which gains a
section on classes, the option and the JSON key, and lists the shared mass codes that
`equal_masses` produces; the changelog.

## Decisions

- Schema: the side table, keyed on the fingerprint, the CNickel string and the two axes. The
  alternative, a column on `integrals` with the key widened, stops 0.4.0 and earlier from writing
  to a migrated file.
- Classes: the five-value `Literal`, with both axes stored now. The first class to add would use
  external `equal`; the database of Fevola, Mizera and Telen spans all nine pairs of zero, equal
  and generic. Adding classes changes `CLASS_OF_AXES` and the `Literal`, and a repair step
  recomputes the stored classes, but the schema stays as it is.
- `kinematic_constraints`: their presence makes the external axis, and so the class, `other`.
  They are not applied to $G$, but they state relations the class would otherwise ignore.
- Non-linear parametrisations, such as $p_1^2 \to M^2$, stay `other`. If they are ever accepted,
  the rank should be the exact generic rank over $\mathbb{Q}(\text{symbols})$, not the rank at a
  random point.
- `fk compare` takes no `--kinematics`.
