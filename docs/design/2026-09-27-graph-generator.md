# Graph generator

Design note for enumerating the one-particle-irreducible (1PI) graphs with given numbers of loops,
propagators and legs, with their mass colourings, as canonical CNickel strings, and for
property-based tests that run over them.

## Purpose

feynkit analyses one graph at a time, given as a CNickel string. Scans over families, and tests
that should hold for every graph, need the graphs themselves: all 1PI graphs with $L$ loops, $E$
propagators and $n$ legs, each once, with every assignment of masses up to the symmetries of the
graph. Two target ranges are used throughout:

- tier 1: one loop, from the bubble to the hexagon ($n = E$ from 2 to 6);
- tier 2: two loops with 2 to 4 legs, which forces $E \le 7$.

This note specifies:

- the graphs generated: what 1PI means here, the degree rule, legs, vacuum graphs, tadpoles and
  the colour alphabet;
- the algorithm and its interface, including an exact pruned search for the canonical labelling;
- three fixes to CNickel that the generator needs;
- counts and run times for both tiers, and what limits the reach;
- a validation that needs no published count beyond one cited sequence;
- property-based tests over relabellings of the generated graphs.

The counts below were computed with a prototype of the generator and reproduced independently: by
a separate enumeration, by Burnside's lemma and through `Graph.cnickel()`. The brute-force test
under Validation recomputes them in CI. Timings are from an Intel Core i5-6440HQ with
Python 3.13; under load they were 20 to 35 per cent higher.

## CNickel as it stands

### Encoding

A graph with $V$ internal vertices is written as $V$ entries, each terminated by `|`. Entry $i$
lists in ascending order the vertices $j \ge i$ joined to vertex $i$, one digit per propagator, so
a double propagator repeats the digit and a self-loop at $i$ is the digit $i$ once. One `e`
follows per external leg at the vertex. Legs carry no labels: the string records how many legs
each vertex has, and `Graph.from_cnickel` numbers them left to right. The triangle is `12e|2e|e|`,
the sunrise `111e|e|`, the tadpole with one leg `0e|`.

After a colon comes one mass code per propagator, in the order the propagators appear in the
topology. `_mass_from_code` in `feynkit/core/graph.py` accepts:

| Code | Mass |
|---|---|
| `z`, `0` | zero |
| `n` | $m_{idx}$, a mass of its own |
| `s` | the shared symbol $m_s$ |
| `a` to `y` except `n` and `s` | the shared symbol $m_c$ for the letter $c$ |

It also accepts `1` to `9` as shared labels, which do not work (see Three gaps).
`CNICKEL_GRAMMAR` in `feynkit/cli.py` and the guide document only `z` and `n`. External masses are
not part of the string; kinematics are set on the `FeynmanIntegral`.

### Canonical form

`Graph.cnickel()` minimises the pair (topology, colours) over all $V!$ labellings of the vertices,
comparing strings by code point, with the colours of parallel propagators sorted.
`nickel_index()` is the same search without colours, and both raise `NotImplementedError` above
$V = 9$. Digits sort before `e` and `e` before `|`, so long entries tend to come first, though
vertex 0 need not have the largest degree: in `112|2|34|4e|e|` it has degree 3 and vertex 2 has
degree 4. Hand-written strings in the tests are often not canonical: the kite `12e|23|3|e|` is
`123|23|e|e|`, the planar double box `15e|24|3e|4e|5|e|` is `123|45|4e|5e|e|e|`, and the box is
`12e|3e|3e|e|`, where the `nickel_index` docstring gives `13e|2e|3e|e|`.

### Three gaps

- Empty last entry. When the last vertex in canonical order has no legs and no self-loop, its
  entry is empty and the string ends in `||`. `from_cnickel` strips every trailing bar
  (`topology.rstrip("|")`), loses that vertex and raises. This happens for `123|4e|4e|4e||`
  (three chains of two propagators, three legs), for `111e||` (the sunrise with one leg) and for
  every vacuum graph with more than one vertex and no self-loop at the last vertex, such as
  `111||`. The guide's statement that `Graph.from_cnickel(s).cnickel() == s` for every canonical
  `s` is false for them. Two of the 25 topologies of tier 2 end in `||`. Empty entries in the
  middle of a string, such as `11122||e|` at three loops, already parse.
- Label names. `cnickel()` keeps the names of shared labels: `12e|2e|e|:bbn` and `12e|2e|e|:aan`
  are one mass pattern under two strings, and a label on a single propagator survives, so `:nan`
  becomes `:ann`, not `:nnn`.
- Digit labels. The digit code $d$ gives the symbol $m_d$, which is also the symbol that `n` gives
  the propagator with index $d$. In `12e|2e|e|:n1z` propagators 1 and 2 both get $m_1$, and the
  string canonicalises to `:aaz`. Digit labels never survive `cnickel()` either, which writes
  shared masses as letters. Nothing in the repository uses them.

## Graphs generated

### One-particle irreducibility and the degree rule

A graph is its internal multigraph, $V$ vertices and $E$ propagators, with $\ell_v \ge 0$ legs at
vertex $v$ and $n = \sum_v \ell_v$. The generator keeps the graphs that are

- connected, so that $L = E - V + 1$ (`Graph.get_loop_count`) and the spanning-tree sums in
  `polynomials/symanzik.py` are non-zero;
- bridgeless: deleting any one propagator leaves the internal graph connected. Legs are not edges
  for this purpose. A bridge carries a fixed momentum, so its propagator factors out of the
  integral and its variable is absent from $U$;
- of degree at least 3 at every vertex, counting a self-loop twice and each leg once. A vertex of
  degree 2 without a leg joins two propagators carrying the same momentum. With equal masses they
  are one propagator with its exponent raised, which the symbolic $\nu_e$ already cover; with
  different masses partial fractions reduce the pair to graphs with one propagator fewer.

Cut vertices are allowed: two bubbles sharing a vertex form a 1PI graph. Its integral factorises
into one-loop integrals, and `one_vertex_irreducible=True` removes such graphs, keeping those
whose propagators cannot be split into two non-empty sets sharing one vertex.

Every vertex of internal degree 2 carries a leg. Suppressing those vertices leaves a core in which
every degree is at least 3, so the core has at most $3(L - 1)$ propagators, and

$$E \le n + 3(L - 1), \qquad V \le n + 2L - 2 \qquad (L \ge 2),$$

while a one-loop graph is a cycle with $V = E \le n$. Two loops and at most four legs therefore
imply at most seven propagators.

### Legs

feynkit's $F$ depends on the legs only through the momentum entering each vertex, the sum of the
momenta of its legs; `polynomials/symanzik.py` skips pairs of legs at the same vertex. Several legs
at one vertex are therefore a one-leg graph with a composite momentum, and at generic kinematics
the support of $F$, and with it the Newton polytope, is that of the graph with one leg there. So
by default a vertex carries at most one leg. Several legs matter at special kinematics, where
$p_1^2 = p_2^2 = 0$ does not make $(p_1 + p_2)^2$ vanish, and `max_legs_per_vertex` allows them
(`None` for no limit). A graph with all its legs at one vertex has no momentum flowing through it
and the $F$ of a vacuum graph.

Legs are unlabelled, as in CNickel, so graphs that differ only in which momentum enters where are
one string. A kinematic assignment that tells legs apart, such as one off-shell leg among on-shell
ones, is applied to the `FeynmanIntegral` afterwards; enumerating those assignments is out of
scope.

### Vacuum graphs, one-leg graphs and tadpoles

- $n = 0$ and $n = 1$ are generated only on request, and default ranges start at $n = 2$. Both
  are still covered by the validation and round-trip tests.
- Under the degree rule the only two-loop vacuum graph without self-loops is the sunrise `111||`.
  The one-loop tadpole `0|` has a vertex of degree 2 and is not generated; the degree rule has no
  exception for it. Massless vacuum graphs have $F = 0$ and are scaleless; the generator does not
  filter them. Their strings need the parser fix below.
- With $n = 1$, momentum conservation sets the momentum of the leg to zero, so $F$ has no momentum
  term and the integral is that of the graph without the leg.
- `FeynmanIntegral.from_cnickel` uses Mandelstam invariants by default, which need two legs; for
  $n \le 1$ the caller passes `use_mandelstam=False`, as `fk` does.
- Self-loops appear only with `self_loops=True`. Massless self-loops are scaleless but are kept,
  like any other colouring.
- From three loops, a block without legs can hang from a cut vertex, such as a vacuum sunrise
  sharing a vertex with a bubble. It is 1PI and kept; its integral carries a vacuum factor, and
  `one_vertex_irreducible=True` removes it.

### Colour alphabet

`masses` chooses the codes emitted.

- `"zn"` (default, and used for both tiers): each propagator is massless or has a mass of its
  own. With symbolic kinematics, equal masses change coefficients of $G$ but not its support: the
  mass term $U \sum_e m_e^2 a_e / \mu^2$ only adds squared masses, and the momentum part has
  independent symbols. The Newton polytope, its volume, f-vector, facets and equivalence class
  therefore depend only on which propagators are massive.
- `"z"`: massless only, one string per topology.
- `"shared"`: equal masses as well. A class of two or more propagators with a common mass is
  written with a letter, `a`, `b`, `c` in order of first appearance in the string; a mass on one
  propagator is `n`. Equal masses matter to the Landau analysis, to point counts and to
  coefficient-preserving symmetries.

The generator never emits `0`, `s` or digits. At seven propagators at most three classes occur.

`cnickel()` changes: it minimises the colour string over renamings of the shared masses as well
as over labellings, and writes a mass on a single propagator as `n`. The minimum names its
classes in order of first appearance, and where several classes first appear in one group of
parallel propagators, the class with more propagators there takes the earlier letter; any other
naming makes that group, and so the string, larger. Only the orders within ties need trying.
`cnickel()` then gives one string per equal-mass family and never writes `s` or a digit, and the
generator's strings are exactly its output. Two tests in `tests/core/test_graph.py` change:
`12e|2e|e|:aab` becomes `:aan` and `12e|2e|e|:sss` becomes `:aaa`. The letters `a` to `y` without
`n` and `s` name at most 23 classes; a graph with more raises `NotImplementedError`.

## Design

### Cores and subdivisions

The generator runs the suppression above backwards.

1. Cores. For $L \ge 2$, enumerate the multisets of $E_c = V_c + L - 1$ vertex pairs on $V_c$
   labelled vertices, $1 \le V_c \le 2(L - 1)$, loops allowed, dropping a partial multiset as
   soon as a vertex can no longer reach degree 3; keep the connected, bridgeless ones and
   deduplicate by canonical Nickel string. Two loops have two cores, the theta `111||` and the
   figure of eight `00|`; three loops have 8 and four loops 43, found in 0.4 s. The one-loop core
   is one vertex with a self-loop. Cores are computed once per $L$ and cached.
2. Subdivision. For each core and each $k \in \mathbb{Z}_{\ge 1}^{E_c}$ with $\sum_e k_e = E$
   ($k_e \ge 2$ on core self-loops unless `self_loops`), replace core edge $e$ by a chain of $k_e$
   propagators. Each new vertex gets from 1 to `max_legs_per_vertex` legs, each core vertex from
   $\max(0, 3 - \deg)$ to `max_legs_per_vertex`, $n$ in all. Subdivision keeps a graph connected
   and bridgeless, so every candidate obeys the rules; conversely every graph arises from its own
   core. With `one_vertex_irreducible=True` only the one-loop core and the loopless cores without a
   cut vertex are used.
3. Topologies. Canonicalise each candidate and keep the first of each Nickel string, with the
   labellings that attain it.
4. Colourings, below.

At two loops the theta core gives three chains of $a$, $b$, $c$ propagators between two vertices,
the family of the sunrise, the kite and both double boxes, and the figure of eight gives two
cycles through one vertex. Candidates outnumber topologies by 5.3 overall and by up to 7.4 in a
single block (three loops, $E = 10$, $n = 4$), so orderly generation, which would avoid building
duplicates, is not needed.

### Canonical labelling

The search in `nickel_index` and `cnickel` moves into one private function in
`feynkit/core/graph.py` that returns the canonical Nickel string and every labelling attaining it.
Those labellings are one of them composed with the automorphisms of the uncoloured graph, legs
included. `nickel_index` returns the string, `cnickel` minimises the colours over the labellings,
and the generator calls the same function, so its strings are those of `cnickel()` by
construction. `compute_graph_automorphisms` takes its automorphisms from the same set in place of
its own scan over $V!$ permutations, keeps those that preserve the colouring, and returns the same
list as before.

The function replaces the full scan by an exact branch and bound over BFS-consistent labellings:

1. Any vertex may take label 0, and every vertex is tried.
2. Vertices are processed in label order. When vertex $i$ is processed, its unlabelled neighbours
   take the next free labels, in decreasing order of the number of propagators joining them to
   $i$, and every order within a tie is tried. Entry $i$ is then fixed. When every labelled vertex
   has been processed, as in a disconnected graph, any unlabelled vertex may take the next label.
3. A branch is abandoned only when its fixed entries form a string strictly greater than the
   prefix of the same length of the best complete string found. Equal prefixes continue. A
   complete string equal to the best adds its labelling to the list, since the colour step needs
   every minimising labelling; a smaller one replaces the best and empties the list.

Every minimising labelling is BFS-consistent. Take a labelling that first departs from rule 2 at
vertex $i$, with the labels below $k$ held by the vertices discovered so far. Entries 0 to $i - 1$
contain only labels below $k$. Giving the new neighbours of $i$ the labels $k, k + 1, \ldots$ by the
rule, and the remaining labels to the other undiscovered vertices, leaves those entries unchanged
and makes entry $i$ strictly smaller: both versions have the same number of digits, and at the
first digit where they differ the rule gives the smaller one. So the search misses no minimiser.

Invariants such as degree and legs may only order the branches, for instance to try likely roots
first and find a good bound early; they must never remove labellings. A search restricted by them
gives a canonical form, but not this one: in 16 of the 412 three-loop topologies with 2 to 4
legs, among them `112|2|34|4e|e|`, vertex 0 does not have the largest degree, and in 108 the
degree plus legs increases somewhere along the canonical order. The canonical strings must not
change, since the database, the tests and the reports hold them.

This search gives the same string and the same set of minimising labellings as the full scan on
200 random multigraphs with up to seven vertices, some disconnected, and on all 536 topologies of
the tiers and of three loops with 2 to 4 legs, up to eight vertices. It takes 0.75 ms on the
9-gon, against 6.4 s. Its cost is at least the number of minimising labellings, the order of the
automorphism group, which legs keep small. The full scan stays in the tests as an oracle for
those graphs, and `cnickel()` must return the string the definition gives for every string in the
test suite.

### Colourings up to automorphism

For a topology with labellings $\Lambda$, a colour word assigns a code to each propagator. Its
canonical colour string is the minimum, over $\lambda \in \Lambda$ and, for `"shared"`, over
renamings of the classes, of the word read in the propagator order of $\lambda$ with parallel
propagators sorted. The generator runs over all words, $2^E$ for `"zn"` and $B_{E+1}$ for
`"shared"` (set partitions of the propagators with one block, possibly empty, marked massless;
4140 at $E = 7$), and keeps each canonical string once. Sorting parallel propagators accounts for
their permutations, so this is the orbit enumeration of colourings under the full automorphism
group acting on propagators. It costs $|\Lambda|$ string builds per word; $|\Lambda|$ is $2n$ for
the $n$-gon with $n \ge 3$, 2 for the bubble, and at most 12 in tier 2.

### Order and determinism

Output comes in blocks of $(L, E, n)$ in increasing order, sorted by code point within a block.
There is no randomness and no dependence on hash seeds or set order. A block is yielded once
complete, so memory holds one block.

## Interfaces

```python
# feynkit/generate.py
def generate_graphs(
    loops: int | Iterable[int],
    legs: int | Iterable[int],
    *,
    edges: int | Iterable[int] | None = None,
    masses: str = "zn",                      # "z", "zn" or "shared"
    self_loops: bool = False,
    max_legs_per_vertex: int | None = 1,
    one_vertex_irreducible: bool = False,
) -> Iterator[str]: ...

def mass_colourings(topology: str, *, masses: str = "zn") -> list[str]: ...
```

- `generate_graphs` yields canonical CNickel strings. `edges=None` means every $E$ the rules
  allow: $L \le E \le n + 3(L - 1)$ for $L \ge 2$, and $2 \le E \le n$ for $L = 1$ ($E = 1$ with
  self-loops).
- It checks its arguments when called, not at the first `next()`: it is a plain function that
  validates and then returns the iterator of an inner generator function. It raises
  `ValidationError` if `loops < 1`, `legs < 0`, `max_legs_per_vertex < 1`, `masses` is unknown,
  or a requested block needs more vertices than the canonical labelling allows, naming the block.
- `mass_colourings` takes a Nickel string, canonical or not, and returns the sorted canonical
  CNickel strings of its colourings, for callers that choose topologies themselves.
- Strings, not `Graph` objects: CNickel is feynkit's identifier (the database column,
  `fk analyse`, the `cnickel` field of the JSON report), strings are hashable and cheap, and a
  `Graph` carries SymPy symbols a scan may never need. `FeynmanIntegral.from_cnickel(s)` builds
  one on demand.
- `generate_graphs` is exported from `feynkit`.

Changes elsewhere:

- `feynkit/core/graph.py`, labelling: the shared function with the pruned search. With it the
  guard rises from $V \le 9$ to $V \le 10$, the limit of one-digit labels.
- `from_cnickel`: strip exactly one terminating bar. Empty entries are legal anywhere, as in
  `111e||` and `11122||e|`. After parsing, when $V > 1$, reject any vertex that has no legs and
  that no propagator touches, counting both ends of every propagator in every entry, since an
  entry lists only higher neighbours; a test on the last entry alone would reject valid strings
  such as `111e||`. Legs count as touching their vertex, so `e|e|` still parses. Input with a
  superfluous trailing bar, such as `12e|2e|e||`, accepted now, then raises. In a trial this
  parser round-tripped 1154 generated strings (tiers 1 and 2 with self-loops and $n$ from 0, three
  loops up to $E = 8$, three colourings each), 195 of them ending in `||` and 30 with an empty
  entry in the middle.
- `from_cnickel` rejects the digit codes `1` to `9`; `0` still means massless.
- `cnickel()` renames shared masses (see Colour alphabet). The box in the `nickel_index`
  docstring is corrected.
- `compute_graph_automorphisms` uses the pruned search (see Canonical labelling).
- `docs/guide.md`: a section on the generator; the CNickel section gives the code table without
  digits, the round trip for the fixed parser and the renaming of shared masses.

## Scope and estimates

| Tier | Topologies | `"zn"` | `"shared"` |
|---|---|---|---|
| 1 | 5 | 34 (3, 4, 6, 8, 13) | 185 (4, 7, 17, 37, 120) |
| 2 | 25 | 675 | 9964 |

Tier 2 by block, as topologies / `"zn"` / `"shared"`:

| $E$ | $n = 2$ | $n = 3$ | $n = 4$ |
|---|---|---|---|
| 3 | 1 / 4 / 7 | | |
| 4 | 2 / 18 / 53 | 2 / 15 / 43 | |
| 5 | 2 / 32 / 166 | 3 / 62 / 344 | 3 / 50 / 258 |
| 6 | | 3 / 83 / 929 | 5 / 183 / 2168 |
| 7 | | | 4 / 228 / 5996 |

Options on tier 2 with `"zn"`: `self_loops=True` gives 31 topologies and 785 graphs; any number
of legs per vertex gives 64 and 1338; $n = 0$ and 1 add 3 topologies (`111||`, `111e||`,
`112|2|e|`) and 17 graphs. With self-loops one loop also has `0e|` at $n = 1$, two graphs, outside
tier 1.

Run time:

- Generation. Tier 1 takes milliseconds. Tier 2 builds 88 candidates with $V \le 6$; the full
  scan costs about 8 ms per candidate and the pruned search a fraction of a millisecond, so either
  way generation takes under a second. The colourings add a fraction of a second, a few seconds
  for `"shared"`; the prototype took 0.1 s for `"zn"` and 0.7 s for `"shared"`.
- Analysis dominates. Building the integral, the support of $G$ and `polytope_data` took 0.11 s
  per graph on average over tiers 1 and 2, 0.3 s at seven propagators: about 5 s for tier 1 and
  75 s for tier 2 with `"zn"`. A `"shared"` graph has the Newton polytope of its `"zn"` image
  (letters replaced by `n`), so data that depend only on the support are computed once per image.
  Landau analyses, toric ideals and point counts cost far more per graph and are budgeted by
  their own notes.

## Performance and reach

The full scan builds one string per permutation. On the $V$-gon `Graph.cnickel()` takes 8 ms at
$V = 6$, and each further vertex multiplies the time by about $V$:

| $V$ | 6 | 7 | 8 | 9 |
|---|---|---|---|---|
| Relative to $V = 6$ | 1 | 8 | 70 | 700 |

So $V = 9$ takes about 6 s and $V = 10$ would take about a minute. `compute_graph_automorphisms`
makes the same scan on every call.

With the full scan, the reach is limited by the number of vertices: the generator runs the scan
once per candidate. By the bound above, $V$ reaches 9 at one loop with 9 legs, two loops with 7,
three loops with 5 and four loops with 3, where the current guard refuses one more leg. $V = 10$ is
the limit of one-digit labels, and $V \ge 11$ needs labels of more than one character, a change of
grammar.

| Range | Largest $V$ | Candidates | Topologies | `"zn"` graphs | Generation, full scan |
|---|---|---|---|---|---|
| Tier 1 | 6 | 5 | 5 | 34 | milliseconds |
| Tier 2 | 6 | 88 | 25 | 675 | under a second |
| Three loops, 2 to 4 legs | 8 | 2203 | 412 | 74480 | about 3 minutes |

The three-loop row was reproduced by the independent enumeration, topologies and `"zn"` counts
alike. Its 252 candidates with $V = 8$ take most of the generation time with the full scan; at
$V = 9$ a range of a few hundred candidates takes up to an hour. The pruned search removes that
cost within the notation's reach, and the analysis becomes the limit: 74480 graphs at half a
second or more each take ten hours or more.

## Validation

### Independent enumeration

A brute force shares only the rules with the generator: no cores, no subdivision, no Nickel
strings, no lexicographic minimum.

1. Topologies. For each block, with $V = E - L + 1$, run over the leg vectors in non-increasing
   order within the limits; for each, over the vectors of internal degrees with every entry at
   least 2 and at least 3 minus the legs there, summing to $2E$ and non-increasing among vertices
   with equal legs; and for each, over the symmetric multiplicity matrices that realise it, filled
   row by row (a diagonal entry counts twice, and is zero unless self-loops are allowed). Keep the
   connected, bridgeless graphs, a bridge being a bridge of the simple graph underneath that
   carries one propagator, found with networkx, and deduplicate with `nx.is_isomorphic` on
   `nx.MultiGraph`, matching leg counts on nodes, within buckets of equal invariants (each
   vertex's degree, legs and self-loops, with those of its neighbours).
2. Colourings. Take the automorphisms of each representative from `GraphMatcher` on its
   subdivided graph (one node per propagator, joined to its ends, self-loop nodes marked), so that
   parallel propagators swap, and count the orbits of colour words, renaming classes in order of
   first appearance for `"shared"`; Burnside's lemma gives the same counts.
3. Compare. Each generated string, parsed, is isomorphic to exactly one representative, legs
   matched, through an isomorphism of the subdivided graphs; carried over by it, its colouring
   lies in one orbit, and every orbit of every representative is hit exactly once. At three loops
   the topologies are matched this way and the `"zn"` counts compared block by block with
   Burnside's lemma, since the 74480 strings are too many to match one by one.

networkx is already a dependency. This enumeration agrees with the generator in every block of
tiers 1 and 2, from $n = 0$ (topologies with and without self-loops and with any number of legs
per vertex, colourings for both alphabets), and in every three-loop block with 2 to 4 legs. The
three-loop enumeration takes about 13 s, 10 s of it for the block with eight vertices
($E = 10$, $n = 4$), which is marked `slow`. So are the three-loop totals, the full scan at seven
and eight vertices and the `"shared"` round trips with self-loops or any number of legs, which
take 6 to 28 s each; the default run of the validation takes about 30 s.

### Hand checks

With the default options:

- One loop: one topology per $n \ge 2$, the $n$-gon. Its automorphisms permute the propagators as
  the dihedral group of order $2n$ permutes the sides of a polygon, so by Burnside the `"zn"`
  count is the number of binary bracelets,
  $$\frac{1}{2n}\sum_{d \mid n} \varphi(d)\, 2^{n/d} +
  \begin{cases} 2^{(n-1)/2} & n \text{ odd} \\ 3 \cdot 2^{n/2 - 2} & n \text{ even,} \end{cases}$$
  3, 4, 6, 8 and 13 for $n = 2$ to 6.
- Two loops: every graph is a theta, three chains of $a, b, c \ge 1$ propagators between two
  vertices, or a figure of eight, two cycles of $a, b \ge 2$ propagators through one vertex. Every
  inner vertex of a chain or cycle carries a leg; the two ends of a theta carry $k \in \{0, 1, 2\}$
  legs between them, and reversing all chains makes the choice of end irrelevant when $k = 1$; the
  shared vertex of a figure of eight carries $k \in \{0, 1\}$. Hence
  $$N_2(E, n) = p_3(E)\,[0 \le n - E + 3 \le 2] + q(E)\,[0 \le n - E + 2 \le 1],$$
  with $p_3(E)$ the number of partitions of $E$ into three positive parts, $q(E)$ that into two
  parts of at least 2, and $[\cdot]$ equal to 1 when the condition holds and 0 otherwise. The
  formula matched the prototypes on 90 blocks. It gives 5, 8 and 12 topologies for
  $n = 2, 3, 4$:
  - $n = 2$: `111e|e|` (sunrise), `112e|2|e|` (chains 1, 1, 2), `1122|e|e|` (two bubbles
    sharing a vertex), `112|3|3e|e|` (1, 1, 3, a bubble inserted in a bubble), `123|23|e|e|`
    (1, 2, 2, the kite);
  - $n = 3$: `112e|2e|e|` (1, 1, 2, both ends with legs), `1122e|e|e|` (two bubbles, leg on the
    shared vertex), `112e|3|3e|e|` (1, 1, 3, one end with a leg), `123e|23|e|e|` (the kite with a
    leg at one end), `1123|e|3e|e|` (bubble and triangle sharing a vertex), `112|3|4e|4e|e|`
    (1, 1, 4), `123|24|e|4e|e|` (1, 2, 3), `123|4e|4e|4e||` (2, 2, 2);
  - $n = 4$, $E = 7$: `123|45|4e|5e|e|e|` (1, 3, 3, planar double box), `123|4e|4e|5e|5|e|`
    (2, 2, 3, non-planar double box), `123|24|e|5e|5e|e|` (1, 2, 4), `112|3|4e|5e|5e|e|`
    (1, 1, 5).
- Colourings by Burnside: the sunrise has 4 (multisets of three codes); the kite's automorphisms
  fix the rung and form a group of order 4 whose three non-identity elements each have three
  cycles on the propagators, giving $(2^5 + 3 \cdot 2^3)/4 = 14$; the chains 1, 1, 3 give
  $(32 + 16 + 16 + 8)/4 = 18$.

### Published counts

The one-loop `"zn"` counts are the binary bracelet numbers, OEIS A000029, "Number of necklaces
with n beads of 2 colors, allowing turning over (these are also called bracelets)": 1, 2, 3, 4, 6,
8, 13, 18, 30, ... from $n = 0$. Published counts of 1PI topologies generally follow other
conventions, such as a fixed vertex degree for a given theory, labelled legs, or scaleless graphs
removed. A000029, the brute force and the closed forms above are the acceptance sources, and
no other published count is used.

### Acceptance

- The brute force and the generator agree on tiers 1 and 2, for both alphabets, with and without
  self-loops, and with at most one leg per vertex as well as with no limit; and on three loops with
  2 to 4 legs, the block with eight vertices under `slow`.
- The closed forms and the lists above hold.
- Every emitted string, for $n$ from 0 upwards, round-trips through `from_cnickel` and `cnickel()`,
  and the parsed graph has the requested $L$, $E$, $n$ and $V$, is connected and bridgeless, and
  meets the degree rule.
- Two runs give identical output, also under a different `PYTHONHASHSEED`.

## Property-based tests

`hypothesis` joins the `dev` group and `uv.lock` as a development dependency.

### Strategy

The pool is the tier 1 graphs and the tier 2 graphs with $E \le 6$ generated with
`self_loops=True`: 591 strings with `"zn"`, 110 of them with a self-loop. The CNickel test also
draws from the 936 `"shared"` graphs of both tiers with $E \le 5$. A composite strategy draws a
string with `sampled_from`, then `permutations` of the $V$ vertices and of the $E$ propagator
indices, and builds the relabelled `Graph` directly from `Edge` objects: propagators keep their
mass symbols, endpoints are mapped, and legs stay in order on the images of their vertices. As
the pool is ordered by $(L, E, n)$, shrinking moves towards the smallest graphs and the identity
permutations. Test helpers `relabel`, `delete_edge` and `contract_edge` build graphs with edge
indices, masses and legs kept; contraction merges the two ends, moves their legs to the merged
vertex and turns propagators parallel to $e$ into self-loops. The two graph operations stay test
helpers.

### Assertions

For a graph $\Gamma$ from the pool and its relabelling $\Gamma'$:

1. `cnickel()` of $\Gamma'$ is the drawn string.
2. The Newton polytope of $\Gamma'$ is that of $\Gamma$ with coordinates permuted, and
   `polytope_data` gives equal `normalized_volume` and `f_vector`.
3. `is_unimodular_equivalent_to` holds between the two integrals, for the graphs with at most
   five propagators and without a massless self-loop.
4. Deletion-contraction: for a drawn propagator $e$ that is not a self-loop,
   $$U_\Gamma = a_e\,U_{\Gamma \setminus e} + U_{\Gamma / e}$$
   exactly, as polynomials in the Schwinger parameters $a_{idx}$. Spanning trees without $e$ give
   the first term and those containing $e$ are the spanning trees of $\Gamma / e$. For a
   self-loop, $U_\Gamma = a_e\,U_{\Gamma \setminus e}$; the pool's self-loop graphs exercise this
   case. As $\Gamma$ is 1PI, $\Gamma \setminus e$ is connected.
5. Coordinate face: for $e$ not a self-loop, with Lee-Pomeransky parameters $u_{idx}$,
   $$G_\Gamma\big|_{u_e = 0} = G_{\Gamma / e}$$
   exactly, with $\Gamma / e$ built with the momentum products of $\Gamma$. Spanning trees and
   spanning 2-forests containing $e$ are those of $\Gamma / e$, with the same momenta on each
   side, and the mass term restricts to $U_{\Gamma/e} \sum_{e' \ne e} m_{e'}^2 u_{e'} / \mu^2$.
   As $\alpha_e \ge 0$ on the polytope, the points with $\alpha_e = 0$ span a face: their index
   set is among `polytope_data(points).faces`, with the dimension of the Newton polytope of
   $G_{\Gamma/e}$.

A prototype checked assertions 4 and 5 on every propagator, and 1 and 2 on one relabelling, of
all 194 graphs of tiers 1 and 2 with $E \le 5$ (tier 1 up to the box), and the self-loop form of
assertion 4 on two graphs with self-loops.

### Budget and acceptance

Hypothesis has a built-in `ci` profile, loaded automatically when the `CI` environment variable
is set, as it is on GitHub Actions: `derandomize=True`, `database=None`, `deadline=None`,
`print_blob=True`, and the too-slow health check suppressed. `tests/conftest.py` builds on it:

```python
settings.register_profile("feynkit", parent=settings.get_profile("ci"), max_examples=25)
settings.register_profile("feynkit-dev", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "feynkit"))
```

Local runs and CI then draw the same examples. Derandomised runs repeat those examples until the
tests, the code under test or the environment (Python, `hypothesis` or another locked dependency)
changes. `HYPOTHESIS_PROFILE=feynkit-dev` gives a larger random search locally.

An example costs about 0.1 s at $E \le 5$, 0.3 s at $E = 6$ and 0.5 s at $E = 7$. With the fixed
examples below, whose graphs with seven propagators take half the time, the tests take about a
minute. Assertion 3 draws from $E \le 5$ only: `is_unimodular_equivalent` took 11 s on the
massless planar double box. Acceptance is that the tests pass under the `feynkit` profile on both
CI Python versions.

A fixed list is attached as `@example` entries from the start: the 13 tier 1 graphs up to the box
and 18 two-loop graphs (the five $n = 2$ topologies and the four with $n = 4$, $E = 7$ listed
above, massless and fully massive), each with three relabellings drawn once from
`random.Random(0)`. Assertion 3 uses those with $E \le 5$.

### Assertion 3 and the exact vertices

Assertion 3 fails on the current code. `is_unimodular_equivalent` restricts both configurations to
the vertices that `hull_vertex_indices` in `normal_forms/_invariants.py` takes from scipy's
`ConvexHull`, which the exact core left in place. Qhull sometimes reports non-extreme points as
vertices, and which ones depends on the order of the coordinates and of the points. For the
massive non-planar double box `123|4e|4e|5e|5|e|:nnnnnnz` the exact core finds 52 vertices, as
does a linear-programming test of each point; the floating hull finds 53, and 57 after one
relabelling of the propagators, so the test answers "not equivalent" for a coordinate
permutation.

- With one relabelling per graph, the floating counts of the two orders differed on 166 of the
  709 `"zn"` graphs of tiers 1 and 2, all but one of them with seven propagators. Each such
  difference is a false negative, since the test compares vertex counts first.
- More relabellings reach smaller graphs: `123|24|e|4e|e|:znnnnn` ($E = 6$) fails under some.
  With coordinates and points in arbitrary order, the floating count is wrong for 48 of the 576
  colour words at $E = 6$ and for 2 of the 288 at $E = 5$.

The equivalence test also needs a full-dimensional polytope. A massless self-loop is scaleless,
and its Newton polytope is not full-dimensional: the 35 graphs of the pool with at most five
propagators and a massless self-loop are "not equivalent" even to themselves. Assertion 3 leaves
them out and draws from the other 237.

The same routines feed the polytope automorphisms and the report's symmetry section. A separate
branch, `exact-automorphisms`, moves `_invariants` onto the exact core and works in the lattice
chart below full dimension; assertion 3 depends on it. Until it lands, two tests carry
`pytest.mark.xfail(strict=True)`: one asserts equivalence for the double box under a failing
relabelling, the other for `012e|3e|3e|e|:znnnn` with itself. The fix, or any change in Qhull's
output, then shows up as a failure until the marks are removed.

## Out of scope

- Kinematic assignments that tell legs apart, such as on-shell and off-shell legs or external
  masses, and their enumeration up to automorphism.
- Numerators and raised propagator powers; vertices of degree 2 without legs are excluded by the
  degree rule.
- Graphs with more than ten vertices, which need labels of more than one character.
- Filtering scaleless graphs.
- Moving `_invariants` onto the exact core and equivalences below full dimension, a separate
  branch on which assertion 3 depends.
- Orderly generation.
- A command line entry, `fk generate`.

## Decisions

- Validation: OEIS A000029, the independent enumeration and the closed forms above. No published
  table of topologies is used.
- Vacuum and one-leg graphs are generated only on request, and are included in the validation and
  round-trip tests. The degree rule has no exception for `0|`.
- Self-loops are off by default; massless ones are kept when they are on.
- Cut vertices are kept by default.
- Both tiers use the `"zn"` colourings: 34 one-loop and 675 two-loop graphs.
- `cnickel()` itself renames shared masses in order of first appearance and writes a label on a
  single propagator as `n`. This is a breaking change.
- The parser rejects the digit mass codes `1` to `9`. This is a breaking change.
- The pruned search replaces the $V!$ scan in `nickel_index`, `cnickel` and
  `compute_graph_automorphisms` and returns exactly the same minimum. The guard rises to
  $V \le 10$. The full scan stays in the tests as an oracle.
- The parser strips exactly one trailing `|`, allows empty entries anywhere, and rejects a vertex
  with neither legs nor propagators, counting the propagators of every entry.
- `hypothesis` is a development dependency.
- Assertion 3 depends on the `exact-automorphisms` branch; until it lands, its known failures carry
  a strict `xfail`.
- The three-loop validation block with eight vertices is marked `slow`, with a few other checks
  of several seconds each.
- `delete_edge` and `contract_edge` stay test helpers.
- There is no `fk generate` for now.
