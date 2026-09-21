"""
Equivalence-rich toric survey, resumable.

For each n-gon (n = 3, 4, 5, and 6 under --all) we insert EVERY mass placement:
all C(n,k) ways of making exactly k out of n propagators massive, for
k = 0, ..., n.  Propagators in each variant carry a common symbol m so the
Newton polytope depends only on WHICH edges are massive, not on the
coefficient values.

Within each (n, k) family, all C(n,k) variants are unimodularly equivalent:
they are related by a permutation of the Lee-Pomeransky parameters (a
permutation matrix is unimodular with det = +/-1).  The equivalence analysis
at the end therefore recovers one non-trivial equivalence class of size C(n,k)
for each k in {1, ..., n-1}, and singletons for the massless (k=0) and
all-massive (k=n) cases.

We also include massless and massive banana families for additional coverage.

Expected non-trivial classes (size >= 2):
  triangle:  k=1 -> 3,  k=2 -> 3
  box:       k=1 -> 4,  k=2 -> 6,  k=3 -> 4
  pentagon:  k=1 -> 5,  k=2 -> 10, k=3 -> 10, k=4 -> 5
  hexagon:   k=1 -> 6,  k=2 -> 15, k=3 -> 20, k=4 -> 15, k=5 -> 6

The hexagon adds 64 of the 130 variants and dominates both the toric-ideal
pass and the pairwise equivalence analysis, so it sits behind --all.  It also
needs 4ti2 (install via pacman/brew/apt); the smaller polygons do not.

Usage
-----
    uv run python examples/equivalence_survey.py [db_path]          # n = 3, 4, 5
    uv run python examples/equivalence_survey.py --all [db_path]    # also n = 6

Default db_path: examples/output/feynkit_equiv_survey.db
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import sympy as sp

from feynkit import Edge, FeynkitDatabase, FeynmanIntegral, Graph

# -- diagram constructors ------------------------------------------------------


def _polygon(n: int, mass_set: frozenset[int], db: FeynkitDatabase) -> FeynmanIntegral:
    """
    n-gon whose propagators with 0-based indices in mass_set carry a mass m.
    The common mass symbol means the Newton polytope depends only on the
    combinatorial placement, not on individual mass values.
    """
    nus = sp.symbols(f"nu1:{n + 1}", positive=True)
    m = sp.Symbol("m", nonnegative=True)

    def mass(i: int) -> sp.Expr:
        return m if i in mass_set else sp.Integer(0)

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

    return FeynmanIntegral(
        g,
        propagator_exponents={i + 1: nus[i] for i in range(n)},
        database=db,
    )


def _banana(n_props: int, *, massive: bool, db: FeynkitDatabase) -> FeynmanIntegral:
    """Banana / sunrise: n_props parallel propagators, (n_props-1)-loop."""
    nus = sp.symbols(f"nu1:{n_props + 1}", positive=True)
    ms = sp.symbols(f"m1:{n_props + 1}", nonnegative=True)

    def mass(i: int) -> sp.Expr:
        return ms[i] if massive else sp.Integer(0)

    edges = [
        Edge(idx=i + 1, v1=1, v2=2, is_internal=True, mass=mass(i), nu=nus[i])
        for i in range(n_props)
    ] + [
        Edge(idx=n_props + 1, v1=1, v2=3, is_internal=False),
        Edge(idx=n_props + 2, v1=2, v2=4, is_internal=False),
    ]
    g = Graph(internal_vertices=2, external_legs=2, edges=edges)
    return FeynmanIntegral(
        g,
        propagator_exponents={i + 1: nus[i] for i in range(n_props)},
        database=db,
    )


# -- diagram families ----------------------------------------------------------

_POLY_NAMES = {2: "bubble", 3: "triangle", 4: "box", 5: "pentagon", 6: "hexagon"}


def _mass_label(mass_set: frozenset[int], n: int) -> str:
    if not mass_set:
        return "massless"
    if len(mass_set) == n:
        return "all-massive"
    edges = ",".join(str(i + 1) for i in sorted(mass_set))
    return f"k={len(mass_set)} edges=[{edges}]"


def build_families(
    db: FeynkitDatabase, *, include_hexagon: bool
) -> list[tuple[str, list[tuple[str, FeynmanIntegral]]]]:
    """
    Return all diagram families for the survey.

    Polygon section: for each n in {3,4,5} (plus n=6 when include_hexagon is
    set) and each k in {0,...,n}, all C(n,k) mass placements.  Within each
    (n,k) group every variant is unimodularly equivalent.

    Banana section: massless n=3..7 and massive n=2..6.
    """
    families: list[tuple[str, list[tuple[str, FeynmanIntegral]]]] = []

    # -- Polygons: n = 3, 4, 5, and 6 on request ------------------------------
    for n in (3, 4, 5, 6) if include_hexagon else (3, 4, 5):
        name = _POLY_NAMES[n]
        for k in range(n + 1):
            placements = list(combinations(range(n), k))
            c_nk = len(placements)
            family_title = (
                f"{name}  k={k}  "
                f"({'massless' if k == 0 else 'all-massive' if k == n else f'C({n},{k})={c_nk} placements'})"
            )
            items = [
                (
                    f"{name}  {_mass_label(frozenset(ms), n)}",
                    _polygon(n, frozenset(ms), db),
                )
                for ms in placements
            ]
            families.append((family_title, items))

    # -- Bananas ---------------------------------------------------------------
    families.append(
        (
            "massless banana  n = 3 ... 7  (massless bubble already in polygons)",
            [
                (f"massless banana  {n} props  ({n-1}-loop)", _banana(n, massive=False, db=db))
                for n in range(3, 8)
            ],
        )
    )
    families.append(
        (
            "massive banana  n = 2 ... 6",
            [
                (f"massive banana  {n} props  ({n-1}-loop)", _banana(n, massive=True, db=db))
                for n in range(2, 7)
            ],
        )
    )

    return families


# -- output helpers ------------------------------------------------------------

W = 74


def _hdr(title: str) -> None:
    print(f"\n  -- {title} {'-' * max(2, W - 6 - len(title))}", flush=True)


def _row_cached(idx: int, total: int, label: str, rec) -> None:
    tag = f"[{idx}/{total}]"
    print(
        f"  {tag:<8} {label:<42}  "
        f"A={rec.n_rows} x {rec.n_cols:<4}  {rec.n_toric_gens:>5} gens  [cached]",
        flush=True,
    )


def _row_stored(idx: int, total: int, label: str, rec, elapsed: float) -> None:
    tag = f"[{idx}/{total}]"
    print(
        f"  {tag:<8} {label:<42}  "
        f"A={rec.n_rows} x {rec.n_cols:<4}  {rec.n_toric_gens:>5} gens  {elapsed:.2f}s",
        flush=True,
    )


# -- equivalence analysis ------------------------------------------------------


def equivalence_analysis(db: FeynkitDatabase) -> None:
    """
    Union-find over all stored integrals; report non-trivial unimodular
    equivalence classes.  All pairwise checks are cached in the database so
    this pass is idempotent.
    """
    _hdr("UNIMODULAR EQUIVALENCE ANALYSIS")
    records = db.all_integrals()
    n = len(records)
    print(f"  Checking {n} integral(s) ...", flush=True)

    parent: dict[str, str] = {r.fingerprint: r.fingerprint for r in records}

    def find(fp: str) -> str:
        while parent[fp] != fp:
            parent[fp] = parent[parent[fp]]
            fp = parent[fp]
        return fp

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i, rec in enumerate(records, 1):
        print(f"  [{i}/{n}] {rec.label} ...", end="\r", flush=True)
        for m in db.find_equivalent_record(rec, relation="unimodular"):
            union(rec.fingerprint, m.fingerprint)

    print(" " * (W + 4), end="\r")

    classes: dict[str, list] = defaultdict(list)
    for rec in records:
        classes[find(rec.fingerprint)].append(rec)

    non_trivial = [c for c in classes.values() if len(c) > 1]
    trivial_count = sum(1 for c in classes.values() if len(c) == 1)

    print(
        f"  {len(classes)} class(es) total  "
        f"({len(non_trivial)} non-trivial, {trivial_count} singletons)\n",
        flush=True,
    )

    for cls in sorted(non_trivial, key=lambda c: -len(c)):
        rep = cls[0]
        labels = ", ".join(r.label or r.fingerprint[:8] for r in cls)
        print(
            f"  [{len(cls):>2} equiv]  A={rep.n_rows} x {rep.n_cols}  {labels}",
            flush=True,
        )


# -- main ----------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument(
        "db_path",
        nargs="?",
        type=Path,
        help="database to resume from (default: examples/output/feynkit_equiv_survey.db)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also survey the hexagon variants (needs 4ti2; adds a couple of minutes)",
    )
    return parser.parse_args()


def _default_db_path() -> Path:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "feynkit_equiv_survey.db"


def main() -> None:
    args = _parse_args()
    db_path = args.db_path if args.db_path is not None else _default_db_path()

    print(flush=True)
    print("=" * W, flush=True)
    print("  FEYNKIT EQUIVALENCE-RICH SURVEY", flush=True)
    print(f"  database: {db_path}", flush=True)
    if not args.all:
        print("  hexagon variants skipped; rerun with --all to survey n = 6 too", flush=True)
    print("=" * W, flush=True)

    with FeynkitDatabase(db_path) as db:

        # Preload all completed records by label so that cached runs never
        # trigger polynomial computation just to find a fingerprint.
        label_cache: dict[str, object] = {
            rec.label: rec
            for rec in db.all_integrals()
            if rec.n_toric_gens is not None and rec.label is not None
        }

        families = build_families(db, include_hexagon=args.all)

        all_items = [(lbl, fi) for _, diags in families for lbl, fi in diags]
        total = len(all_items)
        global_idx = 0

        for family_name, diagrams in families:
            _hdr(family_name)
            for label, fi in diagrams:
                global_idx += 1
                rec = label_cache.get(label)
                if rec is not None:
                    _row_cached(global_idx, total, label, rec)
                    continue

                t0 = time.perf_counter()
                rec = db.store(fi, label=label)
                label_cache[label] = rec
                _row_stored(global_idx, total, label, rec, time.perf_counter() - t0)

        print(flush=True)
        print(db.summary(), flush=True)

        equivalence_analysis(db)

    print(flush=True)
    print("=" * W, flush=True)
    print("  DONE", flush=True)
    print("=" * W, flush=True)


if __name__ == "__main__":
    main()
