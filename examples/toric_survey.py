"""
Systematic toric ideal survey — resumable.

Runs a large family of Feynman diagrams through the full GKZ + toric-ideal
pipeline and stores every result in a SQLite database.  The script is safe
to interrupt at any point: each record is committed atomically before the
next diagram starts, so restarting simply skips everything already stored.

Usage
-----
    uv run python examples/toric_survey.py [db_path]

Default db_path: feynkit_survey.db
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import sympy as sp

from feynkit import Edge, FeynkitDatabase, FeynmanIntegral, Graph

# ── diagram constructors ──────────────────────────────────────────────────────


def _polygon(n: int, n_masses: int, db: FeynkitDatabase) -> FeynmanIntegral:
    """Massless or partially-massive 1-loop polygon with n propagators."""
    nus = sp.symbols(f"nu1:{n + 1}", positive=True)
    ms = sp.symbols(f"m1:{n + 1}", nonnegative=True) if n_masses else []

    def mass(i: int) -> sp.Expr:
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

    return FeynmanIntegral(
        g,
        propagator_exponents={i + 1: nus[i] for i in range(n)},
        database=db,
    )


def _banana(n_props: int, *, massive: bool, db: FeynkitDatabase) -> FeynmanIntegral:
    """Banana / sunrise graph: n_props parallel propagators, (n_props−1)-loop."""
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


# ── diagram families ──────────────────────────────────────────────────────────


def build_families(db: FeynkitDatabase) -> list[tuple[str, list[tuple[str, FeynmanIntegral]]]]:
    """
    Return the full list of diagram families.

    Each family is (family_name, [(label, FeynmanIntegral), ...]).
    Constructing a FeynmanIntegral is cheap (lazy); the expensive work
    only happens when toric_ideal is first accessed.
    """
    names_poly = {
        2: "bubble",
        3: "triangle",
        4: "box",
        5: "pentagon",
        6: "hexagon",
        7: "heptagon",
        8: "octagon",
        9: "nonagon",
    }

    families: list[tuple[str, list[tuple[str, FeynmanIntegral]]]] = []

    # 1. Massless L-gons
    families.append(
        (
            "massless polygons  (L = 2 … 9)",
            [(f"massless {names_poly[n]}  ({n} props)", _polygon(n, 0, db)) for n in range(2, 10)],
        )
    )

    # 2–7. Each polygon with masses added one at a time (skip k=0, already above)
    for n in range(3, 9):
        name = names_poly[n]
        families.append(
            (
                f"{name}: adding masses  (k = 1 … {n})",
                [(f"{name}, {k}/{n} massive", _polygon(n, k, db)) for k in range(1, n + 1)],
            )
        )

    # 8. Massless banana  (n = 3 … 9 props)
    #    n=2 massless = massless bubble, already in family 1
    families.append(
        (
            "massless banana  (n = 3 … 9 props)",
            [
                (f"massless banana  {n} props  ({n - 1}-loop)", _banana(n, massive=False, db=db))
                for n in range(3, 10)
            ],
        )
    )

    # 9. Massive banana  (n = 2 … 9 props)
    families.append(
        (
            "massive banana  (n = 2 … 9 props)",
            [
                (f"massive banana  {n} props  ({n - 1}-loop)", _banana(n, massive=True, db=db))
                for n in range(2, 10)
            ],
        )
    )

    return families


# ── output helpers ────────────────────────────────────────────────────────────

W = 74


def _hdr(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(2, W - 6 - len(title))}", flush=True)


def _row_cached(idx: int, total: int, label: str, rec) -> None:
    tag = f"[{idx}/{total}]"
    print(
        f"  {tag:<8} {label:<42}  "
        f"A={rec.n_rows}×{rec.n_cols:<4}  {rec.n_toric_gens:>5} gens  [cached]",
        flush=True,
    )


def _row_stored(idx: int, total: int, label: str, rec, elapsed: float) -> None:
    tag = f"[{idx}/{total}]"
    print(
        f"  {tag:<8} {label:<42}  "
        f"A={rec.n_rows}×{rec.n_cols:<4}  {rec.n_toric_gens:>5} gens  {elapsed:.2f}s",
        flush=True,
    )


# ── equivalence analysis ──────────────────────────────────────────────────────


def equivalence_analysis(db: FeynkitDatabase) -> None:
    """
    Group all stored integrals into unimodular-equivalence classes and print
    the non-trivial classes (size ≥ 2).  All pairwise checks within each
    shape class are run (and cached), so this pass is idempotent.
    """
    _hdr("UNIMODULAR EQUIVALENCE ANALYSIS")
    records = db.all_integrals()
    n = len(records)
    print(f"  Checking {n} integral(s) …", flush=True)

    # union-find over fingerprints
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
        print(f"  checking equivalences for [{i}/{n}] {rec.label} …", end="\r", flush=True)
        matches = db.find_equivalent_record(rec, relation="unimodular")
        for m in matches:
            union(rec.fingerprint, m.fingerprint)

    print(" " * (W + 4), end="\r")  # clear the \r line

    # group records by equivalence class root
    classes: dict[str, list] = defaultdict(list)
    for rec in records:
        classes[find(rec.fingerprint)].append(rec)

    non_trivial = [cls for cls in classes.values() if len(cls) > 1]
    trivial_count = sum(1 for cls in classes.values() if len(cls) == 1)

    print(
        f"  {len(classes)} class(es) total  "
        f"({len(non_trivial)} non-trivial, {trivial_count} singletons)\n",
        flush=True,
    )

    for cls in sorted(non_trivial, key=lambda c: -len(c)):
        rep = cls[0]
        labels = ", ".join(r.label or r.fingerprint[:8] for r in cls)
        print(f"  [{len(cls)} equivalent]  A={rep.n_rows}×{rep.n_cols}  {labels}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("feynkit_survey.db")

    print(flush=True)
    print("=" * W, flush=True)
    print("  FEYNKIT TORIC IDEAL SURVEY", flush=True)
    print(f"  database: {db_path}", flush=True)
    print("=" * W, flush=True)

    with FeynkitDatabase(db_path) as db:

        # Preload all completed records by label so that cached runs never
        # trigger polynomial computation just to find a fingerprint.
        label_cache: dict[str, object] = {
            rec.label: rec
            for rec in db.all_integrals()
            if rec.n_toric_gens is not None and rec.label is not None
        }

        families = build_families(db)

        # count total diagrams for the progress tag
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
