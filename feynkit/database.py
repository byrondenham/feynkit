"""
Persistent SQLite store for Feynman integral GKZ analysis.

Usage
-----
>>> from feynkit import FeynmanIntegral, FeynkitDatabase
>>>
>>> db = FeynkitDatabase("analysis.db")
>>> fi = FeynmanIntegral(graph, database=db)   # auto-checks on every property
>>> ti = fi.toric_ideal                         # loaded from db if present
>>>
>>> # Explicit API
>>> record = db.store(fi, label="massless triangle")
>>> record = db.lookup(fi)                      # None if not stored
>>> matches = db.find_equivalent(fi)            # unimodular equivalence search
>>> print(db.summary())
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from .integral import FeynmanIntegral


_SCHEMA = """
CREATE TABLE IF NOT EXISTS integrals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint      TEXT    NOT NULL UNIQUE,
    a_matrix         TEXT    NOT NULL,
    newton_points    TEXT    NOT NULL,
    n_rows           INTEGER NOT NULL,
    n_cols           INTEGER NOT NULL,
    loop_count       INTEGER,
    n_props          INTEGER,
    n_ext            INTEGER,
    n_toric_gens     INTEGER,
    toric_gens       TEXT,
    is_binomial      INTEGER,
    cnickel          TEXT,
    label            TEXT,
    poly_aut_order   INTEGER,
    graph_aut_order  INTEGER,
    coeff_pres_order INTEGER,
    vertex_orbits    TEXT,
    stored_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS equivalences (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint_a  TEXT    NOT NULL,
    fingerprint_b  TEXT    NOT NULL,
    relation       TEXT    NOT NULL,
    equivalent     INTEGER NOT NULL,
    witness_map    TEXT,
    checked_at     TEXT    NOT NULL,
    UNIQUE(fingerprint_a, fingerprint_b, relation)
);
"""


@dataclass
class IntegralRecord:
    """A stored Feynman integral record retrieved from the database."""

    id: int
    fingerprint: str
    a_matrix: sp.Matrix
    newton_points: list[tuple[int, ...]]
    n_rows: int
    n_cols: int
    loop_count: int | None
    n_props: int | None
    n_ext: int | None
    n_toric_gens: int | None
    toric_generators: list[sp.Expr] | None
    is_binomial: bool | None
    cnickel: str | None
    label: str | None
    poly_aut_order: int | None
    graph_aut_order: int | None
    coeff_pres_order: int | None
    vertex_orbits: list[list[int]] | None
    stored_at: str

    def __repr__(self) -> str:
        label = f" [{self.label!r}]" if self.label else ""
        gens = f", {self.n_toric_gens} gens" if self.n_toric_gens is not None else ""
        cn = f", cnickel={self.cnickel!r}" if self.cnickel else ""
        aut = f", |Aut(P)|={self.poly_aut_order}" if self.poly_aut_order is not None else ""
        return (
            f"IntegralRecord(id={self.id}, A={self.n_rows}×{self.n_cols}"
            f"{gens}{cn}{aut}{label}, stored={self.stored_at[:10]})"
        )


class FeynkitDatabase:
    """
    SQLite-backed store for Feynman integral GKZ analysis.

    Stores the GKZ A-matrix, Newton polytope points, and toric ideal
    generators for each analysed integral.  Equivalence checks are
    cached in a separate table so repeated comparisons are free.

    Parameters
    ----------
    path
        Path to the SQLite database file.  Created if it does not exist.
        Use ``":memory:"`` for an in-process store (not persisted to disk).
    """

    def __init__(self, path: str | Path = "feynkit.db") -> None:
        self._path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    # ── schema migration ──────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Add columns introduced after the initial schema (idempotent)."""
        new_cols = [
            ("cnickel", "TEXT"),
            ("poly_aut_order", "INTEGER"),
            ("graph_aut_order", "INTEGER"),
            ("coeff_pres_order", "INTEGER"),
            ("vertex_orbits", "TEXT"),
        ]
        for col, coltype in new_cols:
            try:
                self._conn.execute(f"ALTER TABLE integrals ADD COLUMN {col} {coltype}")
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already present

    # ── serialisation helpers ─────────────────────────────────────────────

    @staticmethod
    def _fingerprint(points: list[tuple[int, ...]]) -> str:
        """SHA-256 of the sorted Newton-polytope point list (canonical)."""
        blob = json.dumps(sorted(points), separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _ser_matrix(m: sp.Matrix) -> str:
        return json.dumps([[int(m[i, j]) for j in range(m.cols)] for i in range(m.rows)])

    @staticmethod
    def _deser_matrix(s: str) -> sp.Matrix:
        return sp.Matrix(json.loads(s))

    @staticmethod
    def _ser_points(pts: list[tuple[int, ...]]) -> str:
        return json.dumps([list(p) for p in pts])

    @staticmethod
    def _deser_points(s: str) -> list[tuple[int, ...]]:
        return [tuple(p) for p in json.loads(s)]

    @staticmethod
    def _ser_exprs(exprs: list[sp.Expr]) -> str:
        return json.dumps([sp.srepr(e) for e in exprs])

    @staticmethod
    def _deser_exprs(s: str) -> list[sp.Expr]:
        return [sp.sympify(r) for r in json.loads(s)]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── row → record ──────────────────────────────────────────────────────

    def _to_record(self, row: sqlite3.Row) -> IntegralRecord:
        gens = self._deser_exprs(row["toric_gens"]) if row["toric_gens"] is not None else None
        is_bin = bool(row["is_binomial"]) if row["is_binomial"] is not None else None
        keys = row.keys()

        def _opt(col: str):
            return row[col] if col in keys else None

        raw_orbits = _opt("vertex_orbits")
        orbits = json.loads(raw_orbits) if raw_orbits is not None else None

        return IntegralRecord(
            id=row["id"],
            fingerprint=row["fingerprint"],
            a_matrix=self._deser_matrix(row["a_matrix"]),
            newton_points=self._deser_points(row["newton_points"]),
            n_rows=row["n_rows"],
            n_cols=row["n_cols"],
            loop_count=row["loop_count"],
            n_props=row["n_props"],
            n_ext=row["n_ext"],
            n_toric_gens=row["n_toric_gens"],
            toric_generators=gens,
            is_binomial=is_bin,
            cnickel=_opt("cnickel"),
            label=row["label"],
            poly_aut_order=_opt("poly_aut_order"),
            graph_aut_order=_opt("graph_aut_order"),
            coeff_pres_order=_opt("coeff_pres_order"),
            vertex_orbits=orbits,
            stored_at=row["stored_at"],
        )

    # ── internal fast-path used by FeynmanIntegral ────────────────────────

    def _lookup_toric(self, points: list[tuple[int, ...]]) -> list[sp.Expr] | None:
        """Return cached toric generators for these Newton points, or None."""
        fp = self._fingerprint(points)
        row = self._conn.execute(
            "SELECT toric_gens FROM integrals WHERE fingerprint=?", (fp,)
        ).fetchone()
        if row is None or row["n_toric_gens"] is None:
            return None
        return self._deser_exprs(row["toric_gens"])

    @staticmethod
    def _compute_automorphism_data(fi: "FeynmanIntegral") -> tuple[int, int, int, str]:
        """Return (poly_aut_order, graph_aut_order, coeff_pres_order, vertex_orbits_json)."""
        from .normal_forms.polytope_automorphisms import (
            coefficient_preserving_indices,
            compute_graph_automorphisms,
            compute_polytope_automorphisms,
        )

        auts = compute_polytope_automorphisms(fi.newton_polytope.points)
        gauts = compute_graph_automorphisms(fi.graph)
        cp = coefficient_preserving_indices(fi, auts)
        return (
            auts.order,
            len(gauts),
            len(cp),
            json.dumps(auts.vertex_orbits),
        )

    def _store_toric(
        self,
        fi: "FeynmanIntegral",
        generators: list[sp.Expr],
    ) -> None:
        """
        Upsert an integral record with the given toric generators.
        Called automatically by :class:`FeynmanIntegral` after computing.
        """
        from .algebra import is_binomial_ideal

        pts = fi.newton_polytope.points
        fp = self._fingerprint(pts)
        A = fi.gkz.a_matrix
        is_bin = is_binomial_ideal(generators) if generators else None
        try:
            cn: str | None = fi.graph.cnickel()
        except Exception:
            cn = None
        now = self._now()

        self._conn.execute(
            """INSERT INTO integrals
               (fingerprint, a_matrix, newton_points, n_rows, n_cols,
                loop_count, n_props, n_ext, n_toric_gens, toric_gens,
                is_binomial, cnickel, label, stored_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(fingerprint) DO UPDATE SET
                   toric_gens  = excluded.toric_gens,
                   n_toric_gens= excluded.n_toric_gens,
                   is_binomial = excluded.is_binomial,
                   cnickel     = COALESCE(integrals.cnickel, excluded.cnickel)""",
            (
                fp,
                self._ser_matrix(A),
                self._ser_points(pts),
                A.rows,
                A.cols,
                fi.loop_count,
                len(fi.graph.get_internal_edges()),
                fi.graph.external_legs,
                len(generators),
                self._ser_exprs(generators),
                int(is_bin) if is_bin is not None else None,
                cn,
                None,
                now,
            ),
        )
        self._conn.commit()

    # ── public API ────────────────────────────────────────────────────────

    def store(
        self,
        fi: "FeynmanIntegral",
        *,
        label: str | None = None,
        compute_automorphisms: bool = False,
    ) -> IntegralRecord:
        """
        Store a :class:`FeynmanIntegral` and all of its computed properties.

        If the fingerprint is already present, the record is updated
        (label, cnickel, toric generators, and automorphism data are refreshed).

        Parameters
        ----------
        fi
            The integral to store.
        label
            Optional human-readable label (e.g. ``"massless triangle"``).
        compute_automorphisms
            If ``True``, compute and store the full unimodular automorphism
            group of the Newton polytope, the graph automorphism order, and
            the coefficient-preserving subgroup order.  This can be slow for
            large integrals; it defaults to ``False``.
        """
        pts = fi.newton_polytope.points
        fp = self._fingerprint(pts)
        A = fi.gkz.a_matrix
        ti = fi.toric_ideal
        gens = ti.generators

        from .algebra import is_binomial_ideal

        is_bin = is_binomial_ideal(gens) if gens else None
        try:
            cn: str | None = fi.graph.cnickel()
        except Exception:
            cn = None

        poly_ord: int | None = None
        graph_ord: int | None = None
        cp_ord: int | None = None
        orbits_json: str | None = None
        if compute_automorphisms:
            poly_ord, graph_ord, cp_ord, orbits_json = self._compute_automorphism_data(fi)

        now = self._now()

        self._conn.execute(
            """INSERT INTO integrals
               (fingerprint, a_matrix, newton_points, n_rows, n_cols,
                loop_count, n_props, n_ext, n_toric_gens, toric_gens,
                is_binomial, cnickel, label,
                poly_aut_order, graph_aut_order, coeff_pres_order, vertex_orbits,
                stored_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(fingerprint) DO UPDATE SET
                   toric_gens       = excluded.toric_gens,
                   n_toric_gens     = excluded.n_toric_gens,
                   is_binomial      = excluded.is_binomial,
                   cnickel          = COALESCE(integrals.cnickel, excluded.cnickel),
                   label            = COALESCE(integrals.label, excluded.label),
                   poly_aut_order   = COALESCE(excluded.poly_aut_order,   integrals.poly_aut_order),
                   graph_aut_order  = COALESCE(excluded.graph_aut_order,  integrals.graph_aut_order),
                   coeff_pres_order = COALESCE(excluded.coeff_pres_order, integrals.coeff_pres_order),
                   vertex_orbits    = COALESCE(excluded.vertex_orbits,    integrals.vertex_orbits)""",
            (
                fp,
                self._ser_matrix(A),
                self._ser_points(pts),
                A.rows,
                A.cols,
                fi.loop_count,
                len(fi.graph.get_internal_edges()),
                fi.graph.external_legs,
                len(gens),
                self._ser_exprs(gens) if gens else None,
                int(is_bin) if is_bin is not None else None,
                cn,
                label,
                poly_ord,
                graph_ord,
                cp_ord,
                orbits_json,
                now,
            ),
        )
        self._conn.commit()

        row = self._conn.execute("SELECT * FROM integrals WHERE fingerprint=?", (fp,)).fetchone()
        return self._to_record(row)

    def lookup(self, fi: "FeynmanIntegral") -> IntegralRecord | None:
        """Return the stored record for fi, or None if not present."""
        fp = self._fingerprint(fi.newton_polytope.points)
        row = self._conn.execute("SELECT * FROM integrals WHERE fingerprint=?", (fp,)).fetchone()
        return self._to_record(row) if row else None

    def find_equivalent(
        self,
        fi: "FeynmanIntegral",
        *,
        relation: str = "unimodular",
    ) -> list[IntegralRecord]:
        """
        Find all stored integrals whose Newton polytope is equivalent to fi's.

        Candidates are pre-filtered by ``(n_rows, n_cols, n_toric_gens)``
        before running the full equivalence test.  Results are cached in the
        ``equivalences`` table so repeat queries are instant.

        Parameters
        ----------
        fi
            The integral to compare against the database.
        relation
            ``"unimodular"`` (Liu–Cai) or ``"affine_polytope"`` (rational, hull vertices).
        """
        from .normal_forms.affine_equivalence import (
            is_affinely_equivalent,
            is_unimodular_equivalent,
        )

        pts_q = fi.newton_polytope.points
        fp_q = self._fingerprint(pts_q)
        n_gens = len(fi.toric_ideal.generators)
        A = fi.gkz.a_matrix

        candidates = self._conn.execute(
            """SELECT * FROM integrals
               WHERE n_rows=? AND n_cols=?
               AND fingerprint != ?""",
            (A.rows, A.cols, fp_q),
        ).fetchall()

        results = []
        for row in candidates:
            fp_c = row["fingerprint"]

            cached = self._conn.execute(
                """SELECT equivalent, witness_map FROM equivalences
                   WHERE fingerprint_a=? AND fingerprint_b=? AND relation=?""",
                (fp_q, fp_c, relation),
            ).fetchone()

            if cached is not None:
                if cached["equivalent"]:
                    results.append(self._to_record(row))
                continue

            pts_c = self._deser_points(row["newton_points"])
            if relation == "unimodular":
                result = is_unimodular_equivalent(pts_q, pts_c)
            else:
                result = is_affinely_equivalent(pts_q, pts_c)

            witness = (
                self._ser_matrix(result.witness_map) if result.witness_map is not None else None
            )
            now = self._now()
            for fa, fb in ((fp_q, fp_c), (fp_c, fp_q)):
                self._conn.execute(
                    """INSERT OR IGNORE INTO equivalences
                       (fingerprint_a, fingerprint_b, relation,
                        equivalent, witness_map, checked_at)
                       VALUES (?,?,?,?,?,?)""",
                    (fa, fb, relation, int(result.equivalent), witness, now),
                )
            self._conn.commit()

            if result.equivalent:
                results.append(self._to_record(row))

        return results

    def find_equivalent_record(
        self,
        record: IntegralRecord,
        *,
        relation: str = "unimodular",
    ) -> list[IntegralRecord]:
        """
        Like :meth:`find_equivalent` but works from a stored record, without
        needing a live :class:`FeynmanIntegral`.  Used for post-run analysis.
        """
        from .normal_forms.affine_equivalence import (
            is_affinely_equivalent,
            is_unimodular_equivalent,
        )

        fp_q = record.fingerprint
        pts_q = record.newton_points

        candidates = self._conn.execute(
            """SELECT * FROM integrals
               WHERE n_rows=? AND n_cols=?
               AND fingerprint != ?""",
            (record.n_rows, record.n_cols, fp_q),
        ).fetchall()

        results = []
        for row in candidates:
            fp_c = row["fingerprint"]

            cached = self._conn.execute(
                """SELECT equivalent, witness_map FROM equivalences
                   WHERE fingerprint_a=? AND fingerprint_b=? AND relation=?""",
                (fp_q, fp_c, relation),
            ).fetchone()

            if cached is not None:
                if cached["equivalent"]:
                    results.append(self._to_record(row))
                continue

            pts_c = self._deser_points(row["newton_points"])
            if relation == "unimodular":
                result = is_unimodular_equivalent(pts_q, pts_c)
            else:
                result = is_affinely_equivalent(pts_q, pts_c)

            witness = (
                self._ser_matrix(result.witness_map) if result.witness_map is not None else None
            )
            now = self._now()
            for fa, fb in ((fp_q, fp_c), (fp_c, fp_q)):
                self._conn.execute(
                    """INSERT OR IGNORE INTO equivalences
                       (fingerprint_a, fingerprint_b, relation,
                        equivalent, witness_map, checked_at)
                       VALUES (?,?,?,?,?,?)""",
                    (fa, fb, relation, int(result.equivalent), witness, now),
                )
            self._conn.commit()

            if result.equivalent:
                results.append(self._to_record(row))

        return results

    def clear_equivalence_cache(self, *, relation: str = "unimodular") -> int:
        """
        Delete cached negative equivalence results so they are recomputed on
        the next equivalence scan.

        Only ``equivalent=0`` rows are removed; confirmed positive equivalences
        are preserved.  Call this after upgrading the equivalence algorithm to
        ensure stale False results are recomputed with the corrected code.

        Returns the number of rows deleted.
        """
        cur = self._conn.execute(
            "DELETE FROM equivalences WHERE equivalent=0 AND relation=?",
            (relation,),
        )
        self._conn.commit()
        return cur.rowcount

    def all_integrals(self) -> list[IntegralRecord]:
        """Return every stored record, oldest first."""
        return [
            self._to_record(r)
            for r in self._conn.execute("SELECT * FROM integrals ORDER BY id").fetchall()
        ]

    def get_by_id(self, record_id: int) -> IntegralRecord | None:
        row = self._conn.execute("SELECT * FROM integrals WHERE id=?", (record_id,)).fetchone()
        return self._to_record(row) if row else None

    def summary(self) -> str:
        """Human-readable overview of the database contents."""
        n_int = self._conn.execute("SELECT COUNT(*) FROM integrals").fetchone()[0]
        n_equiv = self._conn.execute("SELECT COUNT(*) FROM equivalences").fetchone()[0]
        rows = self._conn.execute("""SELECT n_rows, n_cols, n_toric_gens, loop_count,
                      n_props, is_binomial, cnickel,
                      poly_aut_order, graph_aut_order, coeff_pres_order,
                      label, stored_at
               FROM integrals ORDER BY id""").fetchall()

        # Decide whether to show automorphism columns (any row has data).
        show_aut = any(r["poly_aut_order"] is not None for r in rows)

        if show_aut:
            header = (
                f"  {'A shape':<10} {'gens':>6}  {'L':>3}  {'props':>5}"
                f"  {'bin':>4}  {'|Aut(P)|':>9}  {'|Aut(G)|':>9}"
                f"  {'|CP|':>5}  label"
            )
            sep = "  " + "─" * 72
        else:
            header = f"  {'A shape':<10} {'gens':>6}  {'L':>3}  {'props':>5}" f"  {'bin':>4}  label"
            sep = "  " + "─" * 52

        lines = [
            f"FeynkitDatabase: {self._path}",
            f"  {n_int} integral(s)  ·  {n_equiv} equivalence check(s) cached",
            "",
            header,
            sep,
        ]
        for r in rows:
            lbl = r["label"] or ""
            bi = {None: "?", 0: "no", 1: "yes"}[r["is_binomial"]]
            base = (
                f"  {r['n_rows']}×{r['n_cols']:<7} {r['n_toric_gens'] or '?':>6}"
                f"  {r['loop_count'] or '?':>3}"
                f"  {r['n_props'] or '?':>5}"
                f"  {bi:>4}"
            )
            if show_aut:
                pa = r["poly_aut_order"]
                ga = r["graph_aut_order"]
                cp = r["coeff_pres_order"]
                lines.append(
                    base
                    + f"  {pa if pa is not None else '?':>9}"
                    + f"  {ga if ga is not None else '?':>9}"
                    + f"  {cp if cp is not None else '?':>5}"
                    + f"  {lbl}"
                )
            else:
                lines.append(base + f"  {lbl}")
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FeynkitDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        n = self._conn.execute("SELECT COUNT(*) FROM integrals").fetchone()[0]
        return f"FeynkitDatabase({str(self._path)!r}, {n} record(s))"
