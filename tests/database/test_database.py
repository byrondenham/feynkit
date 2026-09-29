"""Tests for the SQLite result cache."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

from feynkit import FeynkitDatabase, FeynmanIntegral, Graph
from feynkit.database import StoredKinematics


@pytest.fixture  # type: ignore[misc]
def db(tmp_path: Path) -> FeynkitDatabase:
    with FeynkitDatabase(tmp_path / "test.db") as database:
        yield database


# The on-shell massless box as 0.4.0 stored it: ten columns, four of them monomials whose
# coefficients cancel, in the column order of FeynmanIntegral.gkz, and the ten toric
# generators of that matrix.
STALE_BOX_COLUMNS = (
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)
STALE_BOX_GENERATORS = (
    "z_5*z_9 - z_6*z_8",
    "z_10*z_4 - z_5*z_9",
    "z_3*z_9 - z_6*z_7",
    "z_10*z_2 - z_3*z_9",
    "z_1*z_6 - z_2*z_5",
    "z_1*z_9 - z_2*z_8",
    "z_1*z_6 - z_3*z_4",
    "z_1*z_10 - z_3*z_8",
    "z_1*z_9 - z_4*z_7",
    "z_1*z_10 - z_5*z_7",
)

# The upsert of FeynkitDatabase._store_toric in 0.4.0, which still writes to files that later
# releases open.
STORE_TORIC_0_4_0 = """INSERT INTO integrals
               (fingerprint, a_matrix, newton_points, n_rows, n_cols,
                loop_count, n_props, n_ext, n_toric_gens, toric_gens,
                is_binomial, cnickel, label, stored_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(fingerprint) DO UPDATE SET
                   toric_gens  = excluded.toric_gens,
                   n_toric_gens= excluded.n_toric_gens,
                   is_binomial = excluded.is_binomial,
                   cnickel     = COALESCE(integrals.cnickel, excluded.cnickel)"""


def on_shell_box(database: FeynkitDatabase | None = None) -> FeynmanIntegral:
    """The massless box with p_i^2 = 0: six monomials of G."""
    box = FeynmanIntegral.from_cnickel("12e|3e|3e|e|:zzzz", database=database)
    on_shell = {sp.Symbol(f"p{i}^2", real=True): 0 for i in range(1, 5)}
    products = {k: sp.expand(v.subs(on_shell)) for k, v in box.momentum_products.items()}
    return box.with_(momentum_products=products)


def write_as_0_4_0(connection: sqlite3.Connection, fi: FeynmanIntegral) -> None:
    """Cache the on-shell box's ten generators as 0.4.0 did, with its ten-column matrix."""
    points = fi.newton_polytope.points
    matrix = [[1] * len(STALE_BOX_COLUMNS)] + [[c[i] for c in STALE_BOX_COLUMNS] for i in range(4)]
    generators = [sp.srepr(sp.sympify(g)) for g in STALE_BOX_GENERATORS]
    connection.execute(
        STORE_TORIC_0_4_0,
        (
            FeynkitDatabase._fingerprint(points),
            json.dumps(matrix),
            json.dumps([list(p) for p in points]),
            5,
            10,
            1,
            4,
            4,
            10,
            json.dumps(generators),
            1,
            "12e|3e|3e|e|:zzzz",
            None,
            "2026-09-27T00:00:00+00:00",
        ),
    )
    connection.commit()


class TestStoreAndLookup:
    def test_lookup_of_unknown_integral_is_none(self, db: FeynkitDatabase) -> None:
        assert db.lookup(FeynmanIntegral.from_cnickel("11e|e|:zz")) is None

    def test_store_then_lookup_round_trips_cnickel(self, db: FeynkitDatabase) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        stored = db.store(fi, label="bubble")
        found = db.lookup(fi)
        assert found is not None
        assert found.fingerprint == stored.fingerprint
        assert found.cnickel == "11e|e|:zz"
        assert found.label == "bubble"

    def test_store_is_idempotent_per_fingerprint(self, db: FeynkitDatabase) -> None:
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        db.store(fi)
        db.store(fi, label="again")
        assert len(db.all_integrals()) == 1

    def test_stored_record_has_toric_generator_count(self, db: FeynkitDatabase) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        rec = db.store(fi)
        assert rec.n_toric_gens == len(fi.toric_ideal.generators)

    def test_get_by_id(self, db: FeynkitDatabase) -> None:
        rec = db.store(FeynmanIntegral.from_cnickel("11e|e|:zz"))
        assert db.get_by_id(rec.id) is not None
        assert db.get_by_id(rec.id + 1000) is None


class TestToricCache:
    def test_facade_uses_database_cache(self, db: FeynkitDatabase) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz", database=db)
        assert db.lookup(fi) is None
        gens = fi.toric_ideal.generators
        cached = db.lookup(fi)
        assert cached is not None
        assert cached.n_toric_gens == len(gens)

    def test_cached_generators_are_reused(self, db: FeynkitDatabase) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz", database=db)
        first = fi.toric_ideal.generators
        second = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz", database=db).toric_ideal.generators
        assert [str(g) for g in first] == [str(g) for g in second]

    def test_a_row_of_the_newton_points_is_read(self, db: FeynkitDatabase) -> None:
        # Six columns for six points, and a generator in z_1, ..., z_6: the row is read,
        # not recomputed.
        fi = on_shell_box(db)
        assert len(fi.toric_ideal.generators) == 1
        cached = db._lookup_toric(fi.newton_polytope.points)
        assert cached is not None
        assert [str(g) for g in cached] == ["z_1*z_4*z_5 - z_2*z_3*z_6"]


class TestFindEquivalent:
    def test_one_mass_triangles_are_unimodularly_equivalent(self, db: FeynkitDatabase) -> None:
        fi_a = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        fi_b = FeynmanIntegral.from_cnickel("12e|2e|e|:znz")
        db.store(fi_a, label="a")
        db.store(fi_b, label="b")
        matches = db.find_equivalent(fi_a, relation="unimodular")
        assert [m.label for m in matches] == ["b"]

    def test_result_is_cached_in_both_directions(self, db: FeynkitDatabase) -> None:
        fi_a = FeynmanIntegral.from_cnickel("12e|2e|e|:nzz")
        fi_b = FeynmanIntegral.from_cnickel("12e|2e|e|:znz")
        db.store(fi_a, label="a")
        db.store(fi_b, label="b")
        db.find_equivalent(fi_a)
        n_rows = db._conn.execute("SELECT COUNT(*) FROM equivalences").fetchone()[0]
        assert n_rows == 2
        assert [m.label for m in db.find_equivalent(fi_b)] == ["a"]
        # Positive results survive a cache clear, which only drops negatives.
        assert db.clear_equivalence_cache() == 0

    def test_inequivalent_integrals_are_not_matched(self, db: FeynkitDatabase) -> None:
        bubble = FeynmanIntegral.from_cnickel("11e|e|:zz")
        triangle = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        db.store(bubble)
        db.store(triangle)
        assert db.find_equivalent(bubble) == []


class TestSummary:
    def test_summary_mentions_stored_count(self, db: FeynkitDatabase) -> None:
        db.store(FeynmanIntegral.from_cnickel("11e|e|:zz"))
        assert "1" in db.summary()


class TestEmptyToricIdeal:
    def test_facade_reads_back_an_empty_toric_ideal(self, db: FeynkitDatabase) -> None:
        # The massless bubble has no toric ideal generators (an empty list,
        # not a missing value); storing and re-reading it through the facade
        # must not raise.
        fi = FeynmanIntegral.from_cnickel("11e|e|:zz")
        db.store(fi)
        cached = FeynmanIntegral.from_cnickel("11e|e|:zz", database=db)
        assert cached.toric_ideal.generators == []

    def test_lookup_toric_treats_zero_count_as_an_empty_ideal(self, db: FeynkitDatabase) -> None:
        fp = FeynkitDatabase._fingerprint([(1, 0), (0, 1), (1, 1)])
        db._conn.execute(
            """INSERT INTO integrals
               (fingerprint, a_matrix, newton_points, n_rows, n_cols,
                n_toric_gens, toric_gens, stored_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (fp, "[[1]]", "[[1,0],[0,1],[1,1]]", 1, 3, 0, None, db._now()),
        )
        db._conn.commit()
        assert db._lookup_toric([(1, 0), (0, 1), (1, 1)]) == []

    def test_lookup_toric_flags_a_positive_count_with_a_null_column(
        self, db: FeynkitDatabase
    ) -> None:
        fp = FeynkitDatabase._fingerprint([(1, 0), (0, 1), (1, 1)])
        db._conn.execute(
            """INSERT INTO integrals
               (fingerprint, a_matrix, newton_points, n_rows, n_cols,
                n_toric_gens, toric_gens, stored_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (fp, "[[1]]", "[[1,0],[0,1],[1,1]]", 1, 3, 3, None, db._now()),
        )
        db._conn.commit()
        assert db._lookup_toric([(1, 0), (0, 1), (1, 1)]) is None


class TestStaleRows:
    """Rows that releases up to 0.4.0 write: more columns than Newton points, or generators
    in variables beyond the columns of the row's matrix."""

    def test_a_row_with_more_columns_than_points_is_not_read(self, db: FeynkitDatabase) -> None:
        fi = on_shell_box()
        write_as_0_4_0(db._conn, fi)
        assert db._lookup_toric(fi.newton_polytope.points) is None

    def test_the_facade_recomputes_and_rewrites_such_a_row(self, db: FeynkitDatabase) -> None:
        write_as_0_4_0(db._conn, on_shell_box())
        fi = on_shell_box(db)
        assert len(fi.toric_ideal.generators) == 1
        record = db.lookup(fi)
        assert record is not None
        assert (record.n_rows, record.n_cols, record.n_toric_gens) == (5, 6, 1)
        assert record.a_matrix == fi.gkz.a_matrix

    def test_generators_beyond_the_columns_are_not_read(self, db: FeynkitDatabase) -> None:
        fi = on_shell_box(db)
        assert len(fi.toric_ideal.generators) == 1
        # 0.4.0 caches its ten generators on the six-column row it finds.
        write_as_0_4_0(db._conn, fi)
        record = db.lookup(fi)
        assert record is not None
        assert (record.n_cols, record.n_toric_gens) == (6, 10)
        assert db._lookup_toric(fi.newton_polytope.points) is None
        assert len(on_shell_box(db).toric_ideal.generators) == 1
        record = db.lookup(fi)
        assert record is not None
        assert record.n_toric_gens == 1

    def test_store_refreshes_the_matrix(self, db: FeynkitDatabase) -> None:
        write_as_0_4_0(db._conn, on_shell_box())
        record = db.store(on_shell_box(), label="on-shell box")
        assert (record.n_rows, record.n_cols, record.n_toric_gens) == (5, 6, 1)
        assert record.a_matrix == on_shell_box().gkz.a_matrix


def old_file(path: Path) -> None:
    """A file as 0.4.0 left it, at user_version 0 and without the table of kinematic classes:
    the massless triangle's row, the on-shell box's stale row with its automorphism order, and
    a cached equivalence between them."""
    triangle = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
    with FeynkitDatabase(path) as database:
        database.store(triangle, label="triangle")
    connection = sqlite3.connect(path)
    box = on_shell_box()
    write_as_0_4_0(connection, box)
    box_fp = FeynkitDatabase._fingerprint(box.newton_polytope.points)
    connection.execute(
        "UPDATE integrals SET label='on-shell box', poly_aut_order=72 WHERE fingerprint=?",
        (box_fp,),
    )
    connection.execute(
        """INSERT INTO equivalences
           (fingerprint_a, fingerprint_b, relation, equivalent, witness_map, checked_at)
           VALUES (?,?,'unimodular',0,NULL,'2026-09-27T00:00:00+00:00')""",
        (box_fp, FeynkitDatabase._fingerprint(triangle.newton_polytope.points)),
    )
    connection.execute("DROP TABLE kinematic_classes")
    connection.execute("PRAGMA user_version = 0")
    connection.commit()
    connection.close()


def rows(path: Path) -> tuple[int, list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    """The user_version, the integrals by id and the equivalences of a file."""
    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    integrals = connection.execute("SELECT * FROM integrals ORDER BY id").fetchall()
    equivalences = connection.execute("SELECT * FROM equivalences ORDER BY id").fetchall()
    connection.close()
    return version, integrals, equivalences


class TestRepair:
    """_migrate repairs an old file once: the GKZ columns from user_version 0 to 1, the
    cached normal forms from 1 to 2 and the table of kinematic classes from 2 to 3."""

    def test_a_new_file_starts_at_version_3(self, tmp_path: Path) -> None:
        FeynkitDatabase(tmp_path / "new.db").close()
        assert rows(tmp_path / "new.db")[0] == 3

    def test_an_old_file_is_repaired_when_opened(self, tmp_path: Path) -> None:
        path = tmp_path / "old.db"
        old_file(path)
        _, before, equivalences = rows(path)
        box = on_shell_box()
        with FeynkitDatabase(path) as db:
            record = db.lookup(box)
            assert record is not None
            assert record.a_matrix == box.gkz.a_matrix
            assert (record.n_rows, record.n_cols) == (5, 6)
            assert record.n_toric_gens is None
            assert record.toric_generators is None
            assert record.is_binomial is None
            assert (record.label, record.poly_aut_order) == ("on-shell box", None)
            assert record.newton_points == box.newton_polytope.points
            assert len(on_shell_box(db).toric_ideal.generators) == 1
        version, after, _ = rows(path)
        assert version == 3
        assert after[0] == before[0]  # the triangle's row
        assert equivalences
        assert rows(path)[2] == []

    def test_the_repair_runs_once(self, tmp_path: Path) -> None:
        path = tmp_path / "old.db"
        with FeynkitDatabase(path) as db:
            db.store(FeynmanIntegral.from_cnickel("12e|2e|e|:zzz"))
        # 0.4.0 adds a stale row to the repaired file.
        connection = sqlite3.connect(path)
        write_as_0_4_0(connection, on_shell_box())
        connection.close()
        with FeynkitDatabase(path) as db:
            record = db.lookup(on_shell_box())
            assert record is not None
            assert record.n_cols == 10
            assert db._lookup_toric(on_shell_box().newton_polytope.points) is None
        assert rows(path)[0] == 3

    def test_a_failed_repair_leaves_the_file_as_it_was(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "old.db"
        old_file(path)
        before = rows(path)
        repair = FeynkitDatabase._repair_gkz_columns

        def repair_then_fail(self: FeynkitDatabase) -> None:
            repair(self)
            raise RuntimeError("interrupted")

        monkeypatch.setattr(FeynkitDatabase, "_repair_gkz_columns", repair_then_fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            FeynkitDatabase(path)
        assert rows(path) == before

    def test_an_sql_error_in_a_step_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Not a file that cannot be written, but a bug: the version written before the step
        # is rolled back with it.
        path = tmp_path / "old.db"
        old_file(path)
        before = rows(path)
        new = FeynkitDatabase(tmp_path / "new.db")
        new._conn.execute("PRAGMA user_version = 0")

        def bad_sql(self: FeynkitDatabase) -> None:
            self._conn.execute("UPDATE no_such_table SET n = 1")

        monkeypatch.setattr(FeynkitDatabase, "_repair_gkz_columns", bad_sql)
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            FeynkitDatabase(path)
        assert rows(path) == before
        with new:
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                new._migrate()
            assert not new._conn.in_transaction
        assert rows(tmp_path / "new.db")[0] == 0

    def test_a_read_only_file_opens_unrepaired(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "old.db"
        old_file(path)
        before = rows(path)
        connect = sqlite3.connect

        def read_only(database: str, **kwargs: Any) -> sqlite3.Connection:
            return connect(f"file:{database}?mode=ro", uri=True, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", read_only)
        with FeynkitDatabase(path) as db:
            assert not db._conn.in_transaction
            assert db.lookup(on_shell_box()) is not None
            assert db._lookup_toric(on_shell_box().newton_polytope.points) is None
        monkeypatch.undo()
        assert rows(path) == before
        FeynkitDatabase(path).close()
        assert rows(path)[0] == 3

    def test_a_file_from_a_later_release_is_left_as_it_is(self, tmp_path: Path) -> None:
        path = tmp_path / "old.db"
        old_file(path)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA user_version = 99")
        connection.close()
        before = rows(path)
        FeynkitDatabase(path).close()
        assert rows(path) == before

    def test_a_negative_version_runs_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "old.db"
        old_file(path)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA user_version = -1")
        connection.close()
        before = rows(path)
        FeynkitDatabase(path).close()
        assert rows(path) == before


class TestOpen:
    def test_a_file_that_is_not_a_database_is_closed_before_the_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = tmp_path / "notes.db"
        bad.write_bytes(b"not a database\n" * 64)
        opened: list[sqlite3.Connection] = []
        connect = sqlite3.connect

        def recording_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            connection = connect(*args, **kwargs)
            opened.append(connection)
            return connection

        monkeypatch.setattr(sqlite3, "connect", recording_connect)
        with pytest.raises(sqlite3.DatabaseError):
            FeynkitDatabase(bad)
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")


def file_at_version_1(path: Path) -> tuple[str, str]:
    """A file as this release's first repair leaves one written by 0.4.0: one graph with its
    massless self-loop on two different vertices, with the automorphism order 1 and the
    negative verdict that 0.4.0 computed for them. Returns the two fingerprints."""
    znnn = FeynmanIntegral.from_cnickel("012e|2e|e|:znnn")
    nnzn = FeynmanIntegral.from_cnickel("12e|12e|e|:nnzn")
    with FeynkitDatabase(path) as database:
        database.store(znnn, label="znnn")
        database.store(nnzn, label="nnzn")
    fingerprints = tuple(
        FeynkitDatabase._fingerprint(fi.newton_polytope.points) for fi in (znnn, nnzn)
    )
    connection = sqlite3.connect(path)
    connection.execute("""UPDATE integrals SET poly_aut_order = 1, graph_aut_order = 2,
           coeff_pres_order = 1, vertex_orbits = '[[0], [1], [2], [3], [4], [5]]'""")
    for a, b in (fingerprints, fingerprints[::-1]):
        connection.execute(
            """INSERT INTO equivalences
               (fingerprint_a, fingerprint_b, relation, equivalent, witness_map, checked_at)
               VALUES (?,?,'unimodular',0,NULL,'2026-09-27T00:00:00+00:00')""",
            (a, b),
        )
    connection.execute("DROP TABLE kinematic_classes")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()
    return fingerprints[0], fingerprints[1]


class TestNormalFormRepair:
    """The step from user_version 1 to 2 drops what the floating-point normal forms computed."""

    def test_the_cache_and_the_automorphism_columns_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "old.db"
        file_at_version_1(path)
        _, before, equivalences = rows(path)
        assert len(equivalences) == 2
        FeynkitDatabase(path).close()
        version, after, equivalences = rows(path)
        assert (version, equivalences) == (3, [])
        with FeynkitDatabase(path) as db:
            for record in db.all_integrals():
                assert record.poly_aut_order is None
                assert record.graph_aut_order is None
                assert record.coeff_pres_order is None
                assert record.vertex_orbits is None
        # Everything else is kept.
        columns = [c[1] for c in sqlite3.connect(path).execute("PRAGMA table_info(integrals)")]
        kept = [
            i
            for i, c in enumerate(columns)
            if c not in {"poly_aut_order", "graph_aut_order", "coeff_pres_order", "vertex_orbits"}
        ]
        assert [[row[i] for i in kept] for row in after] == [
            [row[i] for i in kept] for row in before
        ]

    def test_the_verdict_is_recomputed(self, tmp_path: Path) -> None:
        path = tmp_path / "old.db"
        znnn_fp, nnzn_fp = file_at_version_1(path)
        with FeynkitDatabase(path) as db:
            found = db.find_equivalent(FeynmanIntegral.from_cnickel("012e|2e|e|:znnn"))
            assert [record.label for record in found] == ["nnzn"]
            record = db.store(
                FeynmanIntegral.from_cnickel("012e|2e|e|:znnn"), compute_automorphisms=True
            )
            assert (record.poly_aut_order, record.graph_aut_order) == (6, 2)
            assert record.vertex_orbits == [[0, 1, 2], [3, 4, 5]]
        cached = rows(path)[2]
        assert {(row[1], row[2], row[4]) for row in cached} == {
            (znnn_fp, nnzn_fp, 1),
            (nnzn_fp, znnn_fp, 1),
        }

    def test_the_steps_run_in_separate_transactions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A file at version 0 gets both steps; when the second fails, the first stays done.
        path = tmp_path / "old.db"
        old_file(path)

        def fail(self: FeynkitDatabase) -> None:
            raise RuntimeError("interrupted")

        monkeypatch.setattr(FeynkitDatabase, "_repair_automorphisms", fail)
        with pytest.raises(RuntimeError, match="interrupted"):
            FeynkitDatabase(path)
        version, _, equivalences = rows(path)
        assert version == 1
        assert len(equivalences) == 1
        monkeypatch.undo()
        FeynkitDatabase(path).close()
        assert rows(path)[0::2] == (3, [])
        with FeynkitDatabase(path) as db:
            assert [record.poly_aut_order for record in db.all_integrals()] == [None, None]

    def test_a_repaired_file_is_not_repaired_again(self, tmp_path: Path) -> None:
        path = tmp_path / "new.db"
        with FeynkitDatabase(path) as db:
            db.store(FeynmanIntegral.from_cnickel("12e|2e|e|:nzz"), compute_automorphisms=True)
            db.find_equivalent(FeynmanIntegral.from_cnickel("12e|2e|e|:znz"))
        before = rows(path)
        with FeynkitDatabase(path) as db:
            assert db.all_integrals()[0].poly_aut_order == 6
        assert rows(path) == before
        assert len(before[2]) == 2


# The upsert of FeynkitDatabase.store in 0.4.0.
STORE_0_4_0 = """INSERT INTO integrals
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
                   vertex_orbits    = COALESCE(excluded.vertex_orbits,    integrals.vertex_orbits)"""

MASSIVE_BOX = "12e|3e|3e|e|:nnnn"
MASSLESS_BOX = "12e|3e|3e|e|:zzzz"


def tables(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    names = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    connection.close()
    return names


def file_at_version_2(path: Path) -> None:
    """A file as the release before the kinematic classes left it: two rows with their
    automorphism data and a cached equivalence, at user_version 2, without the table."""
    with FeynkitDatabase(path) as database:
        database.store(FeynmanIntegral.from_cnickel("12e|2e|e|:nzz"), compute_automorphisms=True)
        database.find_equivalent(FeynmanIntegral.from_cnickel("12e|2e|e|:znz"))
        database.store(FeynmanIntegral.from_cnickel("12e|2e|e|:znz"), label="znz")
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE kinematic_classes")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()


class TestKinematicClassesRepair:
    """The step from user_version 2 to 3 creates the table of kinematic classes."""

    def test_a_new_file_has_the_table(self, tmp_path: Path) -> None:
        FeynkitDatabase(tmp_path / "new.db").close()
        assert rows(tmp_path / "new.db")[0] == 3
        assert "kinematic_classes" in tables(tmp_path / "new.db")

    @pytest.mark.parametrize("build", [old_file, file_at_version_2], ids=["version-0", "version-2"])
    def test_an_older_file_gains_the_table_once(self, tmp_path: Path, build: Any) -> None:
        path = tmp_path / "old.db"
        build(path)
        assert "kinematic_classes" not in tables(path)
        _, before, equivalences = rows(path)
        with FeynkitDatabase(path) as db:
            records = db.all_integrals()
            assert [record.kinematics for record in records] == [()] * len(before)
        version, after, _ = rows(path)
        assert version == 3
        assert "kinematic_classes" in tables(path)
        assert len(after) == len(before)
        if build is file_at_version_2:
            assert after == before
            assert rows(path)[2] == equivalences
        # Reopening writes nothing.
        content = path.read_bytes()
        with FeynkitDatabase(path) as db:
            db.all_integrals()
        assert path.read_bytes() == content

    def test_a_read_only_file_without_the_table_opens_and_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "old.db"
        file_at_version_2(path)
        before = rows(path)
        connect = sqlite3.connect

        def read_only(database: str, **kwargs: Any) -> sqlite3.Connection:
            return connect(f"file:{database}?mode=ro", uri=True, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", read_only)
        with FeynkitDatabase(path) as db:
            record = db.lookup(FeynmanIntegral.from_cnickel("12e|2e|e|:znz"))
            assert record is not None
            assert record.kinematics == ()
            assert len(db.all_integrals()) == 2
            assert db.all_integrals(kinematic_class="generic") == []
            assert "?" in db.summary()
        monkeypatch.undo()
        assert rows(path) == before
        assert "kinematic_classes" not in tables(path)

    def test_the_0_4_0_store_statement_still_succeeds(self, tmp_path: Path) -> None:
        path = tmp_path / "new.db"
        FeynkitDatabase(path).close()
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        points = fi.newton_polytope.points
        connection = sqlite3.connect(path)
        for _ in range(2):
            connection.execute(
                STORE_0_4_0,
                (
                    FeynkitDatabase._fingerprint(points),
                    FeynkitDatabase._ser_matrix(fi.gkz.a_matrix),
                    json.dumps([list(p) for p in points]),
                    4,
                    6,
                    1,
                    3,
                    3,
                    1,
                    json.dumps([]),
                    None,
                    "12e|2e|e|:zzz",
                    "triangle",
                    None,
                    None,
                    None,
                    None,
                    "2026-09-27T00:00:00+00:00",
                ),
            )
        connection.commit()
        connection.close()
        with FeynkitDatabase(path) as db:
            record = db.lookup(fi)
            assert record is not None
            assert (record.label, record.kinematics) == ("triangle", ())


class TestKinematicClasses:
    def test_generic_and_equal_masses_share_a_polytope(self, db: FeynkitDatabase) -> None:
        generic = FeynmanIntegral.from_cnickel(MASSIVE_BOX)
        equal = generic.with_kinematics("equal_masses")
        first = db.store(generic)
        second = db.store(equal)
        assert first.id == second.id
        assert second.cnickel == MASSIVE_BOX
        assert second.kinematics == (
            StoredKinematics(MASSIVE_BOX, "generic", "off_shell", "generic"),
            StoredKinematics("12e|3e|3e|e|:aaaa", "equal", "off_shell", "equal_masses"),
        )
        assert "classes=generic,equal_masses" in repr(second)
        # Storing again adds no row.
        assert db.store(equal).kinematics == second.kinematics

    def test_off_and_on_shell_are_two_polytopes(self, db: FeynkitDatabase) -> None:
        box = FeynmanIntegral.from_cnickel(MASSLESS_BOX)
        off = db.store(box)
        on = db.store(box.with_kinematics("massless_on_shell"))
        assert off.id != on.id
        assert [k.kinematic_class for k in off.kinematics] == ["massless_off_shell"]
        assert on.kinematics == (
            StoredKinematics(MASSLESS_BOX, "zero", "on_shell", "massless_on_shell"),
        )

    def test_the_massive_box_on_shell_keeps_its_axes(self, db: FeynkitDatabase) -> None:
        box = FeynmanIntegral.from_cnickel(MASSIVE_BOX)
        legs = {sp.Symbol(f"p{i}^2", real=True): 0 for i in range(1, 5)}
        products = {k: sp.expand(v.subs(legs)) for k, v in box.momentum_products.items()}
        db.store(box)
        record = db.store(box.with_(momentum_products=products))
        assert record.kinematics[-1] == StoredKinematics(
            MASSIVE_BOX, "generic", "on_shell", "other"
        )
        assert len(record.kinematics) == 2

    def test_the_toric_cache_records_the_class(self, db: FeynkitDatabase) -> None:
        fi = FeynmanIntegral.from_cnickel(MASSLESS_BOX, database=db).with_kinematics(
            "massless_on_shell"
        )
        assert len(fi.toric_ideal.generators) == 1
        record = db.lookup(fi)
        assert record is not None
        assert [k.kinematic_class for k in record.kinematics] == ["massless_on_shell"]

    def test_a_cnickel_string_that_cannot_be_computed_is_empty(
        self, db: FeynkitDatabase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def too_large(self: Graph) -> str:
            raise NotImplementedError("CNickel index requires V <= 10")

        monkeypatch.setattr(Graph, "cnickel", too_large)
        record = db.store(FeynmanIntegral.from_cnickel("12e|2e|e|:zzz"))
        assert record.cnickel is None
        assert record.kinematics == (
            StoredKinematics("", "zero", "off_shell", "massless_off_shell"),
        )

    def test_all_integrals_filters_by_class_and_axes(self, db: FeynkitDatabase) -> None:
        massive = FeynmanIntegral.from_cnickel(MASSIVE_BOX)
        massless = FeynmanIntegral.from_cnickel(MASSLESS_BOX)
        db.store(massive)
        db.store(massive.with_kinematics("equal_masses"))
        db.store(massless)
        db.store(massless.with_kinematics("massless_on_shell"))
        db.store(FeynmanIntegral.from_cnickel("12e|2e|e|:aaz"))

        def cnickels(**filters: str) -> list[str | None]:
            return [record.cnickel for record in db.all_integrals(**filters)]

        assert len(db.all_integrals()) == 4
        assert cnickels(kinematic_class="equal_masses") == [MASSIVE_BOX]
        assert cnickels(kinematic_class="other") == ["12e|2e|e|:aaz"]
        assert cnickels(internal_axis="zero") == [MASSLESS_BOX, MASSLESS_BOX]
        assert cnickels(external_axis="off_shell") == [MASSIVE_BOX, MASSLESS_BOX, "12e|2e|e|:aaz"]
        assert cnickels(internal_axis="zero", external_axis="on_shell") == [MASSLESS_BOX]
        # The filters hold on one row together.
        assert cnickels(internal_axis="equal", kinematic_class="generic") == []
        assert cnickels(kinematic_class="massless") == []

    def test_summary_lists_the_classes(self, db: FeynkitDatabase) -> None:
        massive = FeynmanIntegral.from_cnickel(MASSIVE_BOX)
        db.store(massive, label="box")
        db.store(massive.with_kinematics("equal_masses"))
        db.store(FeynmanIntegral.from_cnickel("12e|2e|e|:zzz"), label="triangle")
        # A row stored by a release without the table of classes.
        db._conn.execute(
            "DELETE FROM kinematic_classes WHERE kinematic_class = 'massless_off_shell'"
        )
        lines = db.summary().splitlines()
        header = next(line for line in lines if "A shape" in line)
        assert header.index("classes") < header.index("label")
        box = next(line for line in lines if line.endswith("box"))
        assert "generic,equal_masses" in box
        triangle = next(line for line in lines if line.endswith("triangle"))
        assert triangle.split()[-2] == "?"

    def test_values_of_a_later_version_are_returned_as_they_are(self, db: FeynkitDatabase) -> None:
        fi = FeynmanIntegral.from_cnickel("12e|2e|e|:zzz")
        db.store(fi)
        db._conn.execute(
            """INSERT INTO kinematic_classes
               (fingerprint, cnickel, internal_axis, external_axis, kinematic_class, stored_at)
               VALUES (?, 'x', 'light', 'soft', 'soft_limit', 'now')""",
            (FeynkitDatabase._fingerprint(fi.newton_polytope.points),),
        )
        record = db.lookup(fi)
        assert record is not None
        assert record.kinematics[-1] == StoredKinematics("x", "light", "soft", "soft_limit")
        assert [r.cnickel for r in db.all_integrals(kinematic_class="soft_limit")] == [
            "12e|2e|e|:zzz"
        ]
