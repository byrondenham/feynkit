"""Tests for the SQLite result cache."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import sympy as sp

from feynkit import FeynkitDatabase, FeynmanIntegral


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
    """A file as 0.4.0 left it, at user_version 0: the massless triangle's row, the on-shell
    box's stale row with its automorphism order, and a cached equivalence between them."""
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
    """_migrate repairs the rows of an old file once, as the step from user_version 0 to 1."""

    def test_a_new_file_starts_at_version_1(self, tmp_path: Path) -> None:
        FeynkitDatabase(tmp_path / "new.db").close()
        assert rows(tmp_path / "new.db")[0] == 1

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
            assert (record.label, record.poly_aut_order) == ("on-shell box", 72)
            assert record.newton_points == box.newton_polytope.points
            assert len(on_shell_box(db).toric_ideal.generators) == 1
        version, after, _ = rows(path)
        assert version == 1
        assert after[0] == before[0]  # the triangle's row
        assert rows(path)[2] == equivalences

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
        assert rows(path)[0] == 1

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
            assert db.lookup(on_shell_box()) is not None
            assert db._lookup_toric(on_shell_box().newton_polytope.points) is None
        monkeypatch.undo()
        assert rows(path) == before
        FeynkitDatabase(path).close()
        assert rows(path)[0] == 1

    def test_a_file_from_a_later_release_is_left_as_it_is(self, tmp_path: Path) -> None:
        path = tmp_path / "old.db"
        old_file(path)
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA user_version = 2")
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
