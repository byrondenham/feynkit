"""Tests for the SQLite result cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from feynkit import FeynkitDatabase, FeynmanIntegral


@pytest.fixture  # type: ignore[misc]
def db(tmp_path: Path) -> FeynkitDatabase:
    with FeynkitDatabase(tmp_path / "test.db") as database:
        yield database


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
