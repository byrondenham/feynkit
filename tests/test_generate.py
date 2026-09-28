"""Tests for feynkit.generate: the blocks, the options and the errors."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import feynkit
from feynkit import FeynmanIntegral, ValidationError
from feynkit.core import Graph
from feynkit.generate import MAX_VERTICES, generate_graphs, mass_colourings


def topology(cnickel: str) -> str:
    return cnickel.split(":")[0]


def test_generate_graphs_is_exported() -> None:
    assert feynkit.generate_graphs is generate_graphs
    assert "generate_graphs" in feynkit.__all__


def test_tier_one() -> None:
    graphs = list(generate_graphs(1, range(2, 7)))
    assert [topology(s) for s in dict.fromkeys(graphs)][:1] == ["11e|e|"]
    by_legs = Counter(topology(s).count("e") for s in graphs)
    assert [by_legs[n] for n in range(2, 7)] == [3, 4, 6, 8, 13]
    assert sorted({topology(s) for s in graphs}) == [
        "11e|e|",
        "12e|2e|e|",
        "12e|3e|3e|e|",
        "12e|3e|4e|4e|e|",
        "12e|3e|4e|5e|5e|e|",
    ]


# Tier 2 by block (E, n): topologies, "zn" graphs.
TIER_TWO = {
    (3, 2): (1, 4),
    (4, 2): (2, 18),
    (4, 3): (2, 15),
    (5, 2): (2, 32),
    (5, 3): (3, 62),
    (5, 4): (3, 50),
    (6, 3): (3, 83),
    (6, 4): (5, 183),
    (7, 4): (4, 228),
}


def test_tier_two_by_block() -> None:
    graphs = list(generate_graphs(2, range(2, 5)))
    assert len(graphs) == 675
    blocks: dict[tuple[int, int], list[str]] = {}
    for s in graphs:
        g = Graph.from_cnickel(s)
        blocks.setdefault((len(g.get_internal_edges()), g.external_legs), []).append(s)
    found = {block: (len({topology(s) for s in b}), len(b)) for block, b in blocks.items()}
    assert found == TIER_TWO


def test_blocks_come_in_order_and_sorted() -> None:
    # With 2 and 3 legs the ranges of E overlap, so an order by (L, n, E) would fail.
    graphs = list(generate_graphs([2, 1], [3, 2, 3]))
    keys = []
    for s in graphs:
        g = Graph.from_cnickel(s)
        keys.append((g.get_loop_count(), len(g.get_internal_edges()), g.external_legs))
    assert keys == sorted(keys)
    for key in set(keys):
        block = [s for s, k in zip(graphs, keys, strict=True) if k == key]
        assert block == sorted(block)
    assert len(graphs) == len(set(graphs)) == 3 + 4 + 4 + 18 + 15 + 32 + 62 + 83


def test_iterables_of_any_kind() -> None:
    assert list(generate_graphs(iter([2]), (n for n in [2, 3]))) == list(generate_graphs(2, [3, 2]))


def test_numpy_integers_are_integers() -> None:
    assert list(generate_graphs(np.arange(1, 3), np.int64(2), edges=np.arange(2, 5))) == list(
        generate_graphs([1, 2], 2, edges=[2, 3, 4])
    )
    assert list(generate_graphs(2, 3, max_legs_per_vertex=np.int64(2))) == list(
        generate_graphs(2, 3, max_legs_per_vertex=2)
    )


def test_every_string_is_its_own_canonical_form() -> None:
    for masses in ("z", "zn", "shared"):
        for s in generate_graphs(2, range(4), masses=masses, self_loops=True, edges=range(1, 6)):
            assert Graph.from_cnickel(s).cnickel() == s


def test_massless_only() -> None:
    graphs = list(generate_graphs(2, range(2, 5), masses="z"))
    assert len(graphs) == 25
    assert all(set(s.split(":")[1]) == {"z"} for s in graphs)


def test_shared_masses() -> None:
    assert len(list(generate_graphs(1, range(2, 7), masses="shared"))) == 185
    sunrise = list(generate_graphs(2, 2, edges=3, masses="shared"))
    assert sunrise == [
        "111e|e|:aaa",
        "111e|e|:aan",
        "111e|e|:aaz",
        "111e|e|:nnn",
        "111e|e|:nnz",
        "111e|e|:nzz",
        "111e|e|:zzz",
    ]


def test_vacuum_and_one_leg_graphs_on_request() -> None:
    assert [topology(s) for s in generate_graphs(2, [0, 1], masses="z")] == [
        "111||",
        "111e||",
        "112|2|e|",
    ]
    assert list(generate_graphs(1, [0, 1])) == []


def test_self_loops() -> None:
    assert list(generate_graphs(1, 1, self_loops=True)) == ["0e|:n", "0e|:z"]
    assert list(generate_graphs(1, 0, self_loops=True)) == []  # 0| has a vertex of degree 2
    plain = set(generate_graphs(2, range(2, 5)))
    looped = set(generate_graphs(2, range(2, 5), self_loops=True))
    assert plain < looped
    assert len(looped) == 785
    assert len({topology(s) for s in looped}) == 31
    assert "011e|e|:znn" in looped


def test_legs_per_vertex() -> None:
    graphs = list(generate_graphs(2, range(2, 5), max_legs_per_vertex=None))
    assert len(graphs) == 1338
    assert len({topology(s) for s in graphs}) == 64
    assert "11ee|e|:nn" in generate_graphs(1, 3, max_legs_per_vertex=2)


def test_one_vertex_irreducible() -> None:
    kept = {topology(s) for s in generate_graphs(2, 2, one_vertex_irreducible=True)}
    assert kept == {"111e|e|", "112e|2|e|", "112|3|3e|e|", "123|23|e|e|"}
    assert "1122|e|e|" in {topology(s) for s in generate_graphs(2, 2)}


def test_one_vertex_irreducible_at_four_loops() -> None:
    # Two triple propagators sharing a vertex: a core with a cut vertex and no
    # self-loop, which first appears at four loops.
    assert "111222|||:zzzzzz" in generate_graphs(4, 0, edges=6, masses="z")
    kept = list(generate_graphs(4, range(3), masses="z", one_vertex_irreducible=True))
    assert len(kept) == 514
    assert "111222|||:zzzzzz" not in kept


def test_edges() -> None:
    assert {topology(s) for s in generate_graphs(2, 4, edges=7)} == {
        "123|45|4e|5e|e|e|",
        "123|4e|4e|5e|5|e|",
        "123|24|e|5e|5e|e|",
        "112|3|4e|5e|5e|e|",
    }
    assert list(generate_graphs(1, 3, edges=5)) == []
    assert list(generate_graphs(1, 2, edges=15)) == []  # beyond the rules, so no error


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loops": 0, "legs": 2}, "loops must be integers of at least 1"),
        ({"loops": 1, "legs": -1}, "legs must be integers of at least 0"),
        ({"loops": 1, "legs": 2, "edges": 0}, "edges must be integers of at least 1"),
        ({"loops": 1, "legs": [2, "3"]}, "legs must be integers"),
        ({"loops": True, "legs": 2}, "loops must be integers"),
        ({"loops": 1, "legs": 2, "masses": "nz"}, "masses must be one of"),
        ({"loops": 2.0, "legs": 2}, "loops must be integers"),
        ({"loops": 1, "legs": None}, "legs must be integers"),
        ({"loops": 1, "legs": 2, "edges": 2.5}, "edges must be integers"),
        ({"loops": np.bool_(True), "legs": 2}, "loops must be integers"),
        ({"loops": 1, "legs": 2, "max_legs_per_vertex": 0}, "max_legs_per_vertex"),
        ({"loops": 1, "legs": 2, "max_legs_per_vertex": True}, "max_legs_per_vertex"),
        ({"loops": 1, "legs": 2, "max_legs_per_vertex": 1.0}, "max_legs_per_vertex"),
        ({"loops": 1, "legs": 2, "max_legs_per_vertex": [1, 2]}, "max_legs_per_vertex"),
        ({"loops": 1, "legs": 2, "max_legs_per_vertex": "1"}, "max_legs_per_vertex"),
        ({"loops": 1, "legs": 2, "max_legs_per_vertex": np.bool_(True)}, "max_legs_per_vertex"),
        ({"loops": 1, "legs": 11}, "block of 1 loop, 11 propagators and 11 legs has 11 vertices"),
        ({"loops": 6, "legs": 1}, "block of 6 loops, 16 propagators and 1 leg has 11 vertices"),
        ({"loops": 2, "legs": 10, "edges": 12}, "has 11 vertices"),
    ],
)
def test_arguments_are_checked_when_called(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        generate_graphs(**kwargs)


def test_ten_vertices_are_allowed() -> None:
    assert MAX_VERTICES == 10
    decagon = list(generate_graphs(1, 10))
    assert len(decagon) == 78  # binary bracelets of 10 beads
    assert decagon[0] == "12e|3e|4e|5e|6e|7e|8e|9e|9e|e|:nnnnnnnnnn"


def test_mass_colourings() -> None:
    assert mass_colourings("111e|e|") == [
        "111e|e|:nnn",
        "111e|e|:nnz",
        "111e|e|:nzz",
        "111e|e|:zzz",
    ]
    assert len(mass_colourings("12e|23|3|e|")) == 14  # the kite, not canonical
    assert len(mass_colourings("112|3|3e|e|")) == 18  # chains 1, 1, 3
    assert mass_colourings("11e|e|", masses="shared") == [
        "11e|e|:aa",
        "11e|e|:nn",
        "11e|e|:nz",
        "11e|e|:zz",
    ]
    assert mass_colourings("11e|e|", masses="z") == ["11e|e|:zz"]
    with pytest.raises(ValidationError, match="without colours"):
        mass_colourings("11e|e|:zz")
    with pytest.raises(ValidationError, match="masses must be one of"):
        mass_colourings("11e|e|", masses="shared-masses")


@pytest.mark.parametrize("cnickel", ["111||:nnn", "111e||:nzz", "0e|:n"])
def test_vacuum_and_one_leg_graphs_build_without_mandelstam(cnickel: str) -> None:
    fi = FeynmanIntegral.from_cnickel(cnickel, use_mandelstam=False)
    assert fi.cnickel == cnickel
    with pytest.raises(ValueError, match="At least two external legs"):
        FeynmanIntegral.from_cnickel(cnickel)


def test_guide_example_prints_what_the_guide_says(capsys: pytest.CaptureFixture[str]) -> None:
    guide = Path(__file__).resolve().parents[1] / "docs" / "guide.md"
    section = guide.read_text(encoding="utf-8").split("\n### Generating graphs\n", 1)[1]
    code = section.split("```python\n", 1)[1].split("```", 1)[0]
    printed = section.split("prints\n\n```\n", 1)[1].split("```", 1)[0]
    exec(compile(code, "docs/guide.md", "exec"), {})
    assert capsys.readouterr().out == printed
