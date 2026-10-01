"""Pytest configuration and fixtures for feynkit tests."""

import os
from collections.abc import Generator

import pytest
import sympy as sp
import sympy.core.random as sympy_random
from hypothesis import settings

from feynkit.core import Edge, Graph

# Hypothesis's ci profile is derandomised, keeps no example database and has no
# deadline, so local runs and CI draw the same examples. HYPOTHESIS_PROFILE=feynkit-dev
# gives a larger random search.
settings.register_profile("feynkit", parent=settings.get_profile("ci"), max_examples=25)
settings.register_profile("feynkit-dev", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "feynkit"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give the slow and examples tiers a longer limit than the default."""
    for item in items:
        if item.get_closest_marker("timeout") is None and (
            item.get_closest_marker("slow") or item.get_closest_marker("examples")
        ):
            item.add_marker(pytest.mark.timeout(1200))


@pytest.fixture  # type: ignore
def sympy_seeded() -> Generator[None, None, None]:
    """SymPy's random generator seeded for the test, and its state restored afterwards.

    SymPy factors a multivariate polynomial by Wang's algorithm, which draws
    evaluation points from this generator, and from some of its states runs
    for many minutes on a polynomial it otherwise factors in a fraction of a
    second. Seeded, a test factors the same way on every run.
    """
    state = sympy_random.rng.getstate()
    sympy_random.rng.seed(1)
    yield
    sympy_random.rng.setstate(state)


@pytest.fixture  # type: ignore
def simple_masses() -> Generator[tuple[sp.Symbol, sp.Symbol], None, None]:
    """Create simple mass symbols for testing."""
    m1 = sp.Symbol("m1", nonnegative=True)
    m2 = sp.Symbol("m2", nonnegative=True)
    yield m1, m2


@pytest.fixture  # type: ignore
def simple_exponents() -> Generator[tuple[sp.Symbol, sp.Symbol], None, None]:
    """Create simple exponent symbols for testing."""
    nu1 = sp.Symbol("nu1", positive=True)
    nu2 = sp.Symbol("nu2", positive=True)
    yield nu1, nu2


@pytest.fixture  # type: ignore
def bubble_edges(
    simple_masses: tuple[sp.Symbol, sp.Symbol],
    simple_exponents: tuple[sp.Symbol, sp.Symbol],
) -> Generator[list[Edge], None, None]:
    """Create edges for a bubble diagram."""
    m1, m2 = simple_masses
    nu1, nu2 = simple_exponents

    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m1, nu=nu1)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True, mass=m2, nu=nu2)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

    yield [e1, e2, ex1, ex2]


@pytest.fixture  # type: ignore
def bubble_graph(bubble_edges: list[Edge]) -> Generator[Graph, None, None]:
    """Create a bubble diagram graph."""
    yield Graph(internal_vertices=2, external_legs=2, edges=bubble_edges)
