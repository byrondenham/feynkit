"""Pytest configuration and fixtures for feynkit tests."""

from collections.abc import Generator

import pytest
import sympy as sp

from feynkit.core import Edge, Graph


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
