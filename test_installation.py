"""Quick test to verify feynkit installation."""

import sympy as sp

from feynkit import Edge, Graph, __version__


def test_basic_import() -> None:
    """Test that basic imports work."""
    print("yes Successfully imported feynkit core modules")
    print(f"yes Feynkit version: {__version__}")


def test_edge_creation() -> None:
    """Test creating edges."""
    m = sp.Symbol("m", nonnegative=True)
    nu = sp.Symbol("nu", positive=True)

    # Create internal edge
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True, mass=m, nu=nu)
    print(f"yes Created internal edge: {e1}")

    # Create external edge
    e2 = Edge(idx=2, v1=1, v2=3, is_internal=False)
    print(f"yes Created external edge: {e2}")


def test_graph_creation() -> None:
    """Test creating a simple graph."""
    # Create bubble diagram
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

    graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
    print(f"yes Created graph: {graph}")

    # Test basic graph properties
    assert graph.internal_vertices == 2
    assert graph.external_legs == 2
    assert len(graph.get_internal_edges()) == 2
    assert len(graph.get_external_edges()) == 2
    assert graph.get_loop_count() == 1
    print("yes Graph properties verified")


def test_laplacian_calculation() -> None:
    """Test Laplacian matrix calculation."""
    e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
    e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
    ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
    ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

    graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

    # Calculate Laplacian
    L = graph.calculate_laplacian(include_external=False)
    print(f"yes Calculated Laplacian matrix:\n{L}")

    # Calculate W polynomial
    W = graph.calculate_w_polynomial()
    print(f"yes Calculated W polynomial: {W}")


def main() -> None:
    """Run all installation tests."""
    print("=" * 60)
    print("Testing Feynkit Installation")
    print("=" * 60)
    print()

    try:
        test_basic_import()
        print()

        test_edge_creation()
        print()

        test_graph_creation()
        print()

        test_laplacian_calculation()
        print()

        print("=" * 60)
        print("yes All tests passed! Feynkit is correctly installed.")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"no Test failed with error: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
