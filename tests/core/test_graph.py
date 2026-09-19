"""Tests for the Graph class."""

import pytest
import sympy as sp

from feynkit.core import Edge, Graph
from feynkit.core.exceptions import EdgeConnectivityError, GraphTopologyError


class TestGraphCreation:
    """Test Graph instantiation and validation."""

    def test_create_bubble_diagram(self) -> None:
        """Test creating a simple bubble diagram."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
        ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
        ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

        assert graph.internal_vertices == 2
        assert graph.external_legs == 2
        assert len(graph.edges) == 4
        assert len(graph.get_internal_edges()) == 2
        assert len(graph.get_external_edges()) == 2

    def test_create_graph_with_custom_energy_scale(self) -> None:
        """Test creating a graph with custom energy scale."""
        mu = sp.Symbol("mu_custom", positive=True)
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        ex1 = Edge(idx=2, v1=1, v2=3, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=1, edges=[e1, ex1], energy_scale=mu)

        assert graph.energy_scale == mu

    def test_graph_with_duplicate_edge_indices(self) -> None:
        """Test that duplicate edge indices raise error."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=1, v1=2, v2=1, is_internal=True)  # Duplicate idx

        with pytest.raises(GraphTopologyError, match="Duplicate edge indices"):
            Graph(internal_vertices=2, external_legs=0, edges=[e1, e2])

    def test_graph_with_wrong_external_edge_count(self) -> None:
        """Test that mismatch in external edge count raises error."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        ex1 = Edge(idx=2, v1=1, v2=3, is_internal=False)

        # Claim 2 external legs but only provide 1
        with pytest.raises(GraphTopologyError, match="Expected 2 external edges"):
            Graph(internal_vertices=2, external_legs=2, edges=[e1, ex1])

    def test_graph_with_invalid_internal_edge_connectivity(self) -> None:
        """Test that internal edge connecting to external vertex raises error."""
        # Internal edge trying to connect to external vertex 3
        e1 = Edge(idx=1, v1=1, v2=3, is_internal=True)

        with pytest.raises(EdgeConnectivityError, match="internal edge must connect"):
            Graph(internal_vertices=2, external_legs=1, edges=[e1])

    def test_graph_with_invalid_external_edge_connectivity(self) -> None:
        """Test that external edge connecting to internal vertex raises error."""
        # External edge trying to connect to internal vertex 2
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=False)

        with pytest.raises(EdgeConnectivityError, match="external edge must connect"):
            Graph(internal_vertices=2, external_legs=1, edges=[e1])


class TestGraphMethods:
    """Test Graph methods."""

    def test_get_loop_count_bubble(self) -> None:
        """Test loop count calculation for bubble diagram (1 loop)."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
        ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
        ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])

        assert graph.get_loop_count() == 1

    def test_get_loop_count_triangle(self) -> None:
        """Test loop count calculation for triangle diagram (1 loop)."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=3, is_internal=True)
        e3 = Edge(idx=3, v1=3, v2=1, is_internal=True)
        ex1 = Edge(idx=4, v1=1, v2=4, is_internal=False)
        ex2 = Edge(idx=5, v1=2, v2=5, is_internal=False)
        ex3 = Edge(idx=6, v1=3, v2=6, is_internal=False)

        graph = Graph(internal_vertices=3, external_legs=3, edges=[e1, e2, e3, ex1, ex2, ex3])

        assert graph.get_loop_count() == 1

    def test_calculate_laplacian_bubble_internal_only(self) -> None:
        """Test Laplacian calculation for bubble (internal edges only)."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
        ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
        ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
        L = graph.calculate_laplacian(include_external=False)

        # Should be 2x2 matrix
        assert L.shape == (2, 2)

        # Check structure: a_1 and a_2 parameters
        a1 = graph.schwinger_parameters[1]
        a2 = graph.schwinger_parameters[2]

        # L[0,0] should be a_1 + a_2
        assert sp.simplify(L[0, 0] - (a1 + a2)) == 0
        # L[1,1] should be a_1 + a_2
        assert sp.simplify(L[1, 1] - (a1 + a2)) == 0
        # L[0,1] should be -(a_1 + a_2)
        assert sp.simplify(L[0, 1] + (a1 + a2)) == 0

    def test_calculate_w_polynomial_bubble(self) -> None:
        """Test W polynomial calculation for bubble diagram."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
        ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
        ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
        W = graph.calculate_w_polynomial()

        # W should be a non-trivial polynomial in a_1, a_2, b_1, b_2
        assert W != 0
        assert isinstance(W, sp.Expr)

    def test_expand_w_by_external_parameters_bubble(self) -> None:
        """Test W expansion by external parameters for bubble diagram."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
        ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
        ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
        coeffs = graph.expand_w_by_external_parameters()

        # Should have coefficients for different b monomials
        assert isinstance(coeffs, dict)
        assert len(coeffs) > 0

        # All coefficients should be polynomials in a_1, a_2
        a1 = graph.schwinger_parameters[1]
        a2 = graph.schwinger_parameters[2]
        for coeff in coeffs.values():
            # Check that coefficient contains a symbols
            assert a1 in coeff.free_symbols or a2 in coeff.free_symbols or coeff.is_number


class TestCNickelTadpole:
    """Regression tests for tadpole / self-loop CNickel parsing."""

    def test_tadpole_bare_has_one_internal_edge(self) -> None:
        g = Graph.from_cnickel("0|")
        assert len(g.get_internal_edges()) == 1

    def test_tadpole_bare_has_no_external_legs(self) -> None:
        g = Graph.from_cnickel("0|")
        assert g.external_legs == 0

    def test_tadpole_bare_is_self_loop(self) -> None:
        g = Graph.from_cnickel("0|")
        e = g.get_internal_edges()[0]
        assert e.v1 == e.v2

    def test_tadpole_with_mass_color_has_one_edge(self) -> None:
        g = Graph.from_cnickel("0|:z|")
        assert len(g.get_internal_edges()) == 1

    def test_tadpole_with_mass_color_is_massless(self) -> None:
        import sympy as sp

        g = Graph.from_cnickel("0|:z|")
        assert g.get_internal_edges()[0].mass == sp.Integer(0)

    def test_tadpole_cnickel_roundtrip(self) -> None:
        g = Graph.from_cnickel("0|:z|")
        assert g.cnickel() == "0|:z"

    def test_zero_digit_mass_code_is_massless(self) -> None:
        import sympy as sp

        g = Graph.from_cnickel("e11|e|:n00|n|")
        for e in g.get_internal_edges():
            assert e.mass == sp.Integer(0)

    def test_special_mass_code_gives_shared_symbol(self) -> None:
        g = Graph.from_cnickel("12e|2e|e|:sss")
        masses = [str(e.mass) for e in g.get_internal_edges()]
        assert masses == ["m_s", "m_s", "m_s"]

    def test_letter_label_codes_give_shared_symbols(self) -> None:
        g = Graph.from_cnickel("12e|2e|e|:aab")
        masses = [str(e.mass) for e in g.get_internal_edges()]
        assert masses.count("m_a") == 2
        assert masses.count("m_b") == 1

    def test_letter_label_cnickel_roundtrip(self) -> None:
        assert Graph.from_cnickel("12e|2e|e|:aab").cnickel() == "12e|2e|e|:aab"

    def test_special_mass_cnickel_roundtrip(self) -> None:
        assert Graph.from_cnickel("12e|2e|e|:sss").cnickel() == "12e|2e|e|:sss"

    def test_structured_color_string_with_external_legs(self) -> None:
        # 'e11|e|:zzz|z|', colours mirror topology structure including 'e' positions
        g = Graph.from_cnickel("e11|e|:zzz|z|")
        assert len(g.get_internal_edges()) == 2
        assert g.external_legs == 2

    def test_existing_graphs_unaffected(self) -> None:
        for cn in ("11e|e|:zz", "12e|2e|e|:zzz", "111e|e|:zzz"):
            assert Graph.from_cnickel(cn).cnickel() == cn


class TestGraphStringRepresentation:
    """Test Graph string representations."""

    def test_repr_bubble(self) -> None:
        """Test __repr__ for bubble diagram."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
        ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
        ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
        repr_str = repr(graph)

        assert "internal_vertices=2" in repr_str
        assert "external_legs=2" in repr_str
        assert "edges=4" in repr_str
        assert "loops=1" in repr_str

    def test_str_bubble(self) -> None:
        """Test __str__ for bubble diagram."""
        e1 = Edge(idx=1, v1=1, v2=2, is_internal=True)
        e2 = Edge(idx=2, v1=2, v2=1, is_internal=True)
        ex1 = Edge(idx=3, v1=1, v2=3, is_internal=False)
        ex2 = Edge(idx=4, v1=2, v2=4, is_internal=False)

        graph = Graph(internal_vertices=2, external_legs=2, edges=[e1, e2, ex1, ex2])
        str_repr = str(graph)

        assert "2 internal vertices" in str_repr
        assert "2 external legs" in str_repr
        assert "2 propagators" in str_repr
        assert "1 loops" in str_repr
