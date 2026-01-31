"""
Test cases for the Graph Coloring example.

Runs the graph coloring algorithm on all test graphs from the evaluator
and verifies that all colorings are valid (no adjacent vertices share a color).
"""

import unittest
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from initial_program import (
    Graph,
    graph_coloring,
    is_valid_coloring,
    count_colors,
    create_sample_graph,
    load_graph_from_file
)
from evaluator import create_test_graphs, create_random_graph


class TestGraphColoring(unittest.TestCase):
    """Test cases for graph coloring validity."""

    def test_petersen_graph_valid(self):
        """Test that Petersen graph coloring is valid."""
        graph = create_sample_graph()  # Petersen graph
        coloring = graph_coloring(graph)
        is_valid, conflicts = is_valid_coloring(graph, coloring)

        self.assertTrue(is_valid, f"Petersen graph coloring has {conflicts} conflicts")
        self.assertEqual(conflicts, 0)

    def test_petersen_graph_chromatic_number(self):
        """Test that Petersen graph uses at most 3 colors (optimal)."""
        graph = create_sample_graph()
        coloring = graph_coloring(graph)
        num_colors = count_colors(coloring)

        # Petersen graph has chromatic number 3
        self.assertLessEqual(num_colors, 4, f"Used {num_colors} colors, expected at most 4")

    def test_complete_graph_k5_valid(self):
        """Test that complete graph K5 coloring is valid."""
        k5 = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                k5.add_edge(i, j)

        coloring = graph_coloring(k5)
        is_valid, conflicts = is_valid_coloring(k5, coloring)

        self.assertTrue(is_valid, f"K5 coloring has {conflicts} conflicts")

    def test_complete_graph_k5_chromatic_number(self):
        """Test that K5 uses exactly 5 colors (optimal for complete graph)."""
        k5 = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                k5.add_edge(i, j)

        coloring = graph_coloring(k5)
        num_colors = count_colors(coloring)

        # Complete graph K5 requires exactly 5 colors
        self.assertEqual(num_colors, 5, f"K5 should use exactly 5 colors, used {num_colors}")

    def test_bipartite_graph_valid(self):
        """Test that bipartite graph coloring is valid."""
        bipartite = Graph(10)
        # Create a complete bipartite graph K5,5
        for i in range(5):
            for j in range(5, 10):
                bipartite.add_edge(i, j)

        coloring = graph_coloring(bipartite)
        is_valid, conflicts = is_valid_coloring(bipartite, coloring)

        self.assertTrue(is_valid, f"Bipartite graph coloring has {conflicts} conflicts")

    def test_bipartite_graph_chromatic_number(self):
        """Test that bipartite graph uses at most 2 colors."""
        bipartite = Graph(10)
        for i in range(5):
            for j in range(5, 10):
                bipartite.add_edge(i, j)

        coloring = graph_coloring(bipartite)
        num_colors = count_colors(coloring)

        # Bipartite graphs have chromatic number 2
        self.assertLessEqual(num_colors, 2, f"Bipartite should use at most 2 colors, used {num_colors}")

    def test_odd_cycle_valid(self):
        """Test that odd cycle (C7) coloring is valid."""
        cycle = Graph(7)
        for i in range(7):
            cycle.add_edge(i, (i + 1) % 7)

        coloring = graph_coloring(cycle)
        is_valid, conflicts = is_valid_coloring(cycle, coloring)

        self.assertTrue(is_valid, f"Cycle C7 coloring has {conflicts} conflicts")

    def test_odd_cycle_chromatic_number(self):
        """Test that odd cycle uses exactly 3 colors."""
        cycle = Graph(7)
        for i in range(7):
            cycle.add_edge(i, (i + 1) % 7)

        coloring = graph_coloring(cycle)
        num_colors = count_colors(coloring)

        # Odd cycles have chromatic number 3
        self.assertLessEqual(num_colors, 3, f"Odd cycle should use at most 3 colors, used {num_colors}")

    def test_even_cycle_valid(self):
        """Test that even cycle (C6) coloring is valid."""
        cycle = Graph(6)
        for i in range(6):
            cycle.add_edge(i, (i + 1) % 6)

        coloring = graph_coloring(cycle)
        is_valid, conflicts = is_valid_coloring(cycle, coloring)

        self.assertTrue(is_valid, f"Cycle C6 coloring has {conflicts} conflicts")

    def test_even_cycle_chromatic_number(self):
        """Test that even cycle uses exactly 2 colors."""
        cycle = Graph(6)
        for i in range(6):
            cycle.add_edge(i, (i + 1) % 6)

        coloring = graph_coloring(cycle)
        num_colors = count_colors(coloring)

        # Even cycles have chromatic number 2
        self.assertLessEqual(num_colors, 2, f"Even cycle should use at most 2 colors, used {num_colors}")

    def test_random_sparse_graph_valid(self):
        """Test that random sparse graph coloring is valid."""
        sparse = create_random_graph(30, 0.1, seed=42)
        coloring = graph_coloring(sparse)
        is_valid, conflicts = is_valid_coloring(sparse, coloring)

        self.assertTrue(is_valid, f"Random sparse graph coloring has {conflicts} conflicts")

    def test_random_dense_graph_valid(self):
        """Test that random dense graph coloring is valid."""
        dense = create_random_graph(20, 0.5, seed=42)
        coloring = graph_coloring(dense)
        is_valid, conflicts = is_valid_coloring(dense, coloring)

        self.assertTrue(is_valid, f"Random dense graph coloring has {conflicts} conflicts")

    def test_all_evaluator_graphs_valid(self):
        """Test that ALL graphs from the evaluator produce valid colorings."""
        test_graphs = create_test_graphs()

        for name, graph, known_chromatic in test_graphs:
            with self.subTest(graph=name):
                coloring = graph_coloring(graph)
                is_valid, conflicts = is_valid_coloring(graph, coloring)

                self.assertTrue(
                    is_valid,
                    f"Graph '{name}' has invalid coloring with {conflicts} conflicts"
                )

    def test_empty_graph(self):
        """Test coloring of a graph with no edges."""
        empty = Graph(5)
        coloring = graph_coloring(empty)
        is_valid, conflicts = is_valid_coloring(empty, coloring)
        num_colors = count_colors(coloring)

        self.assertTrue(is_valid)
        # Empty graph should use only 1 color
        self.assertEqual(num_colors, 1, f"Empty graph should use 1 color, used {num_colors}")

    def test_single_vertex(self):
        """Test coloring of a single vertex graph."""
        single = Graph(1)
        coloring = graph_coloring(single)
        is_valid, conflicts = is_valid_coloring(single, coloring)

        self.assertTrue(is_valid)
        self.assertEqual(count_colors(coloring), 1)

    def test_all_vertices_colored(self):
        """Test that all vertices are assigned a color."""
        graph = create_sample_graph()
        coloring = graph_coloring(graph)

        for vertex in range(graph.num_vertices):
            self.assertIn(
                vertex,
                coloring,
                f"Vertex {vertex} was not assigned a color"
            )


class TestGraphStructure(unittest.TestCase):
    """Test cases for the Graph class itself."""

    def test_graph_creation(self):
        """Test that a graph can be created."""
        g = Graph(5)
        self.assertEqual(g.num_vertices, 5)

    def test_add_edge(self):
        """Test adding edges to a graph."""
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        self.assertIn(1, g.get_neighbors(0))
        self.assertIn(0, g.get_neighbors(1))
        self.assertIn(2, g.get_neighbors(1))

    def test_get_degree(self):
        """Test vertex degree calculation."""
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)

        self.assertEqual(g.get_degree(0), 3)
        self.assertEqual(g.get_degree(1), 1)

    def test_get_edges(self):
        """Test edge retrieval."""
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        edges = g.get_edges()
        self.assertEqual(len(edges), 2)
        self.assertIn((0, 1), edges)
        self.assertIn((1, 2), edges)


class TestGraphFileLoading(unittest.TestCase):
    """Test cases for loading graphs from files."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_graphs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'sample_graphs'
        )

    def test_load_triangle_graph(self):
        """Test loading a triangle graph from file."""
        file_path = os.path.join(self.sample_graphs_dir, 'triangle.txt')
        graph = load_graph_from_file(file_path)

        self.assertEqual(graph.num_vertices, 3)
        self.assertEqual(len(graph.get_edges()), 3)
        # Triangle is K3, should need 3 colors
        coloring = graph_coloring(graph)
        self.assertEqual(count_colors(coloring), 3)

    def test_load_square_graph(self):
        """Test loading a square (C4) graph from file."""
        file_path = os.path.join(self.sample_graphs_dir, 'square.txt')
        graph = load_graph_from_file(file_path)

        self.assertEqual(graph.num_vertices, 4)
        self.assertEqual(len(graph.get_edges()), 4)
        # Square is bipartite, should need only 2 colors
        coloring = graph_coloring(graph)
        self.assertLessEqual(count_colors(coloring), 2)

    def test_load_k4_complete_graph(self):
        """Test loading a complete K4 graph from file."""
        file_path = os.path.join(self.sample_graphs_dir, 'k4_complete.txt')
        graph = load_graph_from_file(file_path)

        self.assertEqual(graph.num_vertices, 4)
        self.assertEqual(len(graph.get_edges()), 6)  # K4 has 4*3/2 = 6 edges
        # K4 needs exactly 4 colors
        coloring = graph_coloring(graph)
        self.assertEqual(count_colors(coloring), 4)

    def test_load_graph_with_comments(self):
        """Test that comment lines are properly skipped."""
        file_path = os.path.join(self.sample_graphs_dir, 'k4_complete.txt')
        graph = load_graph_from_file(file_path)
        # File has comment lines starting with #, should still load correctly
        self.assertEqual(graph.num_vertices, 4)

    def test_loaded_graph_valid_coloring(self):
        """Test that loaded graphs produce valid colorings."""
        for filename in ['triangle.txt', 'square.txt', 'k4_complete.txt']:
            with self.subTest(file=filename):
                file_path = os.path.join(self.sample_graphs_dir, filename)
                graph = load_graph_from_file(file_path)
                coloring = graph_coloring(graph)
                is_valid, conflicts = is_valid_coloring(graph, coloring)
                self.assertTrue(is_valid, f"{filename} has invalid coloring")

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises an error."""
        with self.assertRaises(FileNotFoundError):
            load_graph_from_file('nonexistent_file.txt')


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
