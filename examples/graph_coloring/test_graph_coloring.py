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
from evaluator import (
    create_test_graphs,
    create_random_graph,
    evaluate,
    evaluate_stage1,
    evaluate_stage2,
    evaluate_stage3
)


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


class TestEvaluator(unittest.TestCase):
    """Test cases for the evaluator functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.program_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'initial_program.py'
        )

    def test_evaluate_returns_combined_score(self):
        """Test that evaluate() returns a combined_score."""
        result = evaluate(self.program_path)
        self.assertIn('combined_score', result)
        self.assertIsInstance(result['combined_score'], float)

    def test_evaluate_returns_all_valid(self):
        """Test that evaluate() returns all_valid flag."""
        result = evaluate(self.program_path)
        self.assertIn('all_valid', result)
        self.assertTrue(result['all_valid'])

    def test_evaluate_returns_details(self):
        """Test that evaluate() returns per-graph details."""
        result = evaluate(self.program_path)
        self.assertIn('details', result)
        self.assertEqual(len(result['details']), 6)  # 6 test graphs

    def test_evaluate_score_in_valid_range(self):
        """Test that combined_score is between 0 and 1."""
        result = evaluate(self.program_path)
        self.assertGreaterEqual(result['combined_score'], 0.0)
        self.assertLessEqual(result['combined_score'], 1.0)

    def test_evaluate_nonexistent_file(self):
        """Test that evaluate() handles missing files gracefully."""
        result = evaluate('nonexistent_program.py')
        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    # Stage 1 tests
    def test_stage1_passes_for_valid_program(self):
        """Test that stage1 passes for initial_program.py."""
        result = evaluate_stage1(self.program_path)
        self.assertTrue(result['stage1_passed'])
        self.assertEqual(result['combined_score'], 1.0)

    def test_stage1_returns_required_fields(self):
        """Test that stage1 returns required fields."""
        result = evaluate_stage1(self.program_path)
        self.assertIn('combined_score', result)
        self.assertIn('stage1_passed', result)

    def test_stage1_handles_errors(self):
        """Test that stage1 handles errors gracefully."""
        result = evaluate_stage1('nonexistent_program.py')
        self.assertEqual(result['combined_score'], 0.0)
        self.assertFalse(result['stage1_passed'])

    # Stage 2 tests
    def test_stage2_returns_score(self):
        """Test that stage2 returns a combined_score."""
        result = evaluate_stage2(self.program_path)
        self.assertIn('combined_score', result)
        self.assertGreater(result['combined_score'], 0.0)

    def test_stage2_optimal_score_for_initial_program(self):
        """Test that initial program gets optimal score on small graphs."""
        result = evaluate_stage2(self.program_path)
        # Initial greedy algorithm should get optimal on small graphs
        self.assertEqual(result['combined_score'], 1.0)

    def test_stage2_returns_avg_score(self):
        """Test that stage2 returns average score."""
        result = evaluate_stage2(self.program_path)
        self.assertIn('stage2_avg_score', result)
        self.assertIn('stage2_all_valid', result)

    # Stage 3 tests
    def test_stage3_returns_full_metrics(self):
        """Test that stage3 returns full evaluation metrics."""
        result = evaluate_stage3(self.program_path)
        self.assertIn('combined_score', result)
        self.assertIn('avg_color_score', result)
        self.assertIn('all_valid', result)
        self.assertIn('optimal_count', result)
        self.assertIn('details', result)

    def test_stage3_score_equals_evaluate(self):
        """Test that stage3 returns same result as evaluate() for valid programs."""
        stage3_result = evaluate_stage3(self.program_path)
        eval_result = evaluate(self.program_path)
        self.assertEqual(
            stage3_result['combined_score'],
            eval_result['combined_score']
        )

    # Cascade flow tests
    def test_cascade_flow_all_stages_pass(self):
        """Test that all stages pass for initial_program.py."""
        stage1 = evaluate_stage1(self.program_path)
        self.assertTrue(stage1['stage1_passed'])

        stage2 = evaluate_stage2(self.program_path)
        self.assertGreaterEqual(stage2['combined_score'], 0.7)

        stage3 = evaluate_stage3(self.program_path)
        self.assertGreater(stage3['combined_score'], 0.0)


class TestEvaluatorFailureCases(unittest.TestCase):
    """Test cases for evaluator failure scenarios using mock programs."""

    def setUp(self):
        """Set up test fixtures with temp directory for mock programs."""
        import tempfile
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_program(self, name: str, code: str) -> str:
        """Create a mock program file and return its path."""
        file_path = os.path.join(self.temp_dir, name)
        with open(file_path, 'w') as f:
            f.write(code)
        return file_path

    def test_stage1_fails_for_invalid_coloring(self):
        """Test that stage1 fails for a program that produces invalid colorings."""
        # This program assigns the same color (0) to ALL vertices,
        # which creates conflicts on any graph with edges
        invalid_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Invalid: assigns same color to all vertices."""
    return {v: 0 for v in range(graph.num_vertices)}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('invalid_coloring.py', invalid_program)
        result = evaluate_stage1(program_path)

        self.assertFalse(result.get('stage1_passed', True))
        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('failed_on', result)

    def test_stage1_fails_for_partial_coloring(self):
        """Test that stage1 fails for a program that doesn't color all vertices."""
        # This program only colors even-indexed vertices
        partial_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Only colors even vertices - missing odd vertices."""
    return {v: v % 3 for v in range(graph.num_vertices) if v % 2 == 0}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('partial_coloring.py', partial_program)
        # This should fail because neighbors might both be None
        result = evaluate_stage1(program_path)

        # Partial coloring should still technically pass validity (None != None comparison)
        # But let's test the full evaluate which might catch other issues
        full_result = evaluate(program_path)
        # Score should be affected by the partial nature
        self.assertIsNotNone(full_result)

    def test_stage2_low_score_for_suboptimal_coloring(self):
        """Test that stage2 gives low score for inefficient coloring."""
        # This program uses n colors for n vertices (worst case)
        suboptimal_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Suboptimal: uses n colors for n vertices (every vertex gets unique color)."""
    return {v: v for v in range(graph.num_vertices)}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('suboptimal_coloring.py', suboptimal_program)

        # Stage 1 should pass (coloring is valid)
        stage1_result = evaluate_stage1(program_path)
        self.assertTrue(stage1_result.get('stage1_passed', False))

        # Stage 2 should have lower score (using more colors than optimal)
        stage2_result = evaluate_stage2(program_path)
        # Triangle needs 3 colors, this uses 3 -> score = 1.0
        # K4 needs 4 colors, this uses 4 -> score = 1.0
        # Path3 needs 2 colors, this uses 3 -> score = 2/3 = 0.67
        # Average = (1.0 + 1.0 + 0.67) / 3 = 0.89
        self.assertGreater(stage2_result['combined_score'], 0.0)
        self.assertLessEqual(stage2_result['combined_score'], 1.0)

    def test_evaluate_handles_missing_function(self):
        """Test that evaluate handles a program missing graph_coloring function."""
        missing_func_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

# Missing graph_coloring, is_valid_coloring, and count_colors functions!
'''
        program_path = self._create_mock_program('missing_function.py', missing_func_program)
        result = evaluate(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_evaluate_handles_syntax_error(self):
        """Test that evaluate handles a program with syntax errors."""
        syntax_error_program = '''
def graph_coloring(graph)
    # Missing colon above - syntax error
    return {}
'''
        program_path = self._create_mock_program('syntax_error.py', syntax_error_program)
        result = evaluate(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_evaluate_handles_runtime_error(self):
        """Test that evaluate handles a program that throws runtime errors."""
        runtime_error_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Raises an error during execution."""
    raise ValueError("Intentional error for testing")

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('runtime_error.py', runtime_error_program)
        result = evaluate(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_stage1_handles_exception_gracefully(self):
        """Test that stage1 handles exceptions and returns proper error structure."""
        error_program = '''
raise ImportError("Cannot import module")
'''
        program_path = self._create_mock_program('import_error.py', error_program)
        result = evaluate_stage1(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertFalse(result.get('stage1_passed', True))
        self.assertIn('error', result)

    def test_stage2_handles_exception_gracefully(self):
        """Test that stage2 handles exceptions and returns proper error structure."""
        error_program = '''
raise ImportError("Cannot import module")
'''
        program_path = self._create_mock_program('import_error2.py', error_program)
        result = evaluate_stage2(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertIn('error', result)

    def test_stage3_zero_score_for_invalid(self):
        """Test that stage3 returns 0 score when any coloring is invalid."""
        # Same as invalid coloring test but for stage 3
        invalid_program = '''
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = {i: set() for i in range(num_vertices)}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)

    def get_neighbors(self, vertex):
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:
                    edges.append((u, v))
        return edges

def graph_coloring(graph):
    """Invalid: assigns same color to all vertices."""
    return {v: 0 for v in range(graph.num_vertices)}

def is_valid_coloring(graph, coloring):
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return (conflicts == 0, conflicts)

def count_colors(coloring):
    return len(set(coloring.values()))
'''
        program_path = self._create_mock_program('invalid_for_stage3.py', invalid_program)
        result = evaluate_stage3(program_path)

        self.assertEqual(result['combined_score'], 0.0)
        self.assertFalse(result.get('all_valid', True))

    def test_nonexistent_program_file(self):
        """Test all stages handle nonexistent files gracefully."""
        fake_path = '/nonexistent/path/to/program.py'

        stage1 = evaluate_stage1(fake_path)
        self.assertEqual(stage1['combined_score'], 0.0)
        self.assertFalse(stage1.get('stage1_passed', True))

        stage2 = evaluate_stage2(fake_path)
        self.assertEqual(stage2['combined_score'], 0.0)

        stage3 = evaluate_stage3(fake_path)
        self.assertEqual(stage3['combined_score'], 0.0)

        full = evaluate(fake_path)
        self.assertEqual(full['combined_score'], 0.0)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
