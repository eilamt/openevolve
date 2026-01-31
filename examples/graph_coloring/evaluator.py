"""
Evaluator for the Graph Coloring example.

This evaluator tests evolved graph coloring algorithms on multiple test graphs
and returns metrics including:
- Number of colors used
- Validity of coloring (no conflicts)
- Performance across different graph types
"""

import importlib.util
import sys
import os
import random
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_program(program_path: str):
    """Dynamically load the evolved program."""
    spec = importlib.util.spec_from_file_location("evolved_module", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_random_graph(num_vertices: int, edge_probability: float, seed: int = None):
    """
    Create a random Erdős–Rényi graph.

    Args:
        num_vertices: Number of vertices
        edge_probability: Probability of edge between any two vertices
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Import Graph class from the module
    from initial_program import Graph

    g = Graph(num_vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                g.add_edge(i, j)
    return g


def create_test_graphs():
    """
    Create a suite of test graphs for evaluation.

    Returns:
        list: List of (name, graph, known_chromatic_number_or_bound) tuples
    """
    from initial_program import Graph

    test_graphs = []

    # 1. Petersen Graph - chromatic number = 3
    petersen = Graph(10)
    for i in range(5):
        petersen.add_edge(i, (i + 1) % 5)
    for i in range(5):
        petersen.add_edge(5 + i, 5 + (i + 2) % 5)
    for i in range(5):
        petersen.add_edge(i, 5 + i)
    test_graphs.append(("Petersen", petersen, 3))

    # 2. Complete graph K5 - chromatic number = 5
    k5 = Graph(5)
    for i in range(5):
        for j in range(i + 1, 5):
            k5.add_edge(i, j)
    test_graphs.append(("K5", k5, 5))

    # 3. Bipartite graph - chromatic number = 2
    bipartite = Graph(10)
    for i in range(5):
        for j in range(5, 10):
            if random.random() < 0.5:
                bipartite.add_edge(i, j)
    test_graphs.append(("Bipartite", bipartite, 2))

    # 4. Cycle graph C7 - chromatic number = 3 (odd cycle)
    cycle = Graph(7)
    for i in range(7):
        cycle.add_edge(i, (i + 1) % 7)
    test_graphs.append(("Cycle7", cycle, 3))

    # 5. Random sparse graph
    sparse = create_random_graph(30, 0.1, seed=42)
    test_graphs.append(("RandomSparse30", sparse, None))

    # 6. Random dense graph
    dense = create_random_graph(20, 0.5, seed=42)
    test_graphs.append(("RandomDense20", dense, None))

    return test_graphs


def evaluate(program_path: str) -> dict:
    """
    Evaluate an evolved graph coloring program.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: Evaluation metrics including combined_score
    """
    try:
        # Load the evolved module
        module = load_program(program_path)

        # Get the coloring function and helper functions
        graph_coloring = module.graph_coloring
        is_valid_coloring = module.is_valid_coloring
        count_colors = module.count_colors

        # Create test graphs
        test_graphs = create_test_graphs()

        total_score = 0
        num_tests = len(test_graphs)
        all_valid = True
        total_colors = 0
        optimal_count = 0

        results_detail = []

        for name, graph, known_chromatic in test_graphs:
            # Time the coloring
            start_time = time.time()
            coloring = graph_coloring(graph)
            elapsed_time = time.time() - start_time

            # Check validity
            is_valid, conflicts = is_valid_coloring(graph, coloring)
            num_colors = count_colors(coloring)

            if not is_valid:
                all_valid = False
                # Invalid coloring gets score of 0 for this graph
                graph_score = 0
            else:
                total_colors += num_colors

                # Score based on colors used
                # If we know optimal, score = optimal / used (max 1.0)
                # Otherwise, use heuristic based on graph size
                if known_chromatic is not None:
                    graph_score = known_chromatic / num_colors
                    if num_colors == known_chromatic:
                        optimal_count += 1
                else:
                    # For unknown optimal, estimate lower bound as max_degree + 1
                    max_degree = max(graph.get_degree(v) for v in range(graph.num_vertices))
                    estimated_lower = max(2, max_degree // 2)
                    graph_score = estimated_lower / num_colors

                graph_score = min(1.0, graph_score)  # Cap at 1.0

            total_score += graph_score
            results_detail.append({
                'name': name,
                'colors': num_colors,
                'valid': is_valid,
                'conflicts': conflicts,
                'score': graph_score,
                'time': elapsed_time
            })

        # Calculate final metrics
        avg_score = total_score / num_tests

        # Combined score: 0 if any invalid, otherwise based on color efficiency
        if not all_valid:
            combined_score = 0.0
        else:
            combined_score = avg_score

        return {
            'combined_score': combined_score,
            'avg_color_score': avg_score,
            'all_valid': all_valid,
            'optimal_count': optimal_count,
            'total_colors': total_colors,
            'num_tests': num_tests,
            'details': results_detail
        }

    except Exception as e:
        # Return minimal score on error
        return {
            'combined_score': 0.0,
            'error': str(e)
        }


# ============================================================
# Cascade Evaluation Stages
# ============================================================

def evaluate_stage1(program_path: str) -> dict:
    """
    Stage 1: Validity Gate - Quick validation check only.

    Tests the algorithm on 3 simple graphs and ONLY checks if colorings
    are valid. No scoring is done. This is the fastest possible rejection
    of broken algorithms.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: combined_score is 1.0 if all valid, 0.0 if any invalid
    """
    try:
        # Load the evolved module
        module = load_program(program_path)

        # Get the coloring function and helper functions
        graph_coloring = module.graph_coloring
        is_valid_coloring = module.is_valid_coloring
        Graph = module.Graph

        # Quick test graphs - small and fast to validate
        quick_tests = []

        # 1. Triangle (K3) - simplest non-trivial graph
        triangle = Graph(3)
        triangle.add_edge(0, 1)
        triangle.add_edge(1, 2)
        triangle.add_edge(0, 2)
        quick_tests.append(("Triangle", triangle))

        # 2. Complete graph K4
        k4 = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                k4.add_edge(i, j)
        quick_tests.append(("K4", k4))

        # 3. Simple path (3 vertices)
        path = Graph(3)
        path.add_edge(0, 1)
        path.add_edge(1, 2)
        quick_tests.append(("Path3", path))

        # Run validity check only - no scoring
        for name, graph in quick_tests:
            coloring = graph_coloring(graph)
            is_valid, conflicts = is_valid_coloring(graph, coloring)

            if not is_valid:
                # Fail fast: any invalid coloring = immediate rejection
                return {
                    'combined_score': 0.0,
                    'stage1_passed': False,
                    'failed_on': name,
                    'conflicts': conflicts
                }

        # All graphs produced valid colorings
        return {
            'combined_score': 1.0,
            'stage1_passed': True
        }

    except Exception as e:
        return {
            'combined_score': 0.0,
            'stage1_passed': False,
            'error': str(e)
        }


def evaluate_stage2(program_path: str) -> dict:
    """
    Stage 2: Quick scoring on small graphs.

    Only runs if stage 1 passed. Computes quality scores on 3 small graphs
    to filter out algorithms that produce valid but inefficient colorings.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: Quick scoring metrics
    """
    try:
        # Load the evolved module
        module = load_program(program_path)

        # Get the coloring function and helper functions
        graph_coloring = module.graph_coloring
        is_valid_coloring = module.is_valid_coloring
        count_colors = module.count_colors
        Graph = module.Graph

        # Same test graphs as stage 1, but now with expected chromatic numbers
        quick_tests = []

        # 1. Triangle (K3) - chromatic number = 3
        triangle = Graph(3)
        triangle.add_edge(0, 1)
        triangle.add_edge(1, 2)
        triangle.add_edge(0, 2)
        quick_tests.append(("Triangle", triangle, 3))

        # 2. Complete graph K4 - chromatic number = 4
        k4 = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                k4.add_edge(i, j)
        quick_tests.append(("K4", k4, 4))

        # 3. Simple path (3 vertices) - chromatic number = 2
        path = Graph(3)
        path.add_edge(0, 1)
        path.add_edge(1, 2)
        quick_tests.append(("Path3", path, 2))

        # Compute scores
        total_score = 0
        all_valid = True

        for name, graph, expected_chromatic in quick_tests:
            coloring = graph_coloring(graph)
            is_valid, conflicts = is_valid_coloring(graph, coloring)
            num_colors = count_colors(coloring)

            if not is_valid:
                all_valid = False
                graph_score = 0
            else:
                graph_score = min(1.0, expected_chromatic / num_colors)

            total_score += graph_score

        avg_score = total_score / len(quick_tests)

        # If any invalid, combined_score = 0
        if not all_valid:
            combined_score = 0.0
        else:
            combined_score = avg_score

        return {
            'combined_score': combined_score,
            'stage2_avg_score': avg_score,
            'stage2_all_valid': all_valid
        }

    except Exception as e:
        return {
            'combined_score': 0.0,
            'error': str(e)
        }


def evaluate_stage3(program_path: str) -> dict:
    """
    Stage 3: Full evaluation on all test graphs.

    Only runs if stage 2 passed threshold. Comprehensive evaluation
    on all 6 test graphs with detailed scoring.
    If any coloring is invalid, combined_score = 0.

    Args:
        program_path: Path to the evolved program file

    Returns:
        dict: Full evaluation metrics with combined_score = 0 if any invalid
    """
    result = evaluate(program_path)

    # Enforce: any invalid coloring = score of 0
    if not result.get('all_valid', False):
        result['combined_score'] = 0.0

    return result


if __name__ == "__main__":
    # Test the 3-stage cascade evaluation with the initial program
    print("=" * 50)
    print("Stage 1: Validity Gate (3 small graphs)")
    print("=" * 50)
    stage1_result = evaluate_stage1("initial_program.py")
    print(f"  Passed: {stage1_result.get('stage1_passed', False)}")
    print(f"  Combined Score: {stage1_result.get('combined_score', 0):.4f}")

    if not stage1_result.get('stage1_passed'):
        print("\n  FAILED - Skipping remaining stages")
        if 'failed_on' in stage1_result:
            print(f"  Failed on: {stage1_result['failed_on']}")
        exit(1)

    print("\n" + "=" * 50)
    print("Stage 2: Quick Scoring (3 small graphs)")
    print("=" * 50)
    stage2_result = evaluate_stage2("initial_program.py")
    print(f"  Combined Score: {stage2_result.get('combined_score', 0):.4f}")
    print(f"  Avg Score: {stage2_result.get('stage2_avg_score', 0):.4f}")
    print(f"  All Valid: {stage2_result.get('stage2_all_valid', False)}")

    print("\n" + "=" * 50)
    print("Stage 3: Full Evaluation (6 graphs)")
    print("=" * 50)
    stage3_result = evaluate_stage3("initial_program.py")
    print(f"  Combined Score: {stage3_result.get('combined_score', 0):.4f}")
    print(f"  Avg Color Score: {stage3_result.get('avg_color_score', 0):.4f}")
    print(f"  All Valid: {stage3_result.get('all_valid', False)}")
    print(f"  Optimal Count: {stage3_result.get('optimal_count', 0)}/{stage3_result.get('num_tests', 0)}")

    if 'details' in stage3_result:
        print("\nPer-graph results:")
        for detail in stage3_result['details']:
            print(f"  {detail['name']}: {detail['colors']} colors, "
                  f"valid={detail['valid']}, score={detail['score']:.3f}")
