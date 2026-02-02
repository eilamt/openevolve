"""Analyze post-processing effectiveness across all DIMACS benchmarks."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from initial_program import Graph

def load_dimacs_graph(file_path):
    """Load a DIMACS format graph."""
    graph = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                num_vertices = int(parts[2])
                graph = Graph(num_vertices)
            elif line.startswith('e'):
                parts = line.split()
                u, v = int(parts[1]) - 1, int(parts[2]) - 1  # Convert to 0-indexed
                graph.add_edge(u, v)
    return graph


def dsatur_coloring(graph):
    """DSatur without post-processing (baseline)."""
    coloring = {}
    uncolored = set(range(graph.num_vertices))
    degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]

    while uncolored:
        best_vertex = None
        best_key = (-1, -1, -1, float('inf'))
        best_neighbor_colors = None

        for vertex in uncolored:
            neighbor_colors = set()
            for neighbor in graph.get_neighbors(vertex):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])

            saturation = len(neighbor_colors)
            uncolored_degree = sum(1 for n in graph.get_neighbors(vertex) if n in uncolored)
            key = (saturation, uncolored_degree, degrees[vertex], vertex)

            if key > best_key:
                best_vertex = vertex
                best_key = key
                best_neighbor_colors = neighbor_colors

        color = 0
        while color in best_neighbor_colors:
            color += 1

        coloring[best_vertex] = color
        uncolored.remove(best_vertex)

    return coloring


def dsatur_with_postprocessing(graph):
    """DSatur WITH post-processing - returns coloring and recoloring stats."""
    # First run DSatur
    coloring = dsatur_coloring(graph)
    initial_coloring = dict(coloring)
    
    degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]
    
    # Track recolorings
    recolorings = []
    
    # Post-processing (same as evolved program)
    max_color = max(coloring.values()) if coloring else 0
    
    for target_color in range(max_color, max(0, max_color - 2), -1):
        vertices_with_color = [v for v, c in coloring.items() if c == target_color]
        
        for vertex in vertices_with_color:
            neighbor_colors = {coloring[n] for n in graph.get_neighbors(vertex)}
            old_color = coloring[vertex]
            
            for new_color in range(target_color):
                if new_color not in neighbor_colors:
                    coloring[vertex] = new_color
                    recolorings.append({
                        'vertex': vertex,
                        'old_color': old_color,
                        'new_color': new_color,
                        'neighbor_colors': neighbor_colors
                    })
                    break
    
    return coloring, initial_coloring, recolorings


def analyze_all_benchmarks():
    """Run analysis on all DIMACS benchmarks."""
    benchmark_dir = os.path.join(os.path.dirname(__file__), 'benchmarks', 'full')
    
    if not os.path.exists(benchmark_dir):
        print(f"Benchmark directory not found: {benchmark_dir}")
        return
    
    results = []
    
    print("=" * 90)
    print(f"{'Graph':<20} {'Vertices':>8} {'Before':>8} {'After':>8} {'Saved':>6} {'Recolorings':>12}")
    print("=" * 90)
    
    total_recolorings = 0
    total_colors_saved = 0
    
    for filename in sorted(os.listdir(benchmark_dir)):
        if not filename.endswith('.col'):
            continue
        
        filepath = os.path.join(benchmark_dir, filename)
        graph = load_dimacs_graph(filepath)
        
        if graph is None:
            continue
        
        # Run both versions
        final_coloring, initial_coloring, recolorings = dsatur_with_postprocessing(graph)
        
        colors_before = len(set(initial_coloring.values()))
        colors_after = len(set(final_coloring.values()))
        colors_saved = colors_before - colors_after
        num_recolorings = len(recolorings)
        
        total_recolorings += num_recolorings
        total_colors_saved += colors_saved
        
        # Print result
        saved_str = f"-{colors_saved}" if colors_saved > 0 else "0"
        print(f"{filename:<20} {graph.num_vertices:>8} {colors_before:>8} {colors_after:>8} {saved_str:>6} {num_recolorings:>12}")
        
        # Store detailed results
        results.append({
            'graph': filename,
            'vertices': graph.num_vertices,
            'colors_before': colors_before,
            'colors_after': colors_after,
            'colors_saved': colors_saved,
            'recolorings': recolorings
        })
    
    print("=" * 90)
    print(f"{'TOTAL':<20} {'':<8} {'':<8} {'':<8} {total_colors_saved:>6} {total_recolorings:>12}")
    print("=" * 90)
    
    # Print detailed recoloring info for graphs where it helped
    print("\n" + "=" * 90)
    print("DETAILED RECOLORING INFO (graphs where post-processing helped):")
    print("=" * 90)
    
    for r in results:
        if r['colors_saved'] > 0:
            print(f"\n{r['graph']} - Saved {r['colors_saved']} color(s):")
            for rc in r['recolorings'][:10]:  # Show first 10
                print(f"  Vertex {rc['vertex']}: color {rc['old_color']} → {rc['new_color']}")
                print(f"    Neighbor colors were: {rc['neighbor_colors']}")
            if len(r['recolorings']) > 10:
                print(f"  ... and {len(r['recolorings']) - 10} more recolorings")


if __name__ == '__main__':
    analyze_all_benchmarks()
