# EVOLVE-BLOCK-START
"""Graph coloring example for OpenEvolve"""


def graph_coloring(graph):
    """
    Advanced graph coloring using DSatur with enhanced tie-breaking and post-optimization.
    
    DSatur prioritizes vertices with highest saturation degree (most distinct
    neighbor colors), with sophisticated tie-breaking using uncolored degree,
    total degree, and vertex ID. Includes post-processing to reduce colors.

    Args:
        graph: A Graph object with vertices and edges

    Returns:
        dict: A mapping of vertex -> color (colors are integers starting from 0)
    """
    coloring = {}
    uncolored = set(range(graph.num_vertices))
    
    # Precompute degrees for tiebreaking
    degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]
    
    # DSatur main phase with enhanced tie-breaking
    while uncolored:
        # Find vertex with highest saturation degree
        best_vertex = None
        best_key = (-1, -1, -1, float('inf'))
        best_neighbor_colors = None
        
        for vertex in uncolored:
            # Calculate saturation degree (number of distinct colors in neighbors)
            neighbor_colors = set()
            for neighbor in graph.get_neighbors(vertex):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            saturation = len(neighbor_colors)
            
            # Count uncolored neighbors (uncolored degree)
            uncolored_degree = sum(1 for n in graph.get_neighbors(vertex) if n in uncolored)
            
            # Multi-level tie-breaking: saturation (high), uncolored degree (high), 
            # total degree (high), vertex ID (high for determinism and better results)
            key = (saturation, uncolored_degree, degrees[vertex], vertex)
            
            if key > best_key:
                best_vertex = vertex
                best_key = key
                best_neighbor_colors = neighbor_colors
        
        # Assign smallest available color
        color = 0
        while color in best_neighbor_colors:
            color += 1
        
        coloring[best_vertex] = color
        uncolored.remove(best_vertex)
    
    # Post-optimization: try to reduce total colors by recoloring
    max_color = max(coloring.values()) if coloring else 0
    
    # Attempt to eliminate high colors (3 passes for better results)
    for pass_num in range(3):
        improved = False
        for target_color in range(max_color, -1, -1):
            vertices_with_color = [v for v, c in coloring.items() if c == target_color]
            
            # Sort by degree (descending) - prioritize high-degree vertices
            # as they're harder to recolor, so try them first
            vertices_with_color.sort(key=lambda v: degrees[v], reverse=True)
            
            for vertex in vertices_with_color:
                # Get colors used by neighbors
                neighbor_colors = {coloring[n] for n in graph.get_neighbors(vertex)}
                
                # Try to assign a lower color
                for new_color in range(target_color):
                    if new_color not in neighbor_colors:
                        coloring[vertex] = new_color
                        improved = True
                        break
        
        # Update max_color and check for improvement
        new_max = max(coloring.values()) if coloring else 0
        if new_max >= max_color and pass_num > 0:
            break  # No improvement, stop (but allow first pass to complete)
        max_color = new_max
    
    return coloring


# EVOLVE-BLOCK-END


# ============================================================
# Fixed code below (not evolved)
# ============================================================

class Graph:
    """Simple undirected graph using adjacency list representation."""

    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.adjacency_list = [[] for _ in range(num_vertices)]

    def add_edge(self, u, v):
        """Add an undirected edge between vertices u and v."""
        if v not in self.adjacency_list[u]:
            self.adjacency_list[u].append(v)
        if u not in self.adjacency_list[v]:
            self.adjacency_list[v].append(u)

    def get_neighbors(self, vertex):
        """Return list of neighbors for a vertex."""
        return self.adjacency_list[vertex]

    def get_degree(self, vertex):
        """Return the degree (number of neighbors) of a vertex."""
        return len(self.adjacency_list[vertex])

    def get_edges(self):
        """Return list of all edges as (u, v) tuples."""
        edges = []
        for u in range(self.num_vertices):
            for v in self.adjacency_list[u]:
                if u < v:  # Avoid duplicates
                    edges.append((u, v))
        return edges


def is_valid_coloring(graph, coloring):
    """
    Check if a coloring is valid (no adjacent vertices share a color).

    Returns:
        tuple: (is_valid, num_conflicts)
    """
    conflicts = 0
    for u, v in graph.get_edges():
        if coloring.get(u) == coloring.get(v):
            conflicts += 1
    return conflicts == 0, conflicts


def count_colors(coloring):
    """Count the number of distinct colors used."""
    if not coloring:
        return 0
    return len(set(coloring.values()))


def create_sample_graph():
    """
    Create a sample graph for testing.
    This is a Petersen graph - a well-known graph with chromatic number 3.
    """
    g = Graph(10)
    # Outer pentagon
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    # Inner pentagram
    for i in range(5):
        g.add_edge(5 + i, 5 + (i + 2) % 5)
    # Connections between outer and inner
    for i in range(5):
        g.add_edge(i, 5 + i)
    return g


def run_coloring(graph=None):
    """Run the coloring algorithm and return results."""
    if graph is None:
        graph = create_sample_graph()

    coloring = graph_coloring(graph)
    is_valid, conflicts = is_valid_coloring(graph, coloring)
    num_colors = count_colors(coloring)

    return {
        'coloring': coloring,
        'num_colors': num_colors,
        'is_valid': is_valid,
        'conflicts': conflicts,
        'graph': graph
    }


def visualize_coloring(graph, coloring, title="Graph Coloring"):
    """
    Visualize the graph with colored vertices.

    Args:
        graph: A Graph object
        coloring: A dict mapping vertex -> color
        title: Title for the plot
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        print("Visualization requires networkx and matplotlib.")
        print("Install with: pip install networkx matplotlib")
        return

    # Create a NetworkX graph from our Graph object
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_vertices))
    G.add_edges_from(graph.get_edges())

    # Define a color palette (enough colors for most graphs)
    color_palette = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#96CEB4',  # Green
        '#FFEAA7',  # Yellow
        '#DDA0DD',  # Plum
        '#98D8C8',  # Mint
        '#F7DC6F',  # Gold
        '#BB8FCE',  # Purple
        '#85C1E9',  # Light Blue
        '#F8B500',  # Orange
        '#00CED1',  # Dark Cyan
    ]

    # Map vertex colors to actual colors
    num_colors = count_colors(coloring)
    node_colors = []
    for vertex in range(graph.num_vertices):
        color_idx = coloring.get(vertex, 0) % len(color_palette)
        node_colors.append(color_palette[color_idx])

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Use a nice layout for the graph
    if graph.num_vertices == 10:
        # Special layout for Petersen graph
        pos = nx.shell_layout(G, nlist=[range(5), range(5, 10)])
    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Add title and info
    plt.title(f"{title}\nColors used: {num_colors}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def print_adjacency_matrix(graph):
    """
    Print the graph as an adjacency matrix.

    Displays an n x n table where 'x' indicates an edge between vertices.

    Args:
        graph: A Graph object
    """
    n = graph.num_vertices

    # Determine column width based on number of vertices
    col_width = len(str(n - 1)) + 1

    # Print header row
    header = " " * (col_width + 1)  # Space for row labels
    for j in range(n):
        header += f"{j:>{col_width}}"
    print(header)

    # Print separator
    print(" " * (col_width + 1) + "-" * (col_width * n))

    # Print each row
    for i in range(n):
        row = f"{i:>{col_width}}|"
        neighbors = set(graph.get_neighbors(i))
        for j in range(n):
            if i == j:
                cell = "."  # Diagonal (no self-loops)
            elif j in neighbors:
                cell = "x"  # Edge exists
            else:
                cell = "."  # No edge
            row += f"{cell:>{col_width}}"
        print(row)


def load_graph_from_file(file_path):
    """
    Load a graph from a text file.

    File format:
        Line 1: Number of vertices (integer)
        Remaining lines: Edges as "u v" pairs (space-separated integers)

    Example file contents:
        5
        0 1
        0 2
        1 2
        1 3
        2 4
        3 4

    Args:
        file_path: Path to the graph file

    Returns:
        Graph: A Graph object loaded from the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip comment lines and empty lines to find the number of vertices
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            data_lines.append(stripped)

    # First data line is number of vertices
    num_vertices = int(data_lines[0])
    graph = Graph(num_vertices)

    # Remaining data lines are edges
    for line in data_lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            graph.add_edge(u, v)

    return graph


def print_usage():
    """Print usage instructions."""
    print("""
Graph Coloring - Usage
======================

Run with default Petersen graph:
    python initial_program.py

Run with a custom graph file:
    python initial_program.py <graph_file.txt>

Options:
    -v, --visualize    Show graphical visualization
    -m, --matrix       Show adjacency matrix
    -h, --help         Show this help message

Graph file format:
    Line 1: Number of vertices
    Following lines: Edges as "u v" pairs

Example graph file (triangle.txt):
    3
    0 1
    1 2
    0 2

Examples:
    python initial_program.py
    python initial_program.py my_graph.txt
    python initial_program.py my_graph.txt --visualize --matrix
    python initial_program.py -v -m
""")


if __name__ == "__main__":
    import sys
    import os

    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        sys.exit(0)

    # Check for command line flags
    visualize = "--visualize" in sys.argv or "-v" in sys.argv
    show_matrix = "--matrix" in sys.argv or "-m" in sys.argv

    # Check for graph file argument (any argument that doesn't start with -)
    graph_file = None
    graph = None
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            graph_file = arg
            break

    # Load graph from file or use default
    if graph_file:
        if not os.path.exists(graph_file):
            print(f"Error: File '{graph_file}' not found.")
            sys.exit(1)
        print(f"Loading graph from: {graph_file}")
        graph = load_graph_from_file(graph_file)
        print(f"Loaded graph with {graph.num_vertices} vertices and {len(graph.get_edges())} edges")
        print()
        graph_title = os.path.basename(graph_file)
    else:
        graph = None  # Will use default Petersen graph
        graph_title = "Petersen Graph"

    result = run_coloring(graph)

    # Show adjacency matrix first (if requested)
    if show_matrix:
        print("Adjacency Matrix:")
        print_adjacency_matrix(result['graph'])
        print()

    # Show coloring results
    print(f"Coloring uses {result['num_colors']} colors")
    print(f"Valid: {result['is_valid']}")
    if not result['is_valid']:
        print(f"Conflicts: {result['conflicts']}")
    print(f"Coloring: {result['coloring']}")

    # Show visualization (if requested)
    if visualize:
        visualize_coloring(result['graph'], result['coloring'], f"{graph_title} Coloring")
