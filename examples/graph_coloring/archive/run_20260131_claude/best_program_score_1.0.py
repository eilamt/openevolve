# EVOLVE-BLOCK-START
"""Graph coloring example for OpenEvolve"""
import random


def graph_coloring(graph):
    """
    Hybrid multi-strategy graph coloring.

    Tries multiple algorithms and random orderings, keeps the best result.
    """
    best_coloring = None
    best_num_colors = float('inf')

    # Strategy 1: DSatur with many random tie-breaks
    for trial in range(30):
        coloring = dsatur_with_random_tiebreak(graph, seed=trial * 13)
        num_colors = len(set(coloring.values())) if coloring else float('inf')
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_coloring = coloring.copy()

    # Strategy 2: Simple greedy with random vertex orderings
    for trial in range(30):
        coloring = greedy_random_order(graph, seed=trial * 17 + 1000)
        num_colors = len(set(coloring.values())) if coloring else float('inf')
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_coloring = coloring.copy()

    # Strategy 3: Largest degree first with random tie-breaks
    for trial in range(20):
        coloring = largest_degree_first(graph, seed=trial * 19 + 2000)
        num_colors = len(set(coloring.values())) if coloring else float('inf')
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_coloring = coloring.copy()

    return best_coloring


def dsatur_with_random_tiebreak(graph, seed=None):
    """DSatur algorithm with randomized tie-breaking."""
    if seed is not None:
        random.seed(seed)

    coloring = {}
    uncolored_vertices = set(range(graph.num_vertices))

    while uncolored_vertices:
        vertex_scores = []
        for vertex in uncolored_vertices:
            neighbor_colors = set()
            degree = 0
            for neighbor in graph.get_neighbors(vertex):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
                else:
                    degree += 1
            saturation_degree = len(neighbor_colors)
            random_factor = random.random()
            vertex_scores.append((saturation_degree, degree, random_factor, vertex))

        vertex_scores.sort(key=lambda x: (-x[0], -x[1], -x[2]))
        best_vertex = vertex_scores[0][3]

        neighbor_colors = set()
        for neighbor in graph.get_neighbors(best_vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[best_vertex] = color
        uncolored_vertices.remove(best_vertex)

    return coloring


def greedy_random_order(graph, seed=None):
    """Simple greedy coloring with random vertex ordering."""
    if seed is not None:
        random.seed(seed)

    vertices = list(range(graph.num_vertices))
    random.shuffle(vertices)

    coloring = {}
    for vertex in vertices:
        neighbor_colors = set()
        for neighbor in graph.get_neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[vertex] = color

    return coloring


def largest_degree_first(graph, seed=None):
    """Greedy coloring processing vertices by degree (highest first)."""
    if seed is not None:
        random.seed(seed)

    # Sort vertices by degree (descending), with random tie-break
    vertices = list(range(graph.num_vertices))
    vertices.sort(key=lambda v: (-graph.get_degree(v), random.random()))

    coloring = {}
    for vertex in vertices:
        neighbor_colors = set()
        for neighbor in graph.get_neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[vertex] = color

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
                if u < v:
                    edges.append((u, v))
        return edges


def is_valid_coloring(graph, coloring):
    """Check if a coloring is valid (no adjacent vertices share a color)."""
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
    """Create a sample Petersen graph."""
    g = Graph(10)
    for i in range(5):
        g.add_edge(i, (i + 1) % 5)
    for i in range(5):
        g.add_edge(5 + i, 5 + (i + 2) % 5)
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
