# Graph Coloring Example

This example demonstrates how OpenEvolve can discover sophisticated graph coloring algorithms starting from a simple greedy implementation.

## Problem Description

**Graph Coloring** is a classic NP-hard problem:
- Given an undirected graph G = (V, E)
- Assign colors to vertices such that no two adjacent vertices share the same color
- Goal: Use the minimum number of colors (the **chromatic number** χ(G))

This problem has many real-world applications:
- **Scheduling**: Exam timetabling, job scheduling
- **Register allocation**: Compiler optimization
- **Frequency assignment**: Radio/cellular networks
- **Map coloring**: Cartography

## Getting Started

To run this example:

```bash
cd examples/graph_coloring
python ../../openevolve-run.py initial_program.py evaluator.py --config config.yaml --iterations 50
```

## Algorithm Evolution

### Initial Algorithm (Simple Greedy)

The initial implementation is a basic greedy algorithm that processes vertices in order and assigns the smallest available color:

```python
def graph_coloring(graph):
    coloring = {}
    for vertex in range(graph.num_vertices):
        neighbor_colors = set()
        for neighbor in graph.get_neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        color = 0
        while color in neighbor_colors:
            color += 1

        coloring[vertex] = color
    return coloring
```

### Evolved Algorithm

*To be updated after running OpenEvolve*

## Key Improvements

Through evolution, OpenEvolve may discover improvements such as:

1. **Vertex Ordering**: Processing high-degree vertices first (Welsh-Powell)
2. **Saturation Degree**: DSatur algorithm - prioritize vertices with most neighbor colors
3. **Independent Set Building**: RLF-style algorithms
4. **Local Search**: Color swapping to reduce total colors

## Test Graphs

The evaluator tests on multiple graph types:
- **Petersen Graph**: Classic graph, χ = 3
- **Complete Graph K5**: χ = 5
- **Bipartite Graphs**: χ = 2
- **Cycle Graphs**: χ = 2 (even) or 3 (odd)
- **Random Graphs**: Varying density

## Results

*To be updated after running OpenEvolve*

| Metric | Initial | Evolved |
|--------|---------|---------|
| Combined Score | TBD | TBD |
| Optimal Colorings | TBD | TBD |

## References

- Welsh, D.J.A. and Powell, M.B. (1967). "An upper bound for the chromatic number of a graph and its application to timetabling problems."
- Brélaz, D. (1979). "New Methods to Color the Vertices of a Graph" (DSatur algorithm)
- Leighton, F.T. (1979). "A graph coloring algorithm for large scheduling problems" (RLF algorithm)
- Hertz, A. and de Werra, D. (1987). "Using Tabu Search Techniques for Graph Coloring"

## Next Steps

Try modifying the config.yaml to:
- Increase iterations for more evolution
- Change LLM models
- Adjust the system message to guide evolution differently
