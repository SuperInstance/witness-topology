# witness-topology

Topological data analysis (TDA) for agent behavior verification using **witness complexes**.

This Rust library constructs sparse topological skeletons from landmark agents and uses persistent homology to detect fleet behavioral regimes. Built from scratch — no external math dependencies.

## Overview

When monitoring a fleet of agents (or any high-dimensional behavioral data), you need to understand the *shape* of the behavior space. Are agents clustering into distinct regimes? Is there a cyclic behavioral pattern? Are there outliers?

Witness topology provides answers through:

1. **Landmark selection** — choose representative agents via max-min sampling
2. **Witness complex construction** — build a sparse approximation of the full topology
3. **Persistent homology** — compute topological invariants (connected components, loops, voids) across scales
4. **Diagram distances** — compare behavioral signatures via bottleneck/Wasserstein metrics
5. **Mapper graphs** — summarize high-dimensional data as intuitive graph structures
6. **Stability guarantees** — provable bounds on how perturbations affect topological conclusions

## Quick Start

```rust
use witness_topology::*;
use witness_topology::landmark::max_min_sampling;
use witness_topology::witness_complex::weak_witness_complex;
use witness_topology::persistence::compute_persistence;

// Create a point cloud from agent behavioral features
let cloud = PointCloud::from_points(vec![
    vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
    vec![1.0, 1.0], vec![0.5, 0.5],
]);

// Select landmarks
let landmarks = max_min_sampling(&cloud, 3);

// Build witness complex
let complex = weak_witness_complex(&cloud, &landmarks, 2, 3);

// Compute persistent homology
let diagram = compute_persistence(&complex);

// Inspect topological features
for (birth, death, dim) in &diagram.points {
    println!("H{}: born at {:.2}, dies at {:.2}", dim, birth, death);
}
```

## Modules

### `landmark` — Landmark Selection

Select a sparse set of representative points from the data:

- **`max_min_sampling`** — Farthest-point iteration for maximum diversity
- **`random_selection`** — Deterministic pseudo-random selection (seeded)
- **`greedy_spacing`** — Max-min with configurable starting point

```rust
let landmarks = max_min_sampling(&cloud, 10);
```

### `witness_complex` — Witness Complex Construction

Build simplicial complexes from landmark-witness relationships:

- **`weak_witness_complex`** — Points witness simplices of their k-nearest landmarks
- **`strong_witness_complex`** — Stricter witnessing condition (fewer simplices)
- **`rips_complex`** — Full Vietoris-Rips on landmarks (for comparison)

Witness complexes are dramatically sparser than full Rips complexes while preserving topology.

### `persistence` — Persistent Homology

Compute topological features at all scales via boundary matrix reduction:

- **`compute_persistence`** — From any witness complex (filtration by dimension)
- **`rips_persistence`** — Distance-based Vietoris-Rips filtration with epsilon threshold

The algorithm implements standard column reduction over ℤ/2 without external dependencies.

```rust
let pd = rips_persistence(&distance_matrix, 2, 5.0);
let h0 = pd.filter_dim(0); // Connected components
let h1 = pd.filter_dim(1); // Loops
```

### `bottleneck` — Diagram Distances

Compare persistence diagrams:

- **`bottleneck_distance`** — ∞-norm optimal matching (binary search + augmenting paths)
- **`wasserstein_distance`** — p-norm matching (greedy approximation)

Both satisfy the triangle inequality. Bottleneck distance between identical diagrams is exactly 0.

```rust
let d = bottleneck_distance(&pd1, &pd2);
println!("Bottleneck distance: {:.4}", d.value);
```

### `mapper` — Mapper Graph

Summarize high-dimensional data as a graph:

- **`build_mapper_graph`** — Custom filter function with configurable intervals and overlap
- **`mapper_graph_first_coord`** — Convenience filter on first coordinate
- **`mapper_graph_norm`** — Convenience filter on L2 norm

```rust
let config = MapperConfig {
    num_intervals: 10,
    overlap: 0.3,
    cluster_epsilon: 0.5,
};
let graph = build_mapper_graph(&cloud, |p| p[0], &config);
println!("Nodes: {}, Components: {}", graph.node_count(), graph.connected_components());
```

### `stability` — Perturbation Bounds

Verify that small input changes produce bounded output changes:

- **`check_stability`** — Perturb and compare persistence diagrams
- **`perturb_point_cloud`** — Add controlled noise to data
- **`hausdorff_distance`** — Measure point cloud distance

The stability theorem guarantees: d_B(Dgm(f), Dgm(g)) ≤ 2 · d_∞(f, g)

## Core Types

| Type | Description |
|------|-------------|
| `PointCloud` | Points with optional labels, distance computation |
| `LandmarkSet` | Selected landmark indices with method metadata |
| `WitnessComplex` | Simplices on landmark vertices |
| `PersistenceDiagram` | (birth, death, dimension) triples |
| `BottleneckDistance` | Wrapped distance value |
| `MapperGraph` | Nodes (point clusters) and edges (overlap) |

All public types derive `Serialize` and `Deserialize` via serde.

## Testing

38 tests covering all modules:

```bash
cargo test
```

Key test scenarios:
- Max-min landmark selection produces diverse landmarks
- Two disconnected clusters → H₀ = 2
- Circle of points → H₁ features detected
- Bottleneck distance between identical diagrams = 0
- Triangle inequality satisfied
- Stability: perturbed diagrams bounded by perturbation size
- Witness complex sparser than full Rips
- Full pipeline: cloud → landmarks → complex → persistence

## Architecture

```
src/
├── lib.rs              # Core types (PointCloud, LandmarkSet, etc.)
├── landmark.rs         # Landmark selection strategies
├── witness_complex.rs  # Weak/strong witness + Rips construction
├── persistence.rs      # Boundary matrix reduction
├── bottleneck.rs       # Bottleneck & Wasserstein distances
├── mapper.rs           # Mapper graph algorithm
└── stability.rs        # Perturbation bounds & verification
```

## Use Cases

- **Agent fleet monitoring** — Detect behavioral regime changes via topology shifts
- **Anomaly detection** — Outliers appear as topological features
- **Behavioral clustering** — Mapper graphs reveal natural groupings
- **Regime transitions** — Persistent homology detects phase changes
- **Comparison** — Bottleneck distances quantify behavioral similarity

## Performance

Witness complexes are O(n·k) where n is the number of data points and k is the number of landmarks, compared to O(n³) for full Rips complexes. This makes topological analysis feasible for large agent fleets.

## License

MIT
