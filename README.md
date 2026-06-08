# witness-topology

[![crates.io](https://img.shields.io/crates/v/witness-topology.svg)](https://crates.io/crates/witness-topology)
[![docs.rs](https://docs.rs/witness-topology/badge.svg)](https://docs.rs/witness-topology)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Problem

You have a high-dimensional point cloud (agent states, sensor readings, word embeddings) and you want to know its topological shape: how many connected components? Any loops? Any voids? The shape tells you about the structure of the data — one cluster = one behavioral mode, a loop = cyclic behavior, two components = bimodal.

The Vietoris-Rips complex is the standard tool: connect all point pairs within distance ε. But for n points, this creates up to O(nᵈ) simplices. For n=10,000 and d=3, that's a trillion simplices. Not happening.

## The Idea: Witness Complexes

The **witness complex** (de Silva & Carlsson, 2004) solves this by selecting a small set of **landmark points** and letting the remaining **witness points** vote on which landmarks should be connected. A simplex {l₁, l₂, ..., lₖ} is included if some witness point w is simultaneously close to all of those landmarks.

This reduces the complex from O(nᵈ) to O(mᵈ) where m << n is the number of landmarks, while provably preserving the topology (under mild sampling conditions — landmarks must be sufficiently dense).

### The analogy

Imagine mapping a city by asking tourists "which landmarks can you see from here?" Each tourist is a witness. If many tourists near landmark A can also see landmark B, then A and B are probably close. The tourist reports reconstruct the city layout without surveying every street.

## How It Works

### Select landmarks

```rust
use witness_topology::{PointCloud, LandmarkSelector};

let cloud = PointCloud::from_vectors(&[
    vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
    vec![1.0, 1.0], vec![0.5, 0.5],
    // ... hundreds more points
]);

// MaxMin: iteratively pick the point farthest from all existing landmarks
// Guarantees good spatial coverage. O(n·m) but worth it.
let landmarks = LandmarkSelector::maxmin(&cloud, 20);
```

Three selection strategies:
- **Random**: O(m), fast but can cluster landmarks in one region
- **MaxMin** (recommended): O(n·m), guaranteed coverage — every point is close to some landmark
- **Density-weighted**: O(n·m), more landmarks in dense regions where detail matters

### Build the witness complex

```rust
use witness_topology::WitnessComplex;

let complex = WitnessComplex::build(&landmarks, &cloud, /* k_nearest */ 3, /* max_dim */ 2);
println!("{} simplices from {} landmarks + {} witnesses",
    complex.simplices.len(), landmarks.len(), cloud.len());
```

### Compute Betti numbers

```rust
use witness_topology::TopologyExtractor;

let topo = TopologyExtractor::from_complex(&complex);
println!("β₀ = {} (connected components)", topo.betti(0));
println!("β₁ = {} (loops)", topo.betti(1));
println!("β₂ = {} (voids)", topo.betti(2));
println!("Euler characteristic χ = {}", topo.euler_characteristic());
```

Verified against known shapes: single point (β₀=1), hollow triangle (β₁=1), solid tetrahedron (β₀=1), figure-eight (β₁=2).

### Nerve construction (alternative approach)

The **nerve** of a cover connects overlapping sets. By the Nerve Theorem, it recovers the correct topology:

```rust
use witness_topology::nerve::NerveConstruction;

let nerve = NerveConstruction::from_balls(&cloud, /* radius */ 0.5);
println!("β₀={}, β₁={}", nerve.topology().betti(0), nerve.topology().betti(1));
```

## When To Use This

- **Agent behavior profiling**: What shape does an agent's state trajectory have? (steady = point, oscillating = loop, chaotic = high-dimensional)
- **Data exploration**: How many natural clusters in your data? Any ring-like structures suggesting cyclic processes?
- **Dimensionality reduction validation**: After t-SNE/UMAP, does the 2D projection preserve the original topology?
- **Sensor network coverage**: Do your sensors see enough of the space to reconstruct its shape?

## Module Map

| Module | What it does |
|---|---|
| `landmark` | `LandmarkSelector` — random, maxmin, density-weighted landmark selection |
| `witness` | `WitnessComplex` — build from landmarks + witness votes |
| `complex` | `SimplicialComplex` — face closure, boundary matrices, Euler characteristic |
| `topology` | `TopologyExtractor` — Betti numbers via Gaussian elimination on boundary matrices |
| `nerve` | `NerveConstruction` — nerve of a cover, ball-cover builder |
| `error` | `WitnessError` |

## Design Decisions

- **Why witness over Vietoris-Rips?** VR is exact but O(nᵈ). Witness is approximate but O(mᵈ) with m << n. For n > 1000, witness is the only practical choice.
- **MaxMin as default**: Random landmarks can cluster, missing entire regions. MaxMin guarantees every point is close to at least one landmark. The O(n·m) cost is negligible compared to the O(mᵈ) complex construction.
- **Betti numbers via Gaussian elimination**: Not the fastest algorithm for huge complexes (PHAT or Ripser are better), but correct and simple. For m < 200 landmarks, it runs in milliseconds.

## Links

- [Documentation](https://docs.rs/witness-topology)
- [Repository](https://github.com/SuperInstance/witness-topology)
- [crates.io](https://crates.io/crates/witness-topology)
- de Silva & Carlsson (2004) — *Topological estimation using witness complexes*

## License

MIT
