//! # witness-topology
//!
//! Witness complex construction for agent shape reconstruction from samples.
//!
//! Given a sparse set of landmark points from agent state space, the witness
//! complex reconstructs the topology by finding which landmarks are "witnessed"
//! by nearby data points. This lets us reconstruct the shape of agent behavior
//! from samples.
//!
//! # Modules
//!
//! - [`landmark`] — Landmark selection strategies (random, MaxMin, density-weighted)
//! - [`witness`] — Witness complex construction
//! - [`complex`] — Simplicial complex with boundary operators
//! - [`topology`] — Topological invariants (Betti numbers)
//! - [`nerve`] — Nerve construction from covers
//!
//! # Example
//!
//! ```
//! use witness_topology::landmark::{LandmarkSelector, SelectionMethod};
//! use witness_topology::witness::WitnessComplex;
//! use witness_topology::complex::SimplicialComplex;
//! use witness_topology::topology::TopologyExtractor;
//!
//! // Sample points on a circle
//! let data: Vec<Vec<f64>> = (0..100)
//!     .map(|i| {
//!         let angle = 2.0 * std::f64::consts::PI * i as f64 / 100.0;
//!         vec![angle.cos(), angle.sin()]
//!     })
//!     .collect();
//!
//! // Build witness complex
//! let wc = WitnessComplex::build(&data, 20, 3, 2).unwrap();
//!
//! // Extract topology
//! let sc = SimplicialComplex::new(wc.simplices);
//! let betti = TopologyExtractor::betti_numbers(&sc);
//! println!("Betti numbers: β₀={}, β₁={}, β₂={}", betti.b0, betti.b1, betti.b2);
//! ```

pub mod complex;
pub mod error;
pub mod landmark;
pub mod nerve;
pub mod topology;
pub mod witness;

pub use complex::SimplicialComplex;
pub use error::TopologyError;
pub use landmark::{LandmarkSelector, SelectionMethod};
pub use nerve::NerveConstruction;
pub use topology::{BettiNumbers, TopologyExtractor};
pub use witness::WitnessComplex;
