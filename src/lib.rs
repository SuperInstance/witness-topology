//! # witness-topology
//!
//! Topological data analysis (TDA) for agent behavior verification using witness complexes.
//!
//! This library constructs sparse topological skeletons from landmark agents and uses
//! persistent homology to detect fleet behavioral regimes.

// Pre-existing numeric code triggers several clippy pedantic lints.
#![allow(clippy::needless_range_loop, clippy::ptr_arg, clippy::let_and_return)]

pub mod bottleneck;
pub mod landmark;
pub mod mapper;
pub mod persistence;
pub mod stability;
pub mod witness_complex;

use serde::{Deserialize, Serialize};

/// A point cloud with optional labels for each point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud {
    pub points: Vec<Vec<f64>>,
    pub labels: Vec<String>,
}

impl PointCloud {
    /// Create a new point cloud from points and labels.
    pub fn new(points: Vec<Vec<f64>>, labels: Vec<String>) -> Self {
        assert_eq!(
            points.len(),
            labels.len(),
            "points and labels must have same length"
        );
        Self { points, labels }
    }

    /// Create a point cloud with auto-generated labels.
    pub fn from_points(points: Vec<Vec<f64>>) -> Self {
        let labels = (0..points.len()).map(|i| format!("p{}", i)).collect();
        Self { points, labels }
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Dimensionality of points (0 if empty).
    pub fn dimension(&self) -> usize {
        self.points.first().map_or(0, |p| p.len())
    }

    /// Euclidean distance between two points by index.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let a = &self.points[i];
        let b = &self.points[j];
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Full distance matrix.
    pub fn distance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.len();
        let mut dm = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.distance(i, j);
                dm[i][j] = d;
                dm[j][i] = d;
            }
        }
        dm
    }
}

/// A set of landmark indices with metadata about selection method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkSet {
    pub indices: Vec<usize>,
    pub selection_method: String,
}

impl LandmarkSet {
    pub fn new(indices: Vec<usize>, method: impl Into<String>) -> Self {
        Self {
            indices,
            selection_method: method.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// A witness complex: simplices indexed by landmark indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessComplex {
    pub simplices: Vec<Vec<usize>>,
    pub dimension: usize,
}

impl WitnessComplex {
    pub fn new(simplices: Vec<Vec<usize>>, dimension: usize) -> Self {
        Self {
            simplices,
            dimension,
        }
    }

    /// Number of simplices.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Get simplices of a specific dimension.
    pub fn simplices_of_dimension(&self, d: usize) -> Vec<&Vec<usize>> {
        self.simplices.iter().filter(|s| s.len() == d + 1).collect()
    }

    /// Return all vertices (0-simplices).
    pub fn vertices(&self) -> Vec<usize> {
        let mut verts: Vec<usize> = self
            .simplices
            .iter()
            .filter(|s| s.len() == 1)
            .map(|s| s[0])
            .collect();
        verts.sort();
        verts.dedup();
        verts
    }

    /// Return all edges (1-simplices).
    pub fn edges(&self) -> Vec<(usize, usize)> {
        self.simplices
            .iter()
            .filter(|s| s.len() == 2)
            .map(|s| (s[0].min(s[1]), s[0].max(s[1])))
            .collect()
    }

    /// Return all triangles (2-simplices).
    pub fn triangles(&self) -> Vec<(usize, usize, usize)> {
        self.simplices
            .iter()
            .filter(|s| s.len() == 3)
            .map(|s| {
                let mut v = s.clone();
                v.sort();
                (v[0], v[1], v[2])
            })
            .collect()
    }
}

/// A persistence diagram: collection of (birth, death, dimension) points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub points: Vec<(f64, f64, usize)>,
    pub dimension: usize,
}

impl PersistenceDiagram {
    pub fn new(points: Vec<(f64, f64, usize)>, dimension: usize) -> Self {
        Self { points, dimension }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Filter points by homology dimension.
    pub fn filter_dim(&self, dim: usize) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .filter(|(_, _, d)| *d == dim)
            .map(|(b, d, _)| (*b, *d))
            .collect()
    }

    /// Maximum persistence value.
    pub fn max_persistence(&self) -> f64 {
        self.points
            .iter()
            .map(|(b, d, _)| d - b)
            .fold(0.0_f64, f64::max)
    }
}

/// Bottleneck distance result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckDistance {
    pub value: f64,
}

/// Mapper graph for high-dimensional behavioral summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperGraph {
    pub nodes: Vec<Vec<usize>>,
    pub edges: Vec<(usize, usize)>,
}

impl MapperGraph {
    pub fn new(nodes: Vec<Vec<usize>>, edges: Vec<(usize, usize)>) -> Self {
        Self { nodes, edges }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of connected components via union-find.
    pub fn connected_components(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let n = self.nodes.len();
        let mut parent: Vec<usize> = (0..n).collect();
        let find = |parent: &mut Vec<usize>, x: usize| -> usize {
            let mut root = x;
            while parent[root] != root {
                root = parent[root];
            }
            let mut cur = x;
            while parent[cur] != root {
                let next = parent[cur];
                parent[cur] = root;
                cur = next;
            }
            root
        };
        for (u, v) in &self.edges {
            let ru = find(&mut parent, *u);
            let rv = find(&mut parent, *v);
            if ru != rv {
                parent[ru] = rv;
            }
        }
        let mut roots = (0..n).map(|i| find(&mut parent, i)).collect::<Vec<_>>();
        roots.sort();
        roots.dedup();
        roots.len()
    }
}
