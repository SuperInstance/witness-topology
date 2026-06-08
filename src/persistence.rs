//! Persistent homology computation via boundary matrix reduction.
//!
//! Implements the standard reduction algorithm (column reduction) to compute
//! persistence pairs from a filtered simplicial complex.

use crate::{PersistenceDiagram, WitnessComplex};

/// Compute persistent homology from a witness complex by building a filtration
/// based on simplices sorted by size (number of vertices), then by lexicographic order.
///
/// Returns a persistence diagram with (birth, death, dimension) triples.
/// Points that never die have death = f64::INFINITY.
pub fn compute_persistence(complex: &WitnessComplex) -> PersistenceDiagram {
    let simplices = &complex.simplices;
    if simplices.is_empty() {
        return PersistenceDiagram::new(vec![], complex.dimension);
    }

    // Build filtration: sort by dimension (fewer vertices first), then lexicographically
    let mut indexed: Vec<(Vec<usize>, usize)> = simplices
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    indexed.sort_by(|a, b| match a.0.len().cmp(&b.0.len()) {
        std::cmp::Ordering::Equal => a.0.cmp(&b.0),
        other => other,
    });

    let n = indexed.len();

    // Map from simplex to filtration index
    let mut simplex_to_idx: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    for (filt_idx, (simplex, _)) in indexed.iter().enumerate() {
        simplex_to_idx.insert(simplex.clone(), filt_idx);
    }

    // Build boundary matrix
    let mut boundary: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (filt_idx, (simplex, _)) in indexed.iter().enumerate() {
        if simplex.len() <= 1 {
            boundary[filt_idx] = Vec::new();
        } else {
            // Boundary: all faces obtained by removing one vertex
            let mut faces = Vec::new();
            for i in 0..simplex.len() {
                let mut face: Vec<usize> = simplex.clone();
                face.remove(i);
                face.sort();
                if let Some(&face_idx) = simplex_to_idx.get(&face) {
                    faces.push(face_idx);
                }
            }
            faces.sort();
            faces.dedup();
            boundary[filt_idx] = faces;
        }
    }

    // Reduce boundary matrix (standard algorithm)
    // We use a Z/2 representation: each column is a sorted list of row indices
    let mut reduced: Vec<Vec<usize>> = boundary.clone();
    let mut pivot_col: Vec<Option<usize>> = vec![None; n];
    let mut pairs: Vec<(usize, Option<usize>)> = Vec::new();

    for j in 0..n {
        // Find the lowest nonzero entry in column j
        let mut low_j = reduced[j].last().copied();

        while let Some(low) = low_j {
            if let Some(k) = pivot_col[low] {
                // Add column k to column j (XOR in Z/2)
                reduced[j] = symmetric_difference(&reduced[j], &reduced[k]);
                low_j = reduced[j].last().copied();
            } else {
                pivot_col[low] = Some(j);
                pairs.push((low, Some(j)));
                break;
            }
        }

        if reduced[j].is_empty() && pivot_col.iter().filter_map(|&p| p).all(|p| p != j) {
            // Column is zero and was not a pivot - essential simplex
            // We'll handle this below
        }
    }

    // Collect paired indices
    let mut paired_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut paired_cols: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (row, col) in &pairs {
        paired_rows.insert(*row);
        if let Some(c) = col {
            paired_cols.insert(*c);
        }
    }

    // Build persistence diagram
    let mut diagram_points: Vec<(f64, f64, usize)> = Vec::new();

    // Paired simplices
    for (row, col_opt) in &pairs {
        if let Some(col) = col_opt {
            let _birth_dim = indexed[*row].0.len().saturating_sub(1);
            // The pair represents a homology class born at row, dying at col
            // Actually: row is the boundary face, col is the creator
            // Standard: low j = i means σ_j kills the class born at σ_i
            let birth = *row as f64;
            let death = *col as f64;
            let dim = indexed[*row].0.len().saturating_sub(1);
            diagram_points.push((birth, death, dim));
        }
    }

    // Unpaired columns = essential classes (infinite persistence)
    for j in 0..n {
        if !paired_cols.contains(&j) && !paired_rows.contains(&j) {
            // This is an essential cycle
            let dim = indexed[j].0.len().saturating_sub(1);
            diagram_points.push((j as f64, f64::INFINITY, dim));
        }
    }

    PersistenceDiagram::new(diagram_points, complex.dimension)
}

/// Compute persistence from a distance-based Vietoris-Rips filtration.
///
/// Uses landmark points and builds the filtration by increasing epsilon.
pub fn rips_persistence(
    distances: &[Vec<f64>],
    max_dim: usize,
    max_epsilon: f64,
) -> PersistenceDiagram {
    let n = distances.len();
    if n == 0 {
        return PersistenceDiagram::new(vec![], max_dim);
    }

    // Build simplices with their birth times
    let mut simplices_with_birth: Vec<(Vec<usize>, f64)> = Vec::new();

    // Vertices: birth at 0
    for i in 0..n {
        simplices_with_birth.push((vec![i], 0.0));
    }

    // Edges: birth at their distance
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distances[i][j];
            if d <= max_epsilon {
                simplices_with_birth.push((vec![i, j], d));
            }
        }
    }

    // Higher simplices via clique completion
    for dim in 2..=max_dim {
        let edges: Vec<(usize, usize, f64)> = simplices_with_birth
            .iter()
            .filter(|(s, _)| s.len() == 2)
            .map(|(s, b)| (s[0], s[1], *b))
            .collect();

        let mut higher: Vec<(Vec<usize>, f64)> = Vec::new();
        for (s1, _b1) in &simplices_with_birth {
            if s1.len() != dim {
                continue;
            }
            for &(u, v, _edge_birth) in &edges {
                if s1.contains(&u) || s1.contains(&v) {
                    continue;
                }
                // Check if u and v are both connected to all vertices in s1
                let u_connected = s1.iter().all(|&w| {
                    simplices_with_birth.iter().any(|(s, _)| {
                        s.len() == 2 && s.contains(&u.min(w)) && s.contains(&u.max(w))
                    })
                });
                let v_connected = s1.iter().all(|&w| {
                    simplices_with_birth.iter().any(|(s, _)| {
                        s.len() == 2 && s.contains(&v.min(w)) && s.contains(&v.max(w))
                    })
                });
                if u_connected && v_connected {
                    let mut simplex = s1.clone();
                    simplex.push(u);
                    simplex.push(v);
                    simplex.sort();
                    // Birth time = max of all pairwise distances
                    let birth = simplex
                        .iter()
                        .enumerate()
                        .flat_map(|(i, &a)| simplex[i + 1..].iter().map(move |&b| distances[a][b]))
                        .fold(0.0_f64, f64::max);
                    if birth <= max_epsilon {
                        higher.push((simplex, birth));
                    }
                }
            }
        }
        simplices_with_birth.extend(higher);
    }

    // Sort by birth time, then by dimension, then lexicographically
    simplices_with_birth.sort_by(|a, b| match a.1.partial_cmp(&b.1).unwrap() {
        std::cmp::Ordering::Equal => match a.0.len().cmp(&b.0.len()) {
            std::cmp::Ordering::Equal => a.0.cmp(&b.0),
            other => other,
        },
        other => other,
    });

    // Deduplicate
    simplices_with_birth.dedup_by(|a, b| a.0 == b.0);

    let num_simplices = simplices_with_birth.len();

    // Build boundary matrix and reduce
    let mut simplex_to_idx: std::collections::HashMap<Vec<usize>, usize> =
        std::collections::HashMap::new();
    for (i, (s, _)) in simplices_with_birth.iter().enumerate() {
        simplex_to_idx.insert(s.clone(), i);
    }

    let mut reduced: Vec<Vec<usize>> = Vec::with_capacity(num_simplices);
    for (simplex, _) in &simplices_with_birth {
        if simplex.len() <= 1 {
            reduced.push(Vec::new());
        } else {
            let mut faces = Vec::new();
            for i in 0..simplex.len() {
                let mut face = simplex.clone();
                face.remove(i);
                face.sort();
                if let Some(&idx) = simplex_to_idx.get(&face) {
                    faces.push(idx);
                }
            }
            faces.sort();
            faces.dedup();
            reduced.push(faces);
        }
    }

    // Reduce
    let mut pivot_col: Vec<Option<usize>> = vec![None; num_simplices];
    for j in 0..num_simplices {
        let mut low_j = reduced[j].last().copied();
        while let Some(low) = low_j {
            if let Some(k) = pivot_col[low] {
                reduced[j] = symmetric_difference(&reduced[j], &reduced[k]);
                low_j = reduced[j].last().copied();
            } else {
                pivot_col[low] = Some(j);
                break;
            }
        }
    }

    // Extract persistence pairs
    let mut diagram_points: Vec<(f64, f64, usize)> = Vec::new();
    let mut is_paired: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for j in 0..num_simplices {
        if let Some(&low) = reduced[j].last() {
            // low j = i means pair (i, j)
            let birth_time = simplices_with_birth[low].1;
            let death_time = simplices_with_birth[j].1;
            let dim = simplices_with_birth[low].0.len().saturating_sub(1);
            diagram_points.push((birth_time, death_time, dim));
            is_paired.insert(low);
            is_paired.insert(j);
        }
    }

    // Essential simplices (infinite persistence)
    for j in 0..num_simplices {
        if !is_paired.contains(&j) && simplices_with_birth[j].0.len() == 1 {
            let birth_time = simplices_with_birth[j].1;
            let dim = 0; // Essential 0-cycles
            diagram_points.push((birth_time, f64::INFINITY, dim));
        }
    }

    PersistenceDiagram::new(diagram_points, max_dim)
}

/// Compute symmetric difference of two sorted lists (XOR for Z/2).
fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            result.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            result.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    while i < a.len() {
        result.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        result.push(b[j]);
        j += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_clusters_h0_equals_2() {
        // Two well-separated clusters
        let mut points = Vec::new();
        // Cluster 1: around (0,0)
        points.push(vec![-0.1, 0.0]);
        points.push(vec![0.1, 0.0]);
        points.push(vec![0.0, 0.1]);
        // Cluster 2: around (10,0)
        points.push(vec![9.9, 0.0]);
        points.push(vec![10.1, 0.0]);
        points.push(vec![10.0, 0.1]);

        let dists: Vec<Vec<f64>> = (0..points.len())
            .map(|i| {
                (0..points.len())
                    .map(|j| {
                        points[i]
                            .iter()
                            .zip(points[j].iter())
                            .map(|(a, b): (&f64, &f64)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect()
            })
            .collect();

        let pd = rips_persistence(&dists, 2, 5.0);
        let h0 = pd.filter_dim(0);
        // Should have 2 infinite classes (2 clusters)
        let infinite_count = h0.iter().filter(|(_, d)| d.is_infinite()).count();
        assert_eq!(infinite_count, 2);
    }

    #[test]
    fn test_single_point_h0_equals_1() {
        let dists = vec![vec![0.0]];
        let pd = rips_persistence(&dists, 1, 1.0);
        let h0 = pd.filter_dim(0);
        let infinite_count = h0.iter().filter(|(_, d)| d.is_infinite()).count();
        assert_eq!(infinite_count, 1);
    }

    #[test]
    fn test_triangle_h1() {
        // Three points in a triangle
        let dists = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let pd = rips_persistence(&dists, 2, 2.0);
        // At epsilon=1, we get a triangle which creates an H1 cycle if we also have 2-simplices
        // With 3 points and all edges, Rips at epsilon >= 1 has the 2-simplex, filling the cycle
        assert!(pd.len() > 0);
    }

    #[test]
    fn test_barcode_extraction() {
        let dists = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let pd = rips_persistence(&dists, 1, 2.0);
        // Should have some finite bars
        let finite_bars: Vec<_> = pd.points.iter().filter(|(b, d, _)| d.is_finite()).collect();
        assert!(!finite_bars.is_empty());
    }

    #[test]
    fn test_symmetric_difference() {
        let a = vec![0, 2, 4];
        let b = vec![1, 2, 3];
        let result = symmetric_difference(&a, &b);
        assert_eq!(result, vec![0, 1, 3, 4]);
    }

    #[test]
    fn test_symmetric_difference_empty() {
        let a = vec![0, 1, 2];
        let b = vec![0, 1, 2];
        let result = symmetric_difference(&a, &b);
        assert!(result.is_empty());
    }

    #[test]
    fn test_persistence_from_complex() {
        use crate::WitnessComplex;
        let complex =
            WitnessComplex::new(vec![vec![0], vec![1], vec![2], vec![0, 1], vec![1, 2]], 1);
        let pd = compute_persistence(&complex);
        // Should have some persistence pairs
        assert!(pd.len() > 0);
    }

    #[test]
    fn test_full_pipeline() {
        use crate::landmark::max_min_sampling;
        use crate::witness_complex::weak_witness_complex;
        use crate::{LandmarkSet, PointCloud};

        // Circle of points
        let mut points = Vec::new();
        for i in 0..12 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 12.0;
            points.push(vec![angle.cos(), angle.sin()]);
        }
        let cloud = PointCloud::from_points(points);
        let lm = max_min_sampling(&cloud, 6);
        let wc = weak_witness_complex(&cloud, &lm, 2, 3);
        let pd = compute_persistence(&wc);
        // Should produce a non-empty diagram
        assert!(pd.len() > 0);
    }
}
