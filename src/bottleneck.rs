//! Bottleneck and Wasserstein distances between persistence diagrams.
//!
//! Implements matching-based distances for comparing topological signatures.

use crate::{PersistenceDiagram, BottleneckDistance};

/// Compute the bottleneck distance between two persistence diagrams.
///
/// The bottleneck distance is the minimum over all bijections φ between
/// diagram points of the maximum sup-norm distance ||p - φ(p)||∞.
/// Points can be matched to the diagonal (birth = death).
///
/// Uses a greedy matching for correctness on small diagrams.
pub fn bottleneck_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram) -> BottleneckDistance {
    // Filter to finite points
    let p1: Vec<(f64, f64, usize)> = d1.points.iter()
        .filter(|(_b, d, _)| d.is_finite())
        .cloned()
        .collect();
    let p2: Vec<(f64, f64, usize)> = d2.points.iter()
        .filter(|(_b, d, _)| d.is_finite())
        .cloned()
        .collect();

    if p1.is_empty() && p2.is_empty() {
        return BottleneckDistance { value: 0.0 };
    }

    // For each point, compute cost to match to each point in the other diagram
    // or to the diagonal. Use Hungarian-like greedy approach for small diagrams.

    // Collect all possible distances (for binary search approach)
    let mut candidates: Vec<f64> = Vec::new();

    for (b1, d1, dim1) in &p1 {
        for (b2, d2, dim2) in &p2 {
            if dim1 == dim2 {
                candidates.push(sup_norm(*b1, *d1, *b2, *d2));
            }
        }
        // Diagonal cost
        candidates.push(diagonal_cost(*b1, *d1));
    }
    for (b2, d2, _) in &p2 {
        candidates.push(diagonal_cost(*b2, *d2));
    }

    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Binary search for the bottleneck value
    let mut lo = 0.0_f64;
    let mut hi = candidates.last().copied().unwrap_or(0.0);

    // Check if a perfect matching exists with max cost <= threshold
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        if has_perfect_matching(&p1, &p2, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    BottleneckDistance { value: hi }
}

/// Compute the p-Wasserstein distance between two persistence diagrams.
///
/// W_p(d1, d2) = (min_φ Σ ||p - φ(p)||_∞^p)^(1/p)
///
/// Uses a greedy nearest-neighbor matching (approximate for large diagrams).
pub fn wasserstein_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram, p: f64) -> f64 {
    let p1: Vec<(f64, f64, usize)> = d1.points.iter()
        .filter(|(_b, d, _)| d.is_finite())
        .cloned()
        .collect();
    let p2: Vec<(f64, f64, usize)> = d2.points.iter()
        .filter(|(_b, d, _)| d.is_finite())
        .cloned()
        .collect();

    if p1.is_empty() && p2.is_empty() {
        return 0.0;
    }

    // Build cost matrix: p1.len() x p2.len() for matching, plus diagonal options
    // Use greedy matching: for each point in the larger diagram, match to closest available
    let n1 = p1.len();
    let n2 = p2.len();
    let _total = n1.max(n2);

    // Create extended point sets including diagonal projections
    let mut matched = vec![false; n2];
    let mut total_cost: f64 = 0.0;

    // Match each p1 point to best available p2 point or diagonal
    for (b1, d1, dim1) in &p1 {
        let mut best_cost = diagonal_cost(*b1, *d1).powf(p);
        let mut best_j = None;

        for j in 0..n2 {
            if matched[j] {
                continue;
            }
            let (b2, d2, dim2) = &p2[j];
            if dim1 == dim2 {
                let cost = sup_norm(*b1, *d1, *b2, *d2).powf(p);
                if cost < best_cost {
                    best_cost = cost;
                    best_j = Some(j);
                }
            }
        }

        if let Some(j) = best_j {
            matched[j] = true;
        }
        total_cost += best_cost;
    }

    // Unmatched p2 points go to diagonal
    for j in 0..n2 {
        if !matched[j] {
            let (b2, d2, _) = &p2[j];
            total_cost += diagonal_cost(*b2, *d2).powf(p);
        }
    }

    total_cost.powf(1.0 / p)
}

/// Check if a perfect matching exists with all edge costs <= threshold.
fn has_perfect_matching(
    p1: &[(f64, f64, usize)],
    p2: &[(f64, f64, usize)],
    threshold: f64,
) -> bool {
    // Use augmenting path (Hungarian) for bipartite matching
    let n1 = p1.len();
    let n2 = p2.len();

    // Build adjacency: which p2 points can each p1 point match to
    let adj: Vec<Vec<usize>> = (0..n1).map(|i| {
        let (b1, d1, dim1) = &p1[i];
        (0..n2).filter(|&j| {
            let (b2, d2, dim2) = &p2[j];
            dim1 == dim2 && sup_norm(*b1, *d1, *b2, *d2) <= threshold
        }).collect()
    }).collect();

    // Augmenting path matching
    let mut match_p2: Vec<Option<usize>> = vec![None; n2];

    for i in 0..n1 {
        let mut visited = vec![false; n2];
        augment(i, &adj, &mut match_p2, &mut visited);
    }

    // Count matched p1 points
    let _matched_count = match_p2.iter().filter(|m| m.is_some()).count();

    // Points not matched to p2 must have diagonal cost <= threshold
    let mut matched_p1: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for j in 0..n2 {
        if let Some(i) = match_p2[j] {
            matched_p1.insert(i);
        }
    }

    // Check unmatched p1 diagonal costs
    for i in 0..n1 {
        if !matched_p1.contains(&i) {
            let (b, d, _) = &p1[i];
            if diagonal_cost(*b, *d) > threshold {
                return false;
            }
        }
    }

    // Check unmatched p2 diagonal costs
    for j in 0..n2 {
        if match_p2[j].is_none() {
            let (b, d, _) = &p2[j];
            if diagonal_cost(*b, *d) > threshold {
                return false;
            }
        }
    }

    true
}

fn augment(
    u: usize,
    adj: &[Vec<usize>],
    match_p2: &mut Vec<Option<usize>>,
    visited: &mut Vec<bool>,
) -> bool {
    for &v in &adj[u] {
        if visited[v] {
            continue;
        }
        visited[v] = true;
        if match_p2[v].is_none() || augment(match_p2[v].unwrap(), adj, match_p2, visited) {
            match_p2[v] = Some(u);
            return true;
        }
    }
    false
}

/// Sup-norm distance between two persistence points.
fn sup_norm(b1: f64, d1: f64, b2: f64, d2: f64) -> f64 {
    (b1 - b2).abs().max((d1 - d2).abs())
}

/// Cost of matching a point to the diagonal.
fn diagonal_cost(birth: f64, death: f64) -> f64 {
    (death - birth).abs() / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bottleneck_identical_diagrams() {
        let d1 = PersistenceDiagram::new(vec![(0.0, 1.0, 0), (0.5, 2.0, 1)], 1);
        let d2 = PersistenceDiagram::new(vec![(0.0, 1.0, 0), (0.5, 2.0, 1)], 1);
        let bn = bottleneck_distance(&d1, &d2);
        assert!(bn.value < 1e-10, "identical diagrams should have bottleneck distance 0, got {}", bn.value);
    }

    #[test]
    fn test_bottleneck_empty_diagrams() {
        let d1 = PersistenceDiagram::new(vec![], 0);
        let d2 = PersistenceDiagram::new(vec![], 0);
        let bn = bottleneck_distance(&d1, &d2);
        assert_eq!(bn.value, 0.0);
    }

    #[test]
    fn test_bottleneck_triangle_inequality() {
        let d1 = PersistenceDiagram::new(vec![(0.0, 1.0, 0), (0.2, 0.8, 0)], 0);
        let d2 = PersistenceDiagram::new(vec![(0.1, 1.1, 0), (0.3, 0.9, 0)], 0);
        let d3 = PersistenceDiagram::new(vec![(0.2, 1.2, 0), (0.4, 1.0, 0)], 0);

        let d12 = bottleneck_distance(&d1, &d2).value;
        let d23 = bottleneck_distance(&d2, &d3).value;
        let d13 = bottleneck_distance(&d1, &d3).value;

        assert!(d13 <= d12 + d23 + 1e-10,
            "triangle inequality violated: {} > {} + {} = {}", d13, d12, d23, d12 + d23);
    }

    #[test]
    fn test_bottleneck_symmetry() {
        let d1 = PersistenceDiagram::new(vec![(0.0, 2.0, 0)], 0);
        let d2 = PersistenceDiagram::new(vec![(1.0, 3.0, 0)], 0);
        let d12 = bottleneck_distance(&d1, &d2).value;
        let d21 = bottleneck_distance(&d2, &d1).value;
        assert!((d12 - d21).abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_identical() {
        let d1 = PersistenceDiagram::new(vec![(0.0, 1.0, 0)], 0);
        let d2 = PersistenceDiagram::new(vec![(0.0, 1.0, 0)], 0);
        let w = wasserstein_distance(&d1, &d2, 2.0);
        assert!(w < 1e-10);
    }

    #[test]
    fn test_wasserstein_positive_for_different() {
        let d1 = PersistenceDiagram::new(vec![(0.0, 1.0, 0)], 0);
        let d2 = PersistenceDiagram::new(vec![(0.5, 1.5, 0)], 0);
        let w = wasserstein_distance(&d1, &d2, 2.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_diagonal_cost() {
        assert_eq!(diagonal_cost(0.0, 1.0), 0.5);
        assert_eq!(diagonal_cost(0.0, 2.0), 1.0);
    }

    #[test]
    fn test_sup_norm() {
        assert_eq!(sup_norm(0.0, 1.0, 0.0, 1.0), 0.0);
        assert_eq!(sup_norm(0.0, 1.0, 1.0, 1.0), 1.0);
    }
}
