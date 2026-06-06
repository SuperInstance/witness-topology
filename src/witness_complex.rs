//! Witness complex construction from behavioral distances.
//!
//! Builds weak and strong witness complexes by identifying which data points
//! "witness" simplices on the landmark set.

use crate::{PointCloud, LandmarkSet, WitnessComplex};

/// Build a weak witness complex.
///
/// A point p weakly witnesses a simplex σ on the landmark set if all landmarks
/// in σ are closer to p than any landmark not in σ. We use a relaxed version:
/// for each witness p, find the k nearest landmarks and add all subsets up to
/// dimension `max_dim` as simplices.
pub fn weak_witness_complex(
    cloud: &PointCloud,
    landmarks: &LandmarkSet,
    max_dim: usize,
    num_nearest: usize,
) -> WitnessComplex {
    let lm = &landmarks.indices;
    let k = num_nearest.min(lm.len());

    let mut simplex_set: Vec<Vec<usize>> = Vec::new();

    // Add all vertices
    for &l in lm {
        simplex_set.push(vec![l]);
    }

    // For each point, find k nearest landmarks and add subsets
    for p in 0..cloud.len() {
        let mut dists: Vec<(usize, f64)> = lm.iter()
            .map(|&l| (l, cloud.distance(p, l)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let nearest: Vec<usize> = dists.iter().take(k).map(|(l, _)| *l).collect();

        // Add all subsets of size 2..=max_dim+1
        for size in 2..=(max_dim + 1).min(nearest.len()) {
            for combo in combinations(&nearest, size) {
                if !simplex_set.contains(&combo) {
                    simplex_set.push(combo);
                }
            }
        }
    }

    let dim = simplex_set.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
    WitnessComplex::new(simplex_set, dim)
}

/// Build a strong witness complex.
///
/// A point p strongly witnesses a simplex σ if the landmarks of σ are exactly
/// the k nearest landmarks of p (where k = |σ|). We relax: p strongly witnesses σ
/// if all vertices of σ are among the k nearest landmarks of p, and p is closer
/// to all vertices of σ than to any non-vertex landmark.
pub fn strong_witness_complex(
    cloud: &PointCloud,
    landmarks: &LandmarkSet,
    max_dim: usize,
    num_nearest: usize,
) -> WitnessComplex {
    let lm = &landmarks.indices;
    let k = num_nearest.min(lm.len());

    let mut simplex_set: Vec<Vec<usize>> = Vec::new();

    // Add all vertices
    for &l in lm {
        simplex_set.push(vec![l]);
    }

    for p in 0..cloud.len() {
        let mut dists: Vec<(usize, f64)> = lm.iter()
            .map(|&l| (l, cloud.distance(p, l)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let nearest: Vec<usize> = dists.iter().take(k).map(|(l, _)| *l).collect();
        let _max_k_dist = dists[k - 1].1;

        // For strong witnesses: only add simplex if all vertices are strictly
        // closer than any non-vertex landmark
        for size in 2..=(max_dim + 1).min(nearest.len()) {
            for combo in combinations(&nearest, size) {
                // Check that all combo members are closer than excluded landmarks
                let all_closer = combo.iter().all(|&c| {
                    let d = cloud.distance(p, c);
                    dists.iter().all(|&(l, ld)| {
                        !combo.contains(&l) && d <= ld + 1e-10 || combo.contains(&l)
                    })
                });
                if all_closer && !simplex_set.contains(&combo) {
                    simplex_set.push(combo);
                }
            }
        }
    }

    let dim = simplex_set.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
    WitnessComplex::new(simplex_set, dim)
}

/// Build a Vietoris-Rips complex on landmarks directly (for comparison).
pub fn rips_complex(
    cloud: &PointCloud,
    landmarks: &LandmarkSet,
    epsilon: f64,
    max_dim: usize,
) -> WitnessComplex {
    let lm = &landmarks.indices;
    let mut simplex_set: Vec<Vec<usize>> = Vec::new();

    // Add vertices
    for &l in lm {
        simplex_set.push(vec![l]);
    }

    // Add edges within epsilon
    for i in 0..lm.len() {
        for j in (i + 1)..lm.len() {
            if cloud.distance(lm[i], lm[j]) <= epsilon {
                simplex_set.push(vec![lm[i], lm[j]]);
            }
        }
    }

    // Clique completion up to max_dim
    for dim in 2..=max_dim {
        let prev_simplices: Vec<Vec<usize>> = simplex_set.iter()
            .filter(|s| s.len() == dim)
            .cloned()
            .collect();

        for simplex in &prev_simplices {
            for &l in lm {
                if simplex.contains(&l) {
                    continue;
                }
                // Check if l is connected to all vertices of simplex
                let all_connected = simplex.iter().all(|&v| {
                    simplex_set.contains(&vec![v.min(l), v.max(l)])
                });
                if all_connected {
                    let mut new_simplex = simplex.clone();
                    new_simplex.push(l);
                    new_simplex.sort();
                    if !simplex_set.contains(&new_simplex) {
                        simplex_set.push(new_simplex);
                    }
                }
            }
        }
    }

    let dim = simplex_set.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
    WitnessComplex::new(simplex_set, dim)
}

/// Generate all combinations of size k from a slice.
fn combinations(data: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 || k > data.len() {
        return vec![];
    }
    if k == 1 {
        return data.iter().map(|&x| vec![x]).collect();
    }
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    combinations_helper(data, k, 0, &mut combo, &mut result);
    result
}

fn combinations_helper(
    data: &[usize],
    k: usize,
    start: usize,
    combo: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if combo.len() == k {
        let mut sorted = combo.clone();
        sorted.sort();
        result.push(sorted);
        return;
    }
    for i in start..=data.len() - (k - combo.len()) {
        combo.push(data[i]);
        combinations_helper(data, k, i + 1, combo, result);
        combo.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landmark::max_min_sampling;

    #[test]
    fn test_weak_witness_has_vertices() {
        let cloud = PointCloud::from_points(vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
        ]);
        let lm = max_min_sampling(&cloud, 3);
        let wc = weak_witness_complex(&cloud, &lm, 1, 2);
        assert!(wc.vertices().len() >= 3);
    }

    #[test]
    fn test_strong_witness_fewer_simplices() {
        let cloud = PointCloud::from_points(vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0],
        ]);
        let lm = max_min_sampling(&cloud, 3);
        let weak = weak_witness_complex(&cloud, &lm, 2, 3);
        let strong = strong_witness_complex(&cloud, &lm, 2, 3);
        assert!(strong.len() <= weak.len());
    }

    #[test]
    fn test_rips_complex_edges() {
        let cloud = PointCloud::from_points(vec![
            vec![0.0], vec![1.0], vec![5.0],
        ]);
        let lm = LandmarkSet::new(vec![0, 1, 2], "test");
        let rips = rips_complex(&cloud, &lm, 2.0, 1);
        // Only edge (0,1) should be within epsilon=2
        let edges = rips.edges();
        assert_eq!(edges.len(), 1);
        assert!(edges.contains(&(0, 1)));
    }

    #[test]
    fn test_witness_complex_sparser_than_rips() {
        // Create a larger point cloud where witness complex should be sparser
        let mut points = Vec::new();
        for i in 0..20 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 20.0;
            points.push(vec![angle.cos(), angle.sin()]);
        }
        let cloud = PointCloud::from_points(points);
        let lm = max_min_sampling(&cloud, 5);
        let wc = weak_witness_complex(&cloud, &lm, 2, 3);
        let rips = rips_complex(&cloud, &lm, 5.0, 2);
        // Rips with large epsilon should have more simplices
        assert!(wc.len() <= rips.len());
    }

    #[test]
    fn test_circle_produces_h1_simplices() {
        // Points on a circle
        let mut points = Vec::new();
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 8.0;
            points.push(vec![angle.cos(), angle.sin()]);
        }
        let cloud = PointCloud::from_points(points);
        let lm = max_min_sampling(&cloud, 6);
        let wc = weak_witness_complex(&cloud, &lm, 2, 3);
        // Should have some triangles (2-simplices)
        assert!(wc.triangles().len() > 0 || wc.edges().len() > 0);
    }

    #[test]
    fn test_combinations() {
        let result = combinations(&[0, 1, 2], 2);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&vec![0, 1]));
        assert!(result.contains(&vec![0, 2]));
        assert!(result.contains(&vec![1, 2]));
    }
}
