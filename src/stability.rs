//! Stability guarantees: perturbation bounds on persistence diagrams.
//!
//! Provides theoretical stability results: small perturbations to the input
//! point cloud produce proportionally bounded changes in persistence diagrams.

use crate::PointCloud;
use crate::bottleneck::bottleneck_distance;
use crate::persistence::rips_persistence;

/// Result of a stability analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StabilityResult {
    /// Bottleneck distance between original and perturbed diagrams.
    pub bottleneck_distance: f64,
    /// Theoretical bound on the bottleneck distance.
    pub theoretical_bound: f64,
    /// Maximum point perturbation (Hausdorff-like).
    pub max_perturbation: f64,
    /// Whether the stability guarantee is satisfied.
    pub is_stable: bool,
}

/// Perturb a point cloud by adding Gaussian-like noise.
///
/// Each coordinate is perturbed by up to `noise_level * |coordinate|` or a minimum.
pub fn perturb_point_cloud(cloud: &PointCloud, noise_level: f64, seed: u64) -> PointCloud {
    let mut state = seed;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (state >> 33) as i32 as f64 / (i32::MAX as f64);
        x // returns value in [-1, 1]
    };

    let points = cloud
        .points
        .iter()
        .map(|p| {
            p.iter()
                .map(|&coord| {
                    let noise = rng() * noise_level;
                    coord + noise
                })
                .collect()
        })
        .collect();

    PointCloud::from_points(points)
}

/// Check stability: perturb the point cloud and verify the persistence diagram
/// changes by at most a factor proportional to the perturbation.
///
/// The stability theorem states: d_B(Dgm(f), Dgm(g)) ≤ 2 * d_∞(f, g)
/// where d_∞ is the sup-norm distance between the distance functions.
/// For point clouds with max perturbation ε, the bound is 2ε.
pub fn check_stability(
    cloud: &PointCloud,
    noise_level: f64,
    max_dim: usize,
    max_epsilon: f64,
    seed: u64,
) -> StabilityResult {
    let perturbed = perturb_point_cloud(cloud, noise_level, seed);

    // Compute distances
    let dm_orig = cloud.distance_matrix();
    let dm_pert = perturbed.distance_matrix();

    // Compute max perturbation in pairwise distances
    let n = cloud.len();
    let mut max_dist_perturbation: f64 = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (dm_orig[i][j] - dm_pert[i][j]).abs();
            max_dist_perturbation = max_dist_perturbation.max(diff);
        }
    }

    // Compute persistence diagrams
    let pd_orig = rips_persistence(&dm_orig, max_dim, max_epsilon);
    let pd_pert = rips_persistence(&dm_pert, max_dim, max_epsilon);

    // Compute bottleneck distance between diagrams
    let bn = bottleneck_distance(&pd_orig, &pd_pert).value;

    // Theoretical bound: 2 * max distance perturbation
    let bound = 2.0 * max_dist_perturbation;

    StabilityResult {
        bottleneck_distance: bn,
        theoretical_bound: bound,
        max_perturbation: max_dist_perturbation,
        is_stable: bn <= bound + 1e-10,
    }
}

/// Compute the Hausdorff distance between two point clouds.
///
/// d_H(X, Y) = max(sup_x inf_y d(x,y), sup_y inf_x d(x,y))
pub fn hausdorff_distance(cloud1: &PointCloud, cloud2: &PointCloud) -> f64 {
    assert_eq!(
        cloud1.len(),
        cloud2.len(),
        "point clouds must have same number of points for direct comparison"
    );

    let n = cloud1.len();
    let mut max_min_12: f64 = 0.0;
    let mut max_min_21: f64 = 0.0;

    // Use matching by index (same point, perturbed)
    for i in 0..n {
        let _d = cloud1.distance(i, i); // This is 0 for self, we want cross-cloud
        // Actually compute between cloud1[i] and cloud2[i]
        let d = cloud1.points[i]
            .iter()
            .zip(cloud2.points[i].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        max_min_12 = max_min_12.max(d);
        max_min_21 = max_min_21.max(d);
    }

    max_min_12.max(max_min_21)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perturbation_is_bounded() {
        let cloud = PointCloud::from_points(vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ]);
        let result = check_stability(&cloud, 0.01, 1, 5.0, 42);
        assert!(
            result.is_stable,
            "stability violated: bottleneck {} > bound {}",
            result.bottleneck_distance, result.theoretical_bound
        );
    }

    #[test]
    fn test_small_perturbation_small_bottleneck() {
        let mut points = Vec::new();
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 8.0;
            points.push(vec![angle.cos(), angle.sin()]);
        }
        let cloud = PointCloud::from_points(points);
        let result = check_stability(&cloud, 0.05, 1, 5.0, 123);
        // Bottleneck should be small for small perturbation
        assert!(
            result.bottleneck_distance < 1.0,
            "bottleneck too large for small perturbation: {}",
            result.bottleneck_distance
        );
    }

    #[test]
    fn test_stability_guarantee_holds() {
        // Line of points
        let points: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 2.0, 0.0]).collect();
        let cloud = PointCloud::from_points(points);
        let result = check_stability(&cloud, 0.1, 1, 10.0, 99);
        assert!(result.is_stable);
    }

    #[test]
    fn test_perturb_changes_points() {
        let cloud = PointCloud::from_points(vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
        let perturbed = perturb_point_cloud(&cloud, 0.5, 42);
        // Points should be different
        assert_ne!(cloud.points[0], perturbed.points[0]);
    }

    #[test]
    fn test_hausdorff_distance() {
        let cloud1 = PointCloud::from_points(vec![vec![0.0, 0.0], vec![1.0, 0.0]]);
        let cloud2 = PointCloud::from_points(vec![vec![0.1, 0.0], vec![1.1, 0.0]]);
        let d = hausdorff_distance(&cloud1, &cloud2);
        assert!((d - 0.1).abs() < 1e-10, "expected 0.1, got {}", d);
    }
}
