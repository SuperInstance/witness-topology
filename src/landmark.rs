//! Landmark selection strategies for witness complex construction.
//!
//! Provides max-min sampling (farthest-point) and random selection to choose
//! a sparse set of landmark points from a point cloud.

use crate::{PointCloud, LandmarkSet};

/// Select landmarks using max-min (farthest point) sampling.
///
/// Iteratively picks the point farthest from all previously selected landmarks,
/// producing a well-spread set of landmarks.
pub fn max_min_sampling(cloud: &PointCloud, num_landmarks: usize) -> LandmarkSet {
    let n = cloud.len();
    assert!(num_landmarks <= n && num_landmarks > 0, "num_landmarks must be in 1..={}", n);

    if num_landmarks == n {
        return LandmarkSet::new((0..n).collect(), "max_min");
    }

    let mut landmarks = Vec::with_capacity(num_landmarks);
    let mut min_dists: Vec<f64> = vec![f64::INFINITY; n];

    // Start with the first point
    landmarks.push(0);
    for i in 0..n {
        min_dists[i] = cloud.distance(0, i);
    }

    for _ in 1..num_landmarks {
        // Find point with maximum min-distance to existing landmarks
        let (farthest, _) = (0..n)
            .filter(|i| !landmarks.contains(i))
            .map(|i| (i, min_dists[i]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        landmarks.push(farthest);
        // Update min distances
        for i in 0..n {
            let d = cloud.distance(farthest, i);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    landmarks.sort();
    LandmarkSet::new(landmarks, "max_min")
}

/// Select landmarks uniformly at random.
///
/// Uses a simple deterministic shuffle for reproducibility (seeded by input).
pub fn random_selection(cloud: &PointCloud, num_landmarks: usize, seed: u64) -> LandmarkSet {
    let n = cloud.len();
    assert!(num_landmarks <= n && num_landmarks > 0, "num_landmarks must be in 1..={}", n);

    // Simple LCG-based shuffle
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(n as u64);
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    let mut selected: Vec<usize> = indices.into_iter().take(num_landmarks).collect();
    selected.sort();
    LandmarkSet::new(selected, "random")
}

/// Select landmarks by greedy spacing: pick the first, then iteratively pick
/// the point maximizing the minimum distance to all landmarks chosen so far.
/// This is an alias for max-min sampling with a random start.
pub fn greedy_spacing(cloud: &PointCloud, num_landmarks: usize, start: usize) -> LandmarkSet {
    let n = cloud.len();
    assert!(num_landmarks <= n && num_landmarks > 0, "num_landmarks must be in 1..={}", n);
    assert!(start < n, "start index must be < {}", n);

    let mut landmarks = Vec::with_capacity(num_landmarks);
    let mut min_dists: Vec<f64> = vec![f64::INFINITY; n];

    landmarks.push(start);
    for i in 0..n {
        min_dists[i] = cloud.distance(start, i);
    }

    for _ in 1..num_landmarks {
        let (farthest, _) = (0..n)
            .filter(|i| !landmarks.contains(i))
            .map(|i| (i, min_dists[i]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        landmarks.push(farthest);
        for i in 0..n {
            let d = cloud.distance(farthest, i);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    landmarks.sort();
    LandmarkSet::new(landmarks, "greedy_spacing")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_min_produces_correct_count() {
        let cloud = PointCloud::from_points(vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
            vec![1.0, 1.0], vec![0.5, 0.5],
        ]);
        let lm = max_min_sampling(&cloud, 3);
        assert_eq!(lm.indices.len(), 3);
        assert_eq!(lm.selection_method, "max_min");
    }

    #[test]
    fn test_max_min_produces_diverse_landmarks() {
        // Four corners of a square
        let cloud = PointCloud::from_points(vec![
            vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0], vec![10.0, 10.0],
            vec![5.0, 5.0],
        ]);
        let lm = max_min_sampling(&cloud, 4);
        // Should pick corners, not the center
        assert!(!lm.indices.contains(&4) || lm.indices.len() == 5);
    }

    #[test]
    fn test_max_min_all_points() {
        let cloud = PointCloud::from_points(vec![vec![0.0], vec![1.0]]);
        let lm = max_min_sampling(&cloud, 2);
        assert_eq!(lm.indices, vec![0, 1]);
    }

    #[test]
    fn test_random_selection_correct_count() {
        let cloud = PointCloud::from_points(vec![
            vec![0.0], vec![1.0], vec![2.0], vec![3.0],
        ]);
        let lm = random_selection(&cloud, 2, 42);
        assert_eq!(lm.indices.len(), 2);
        assert_eq!(lm.selection_method, "random");
        // Indices should be sorted
        assert_eq!(lm.indices.windows(2).all(|w| w[0] < w[1]), true);
    }

    #[test]
    fn test_greedy_spacing_with_start() {
        let cloud = PointCloud::from_points(vec![
            vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0], vec![10.0, 10.0],
        ]);
        let lm = greedy_spacing(&cloud, 2, 0);
        assert_eq!(lm.indices.len(), 2);
        assert!(lm.indices.contains(&0));
        // The farthest from 0 should be one of the corners at distance ~14.14
        // (10,10) is farthest from (0,0)
    }

    #[test]
    fn test_landmark_set_on_single_point() {
        let cloud = PointCloud::from_points(vec![vec![0.0, 0.0]]);
        let lm = max_min_sampling(&cloud, 1);
        assert_eq!(lm.indices, vec![0]);
    }
}
