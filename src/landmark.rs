use serde::{Deserialize, Serialize};

use crate::error::TopologyError;

/// Strategy for selecting landmark points from a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionMethod {
    /// Choose landmarks uniformly at random.
    Random,
    /// Iterative farthest-point sampling (MaxMin).
    MaxMin,
    /// Density-weighted random selection.
    Density,
}

/// Selects a subset of landmark points from a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandmarkSelector {
    pub landmarks: Vec<Vec<f64>>,
    pub method: SelectionMethod,
}

impl LandmarkSelector {
    /// Select `k` landmarks from `data` using the given method.
    pub fn select(
        data: &[Vec<f64>],
        k: usize,
        method: SelectionMethod,
    ) -> Result<Self, TopologyError> {
        if data.is_empty() {
            return Err(TopologyError::EmptyData);
        }
        if k == 0 {
            return Err(TopologyError::InvalidParameter("k must be > 0".into()));
        }
        if k > data.len() {
            return Err(TopologyError::InsufficientData {
                have: data.len(),
                need: k,
            });
        }

        let dim = data[0].len();
        for p in data.iter() {
            if p.len() != dim {
                return Err(TopologyError::DimensionMismatch {
                    expected: dim,
                    got: p.len(),
                });
            }
        }

        let indices = match method {
            SelectionMethod::Random => select_random(data, k),
            SelectionMethod::MaxMin => select_maxmin(data, k),
            SelectionMethod::Density => select_density(data, k),
        };

        let landmarks = indices.iter().map(|&i| data[i].clone()).collect();
        Ok(Self { landmarks, method })
    }
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn select_random(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    // Simple deterministic pseudo-random selection using linear congruential generator
    let n = data.len();
    let mut chosen = vec![false; n];
    let mut result = Vec::with_capacity(k);
    let mut seed: u64 = 42;
    while result.len() < k {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (seed >> 33) as usize % n;
        if !chosen[idx] {
            chosen[idx] = true;
            result.push(idx);
        }
    }
    result
}

fn select_maxmin(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = data.len();
    let mut result = Vec::with_capacity(k);
    // Start with the first point
    result.push(0);
    // min_dist[i] = distance from data[i] to nearest landmark so far
    let mut min_dist: Vec<f64> = (0..n).map(|i| euclidean_dist(&data[i], &data[0])).collect();

    while result.len() < k {
        // Find the point with the largest minimum distance to existing landmarks
        let (farthest, _) = min_dist
            .iter()
            .enumerate()
            .filter(|(i, _)| !result.contains(i))
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        result.push(farthest);
        // Update minimum distances
        for i in 0..n {
            let d = euclidean_dist(&data[i], &data[farthest]);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
    }
    result
}

fn select_density(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = data.len();
    // Estimate local density for each point using average distance to nearest neighbors
    let bandwidth = estimate_bandwidth(data);
    let mut density: Vec<f64> = data
        .iter()
        .map(|p| {
            let sum: f64 = data
                .iter()
                .map(|q| {
                    let d = euclidean_dist(p, q);
                    (-0.5 * (d / bandwidth).powi(2)).exp()
                })
                .sum();
            sum / n as f64
        })
        .collect();

    // Normalize densities for weighted selection
    let total: f64 = density.iter().sum();
    if total == 0.0 {
        // Fall back to uniform random
        return select_random(data, k);
    }
    for d in &mut density {
        *d /= total;
    }

    // Weighted selection without replacement
    let mut chosen = vec![false; n];
    let mut result = Vec::with_capacity(k);
    let mut rng_state: u64 = 12345;

    while result.len() < k {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let threshold = ((rng_state >> 11) as f64) / (1u64 << 53) as f64;
        let mut cumsum = 0.0;
        let mut picked = false;
        for i in 0..n {
            if chosen[i] {
                continue;
            }
            cumsum += density[i];
            if cumsum >= threshold {
                chosen[i] = true;
                result.push(i);
                density[i] = 0.0;
                let remaining_total: f64 = density.iter().sum();
                if remaining_total > 0.0 {
                    for d in &mut density {
                        *d /= remaining_total;
                    }
                }
                picked = true;
                break;
            }
        }
        if !picked {
            // Fallback: pick first unchosen
            for i in 0..n {
                if !chosen[i] {
                    chosen[i] = true;
                    result.push(i);
                    density[i] = 0.0;
                    let remaining_total: f64 = density.iter().sum();
                    if remaining_total > 0.0 {
                        for d in &mut density {
                            *d /= remaining_total;
                        }
                    }
                    break;
                }
            }
        }
    }
    result
}

fn estimate_bandwidth(data: &[Vec<f64>]) -> f64 {
    // Silverman's rule of thumb: 1.06 * std * n^(-1/5)
    let n = data.len() as f64;
    let dim = data[0].len();
    let mean: Vec<f64> = (0..dim)
        .map(|j| data.iter().map(|p| p[j]).sum::<f64>() / n)
        .collect();
    let variance: f64 = data
        .iter()
        .map(|p| {
            let d = euclidean_dist(p, &mean);
            d * d
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();
    1.06 * std * n.powf(-0.2).max(0.01)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle_points(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                vec![angle.cos(), angle.sin()]
            })
            .collect()
    }

    #[test]
    fn test_random_selection() {
        let data = circle_points(100);
        let sel = LandmarkSelector::select(&data, 10, SelectionMethod::Random).unwrap();
        assert_eq!(sel.landmarks.len(), 10);
        assert_eq!(sel.method, SelectionMethod::Random);
    }

    #[test]
    fn test_maxmin_selection() {
        let data = circle_points(100);
        let sel = LandmarkSelector::select(&data, 10, SelectionMethod::MaxMin).unwrap();
        assert_eq!(sel.landmarks.len(), 10);
        assert_eq!(sel.method, SelectionMethod::MaxMin);
    }

    #[test]
    fn test_density_selection() {
        let data = circle_points(100);
        let sel = LandmarkSelector::select(&data, 10, SelectionMethod::Density).unwrap();
        assert_eq!(sel.landmarks.len(), 10);
        assert_eq!(sel.method, SelectionMethod::Density);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let result = LandmarkSelector::select(&data, 5, SelectionMethod::Random);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_many_landmarks() {
        let data = vec![vec![0.0], vec![1.0]];
        let result = LandmarkSelector::select(&data, 5, SelectionMethod::Random);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_k() {
        let data = circle_points(10);
        let result = LandmarkSelector::select(&data, 0, SelectionMethod::Random);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let data = vec![vec![0.0, 0.0], vec![1.0], vec![2.0, 2.0]];
        let result = LandmarkSelector::select(&data, 2, SelectionMethod::Random);
        assert!(result.is_err());
    }

    #[test]
    fn test_maxmin_spreads_points() {
        // MaxMin should spread points across the circle
        let data = circle_points(50);
        let sel = LandmarkSelector::select(&data, 4, SelectionMethod::MaxMin).unwrap();
        // The landmarks should be roughly evenly spaced
        for i in 0..sel.landmarks.len() {
            for j in (i + 1)..sel.landmarks.len() {
                let d = euclidean_dist(&sel.landmarks[i], &sel.landmarks[j]);
                assert!(d > 0.5, "landmarks too close: {d}");
            }
        }
    }

    #[test]
    fn test_all_methods_produce_valid_output() {
        let data = circle_points(30);
        for method in [
            SelectionMethod::Random,
            SelectionMethod::MaxMin,
            SelectionMethod::Density,
        ] {
            let sel = LandmarkSelector::select(&data, 5, method).unwrap();
            assert_eq!(sel.landmarks.len(), 5);
            for lm in &sel.landmarks {
                assert_eq!(lm.len(), 2);
            }
        }
    }
}
