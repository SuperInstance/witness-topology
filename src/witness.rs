use serde::{Deserialize, Serialize};

use crate::error::TopologyError;
use crate::landmark::LandmarkSelector;

/// A witness complex built from landmark and witness points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessComplex {
    pub landmarks: Vec<Vec<f64>>,
    pub witnesses: Vec<Vec<f64>>,
    pub simplices: Vec<Vec<usize>>,
    pub max_dimension: usize,
}

impl WitnessComplex {
    /// Build a witness complex from data points.
    ///
    /// Selects `n_landmarks` landmarks from `data`, then for each data point
    /// acting as a witness, finds the `k` nearest landmarks and adds all
    /// subsets up to `max_dimension` as simplices.
    pub fn build(
        data: &[Vec<f64>],
        n_landmarks: usize,
        k: usize,
        max_dimension: usize,
    ) -> Result<Self, TopologyError> {
        let selector = LandmarkSelector::select(data, n_landmarks, crate::landmark::SelectionMethod::MaxMin)?;
        Self::build_with_landmarks(data, &selector.landmarks, k, max_dimension)
    }

    /// Build a witness complex using pre-selected landmarks.
    pub fn build_with_landmarks(
        data: &[Vec<f64>],
        landmarks: &[Vec<f64>],
        k: usize,
        max_dimension: usize,
    ) -> Result<Self, TopologyError> {
        if data.is_empty() {
            return Err(TopologyError::EmptyData);
        }
        if landmarks.is_empty() {
            return Err(TopologyError::InvalidParameter("no landmarks".into()));
        }
        if k == 0 {
            return Err(TopologyError::InvalidParameter("k must be > 0".into()));
        }
        let effective_k = k.min(landmarks.len());

        let mut simplex_set: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();

        // Add vertices
        for i in 0..landmarks.len() {
            simplex_set.insert(vec![i]);
        }

        // For each witness, find k nearest landmarks and add simplices
        for witness in data {
            let mut dists: Vec<(usize, f64)> = landmarks
                .iter()
                .enumerate()
                .map(|(i, lm)| (i, euclidean_dist(witness, lm)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let nearest: Vec<usize> = dists[..effective_k].iter().map(|(i, _)| *i).collect();

            // Add all subsets of size 1..=max_dimension+1
            add_subsets(&nearest, max_dimension, &mut simplex_set);
        }

        let simplices: Vec<Vec<usize>> = simplex_set.into_iter().collect();

        Ok(Self {
            landmarks: landmarks.to_vec(),
            witnesses: data.to_vec(),
            simplices,
            max_dimension,
        })
    }

    /// Build with a specific selection method.
    pub fn build_with_method(
        data: &[Vec<f64>],
        n_landmarks: usize,
        k: usize,
        max_dimension: usize,
        method: crate::landmark::SelectionMethod,
    ) -> Result<Self, TopologyError> {
        let selector = LandmarkSelector::select(data, n_landmarks, method)?;
        Self::build_with_landmarks(data, &selector.landmarks, k, max_dimension)
    }
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Add all subsets of `vertices` of size 2..=max_dim+1 to the simplex set.
fn add_subsets(
    vertices: &[usize],
    max_dim: usize,
    set: &mut std::collections::BTreeSet<Vec<usize>>,
) {
    let n = vertices.len();
    if n < 2 {
        return;
    }
    let max_size = (max_dim + 1).min(n);
    for size in 2..=max_size {
        // Generate all combinations of given size
        let mut combo: Vec<usize> = Vec::new();
        generate_combinations(vertices, size, 0, &mut combo, set);
    }
}

fn generate_combinations(
    vertices: &[usize],
    size: usize,
    start: usize,
    current: &mut Vec<usize>,
    set: &mut std::collections::BTreeSet<Vec<usize>>,
) {
    if current.len() == size {
        let mut sorted = current.clone();
        sorted.sort();
        set.insert(sorted);
        return;
    }
    let remaining = size - current.len();
    for i in start..=(vertices.len() - remaining) {
        current.push(vertices[i]);
        generate_combinations(vertices, size, i + 1, current, set);
        current.pop();
    }
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
    fn test_build_circle() {
        let data = circle_points(100);
        let wc = WitnessComplex::build(&data, 20, 3, 2).unwrap();
        assert!(!wc.simplices.is_empty());
        // Should have vertices
        assert!(wc.simplices.iter().any(|s| s.len() == 1));
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        let result = WitnessComplex::build(&data, 5, 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_with_landmarks() {
        let data = circle_points(50);
        let landmarks = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let wc = WitnessComplex::build_with_landmarks(&data, &landmarks, 2, 1).unwrap();
        assert!(wc.simplices.len() >= 4); // at least 4 vertices
    }

    #[test]
    fn test_k_larger_than_landmarks() {
        let data = circle_points(20);
        let landmarks = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        // k=5 but only 2 landmarks — should still work with k clamped
        let wc = WitnessComplex::build_with_landmarks(&data, &landmarks, 5, 1).unwrap();
        assert!(wc.simplices.len() >= 2);
    }

    #[test]
    fn test_simplices_are_sorted() {
        let data = circle_points(50);
        let wc = WitnessComplex::build(&data, 10, 3, 2).unwrap();
        for simplex in &wc.simplices {
            let mut sorted = simplex.clone();
            sorted.sort();
            assert_eq!(*simplex, sorted);
        }
    }

    #[test]
    fn test_max_dimension_respected() {
        let data = circle_points(50);
        let wc = WitnessComplex::build(&data, 10, 3, 1).unwrap();
        for simplex in &wc.simplices {
            assert!(simplex.len() <= 2);
        }
    }

    #[test]
    fn test_build_with_all_methods() {
        let data = circle_points(30);
        for method in [
            crate::landmark::SelectionMethod::Random,
            crate::landmark::SelectionMethod::MaxMin,
            crate::landmark::SelectionMethod::Density,
        ] {
            let wc = WitnessComplex::build_with_method(&data, 8, 2, 1, method).unwrap();
            assert!(!wc.simplices.is_empty());
        }
    }
}
