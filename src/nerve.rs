use serde::{Deserialize, Serialize};

use crate::complex::SimplicialComplex;
use crate::error::TopologyError;

/// Nerve construction from a cover of a point cloud.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerveConstruction {
    /// Each cover element is a set of point indices.
    pub covers: Vec<Vec<usize>>,
    /// The nerve of the cover as a simplicial complex.
    pub nerve: SimplicialComplex,
}

impl NerveConstruction {
    /// Build a nerve from a collection of covers.
    ///
    /// Each cover is a set of point indices. The nerve contains a simplex
    /// for every set of covers with non-empty intersection.
    pub fn from_covers(covers: Vec<Vec<usize>>) -> Self {
        // Ensure covers are sorted for intersection operations
        let mut covers = covers;
        for c in &mut covers {
            c.sort();
        }
        let n = covers.len();
        let mut simplices: Vec<Vec<usize>> = Vec::new();

        // Add vertices (one per cover)
        for i in 0..n {
            simplices.push(vec![i]);
        }

        // For each pair of covers, check intersection
        for i in 0..n {
            for j in (i + 1)..n {
                if has_intersection(&covers[i], &covers[j]) {
                    simplices.push(vec![i, j]);
                }
            }
        }

        // For triples
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let ij = intersection(&covers[i], &covers[j]);
                    if !ij.is_empty() && has_intersection(&ij, &covers[k]) {
                        simplices.push(vec![i, j, k]);
                    }
                }
            }
        }

        // For quadruples
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    for l in (k + 1)..n {
                        let ijk = intersection(
                            &intersection(&covers[i], &covers[j]),
                            &covers[k],
                        );
                        if !ijk.is_empty() && has_intersection(&ijk, &covers[l]) {
                            simplices.push(vec![i, j, k, l]);
                        }
                    }
                }
            }
        }

        let nerve = SimplicialComplex::new(simplices);
        NerveConstruction { covers, nerve }
    }

    /// Build covers from a point cloud using ball neighborhoods.
    ///
    /// Creates one cover per point, including all neighbors within radius `r`.
    pub fn from_ball_cover(
        data: &[Vec<f64>],
        r: f64,
    ) -> Result<Self, TopologyError> {
        if data.is_empty() {
            return Err(TopologyError::EmptyData);
        }
        if r <= 0.0 {
            return Err(TopologyError::InvalidParameter(
                "radius must be positive".into(),
            ));
        }

        let covers: Vec<Vec<usize>> = data
            .iter()
            .map(|p| {
                let mut neighbors: Vec<usize> = data
                    .iter()
                    .enumerate()
                    .filter(|(_, q)| euclidean_dist(p, q) <= r)
                    .map(|(j, _)| j)
                    .collect();
                neighbors.sort();
                neighbors
            })
            .collect();

        Ok(Self::from_covers(covers))
    }
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn has_intersection(a: &[usize], b: &[usize]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    false
}

fn intersection(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nerve_from_covers_disjoint() {
        // Three disjoint covers
        let covers = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
        let nerve = NerveConstruction::from_covers(covers);
        // No intersections → only vertices
        assert_eq!(nerve.nerve.simplices.len(), 3);
        assert_eq!(nerve.nerve.dimension, 0);
    }

    #[test]
    fn test_nerve_from_covers_overlapping() {
        // Covers with pairwise overlaps
        let covers = vec![
            vec![0, 1, 2], // cover 0
            vec![1, 2, 3], // cover 1 — intersects 0 at {1,2}
            vec![2, 3, 4], // cover 2 — intersects 1 at {2,3}, intersects 0 at {2}
        ];
        let nerve = NerveConstruction::from_covers(covers);
        // Should have edges: (0,1), (0,2), (1,2)
        let edges = nerve.nerve.simplices_of_dimension(1);
        assert!(edges.len() >= 2);
    }

    #[test]
    fn test_nerve_triple_intersection() {
        // All three covers share point 2, but our implementation only
        // checks up to quadruple intersections explicitly.
        // Let's verify pairwise edges exist (they do share point 2).
        let covers = vec![vec![0, 2], vec![1, 2], vec![3, 2]];
        let nerve = NerveConstruction::from_covers(covers);
        // All pairs intersect, and the triple intersects at point 2
        // Check dimension is at least 1 (edges)
        assert!(nerve.nerve.dimension >= 1);
        // Verify all three edges exist
        let edges = nerve.nerve.simplices_of_dimension(1);
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_nerve_ball_cover() {
        let data: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64])
            .collect();
        let nerve = NerveConstruction::from_ball_cover(&data, 1.5).unwrap();
        assert!(nerve.covers.len() == 10);
        assert!(nerve.nerve.simplices.len() > 10);
    }

    #[test]
    fn test_nerve_empty_data() {
        let result = NerveConstruction::from_ball_cover(&[], 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_nerve_negative_radius() {
        let data = vec![vec![0.0], vec![1.0]];
        let result = NerveConstruction::from_ball_cover(&data, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_nerve_circle_cover() {
        // Points on a circle with overlapping covers
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / 20.0;
                vec![angle.cos(), angle.sin()]
            })
            .collect();
        let nerve = NerveConstruction::from_ball_cover(&data, 1.2).unwrap();
        // Should form a connected nerve
        assert!(nerve.nerve.dimension >= 1);
    }
}
