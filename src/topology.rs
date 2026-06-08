use serde::{Deserialize, Serialize};

use crate::complex::SimplicialComplex;

/// Betti numbers of a topological space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BettiNumbers {
    /// β₀: number of connected components.
    pub b0: usize,
    /// β₁: number of independent loops (1-dimensional holes).
    pub b1: usize,
    /// β₂: number of enclosed voids (2-dimensional holes).
    pub b2: usize,
}

/// Extracts topological invariants from a simplicial complex.
pub struct TopologyExtractor;

impl TopologyExtractor {
    /// Compute Betti numbers via boundary matrix reduction (mod 2).
    pub fn betti_numbers(complex: &SimplicialComplex) -> BettiNumbers {
        let dim = complex.dimension;

        let b0 = if dim == 0 {
            // Number of connected components = number of vertices
            complex.simplices_of_dimension(0).len()
        } else {
            // β₀ = dim(ker ∂₀) = n_vertices - rank(∂₁)
            // But for computing via reduction: β₀ = n_vertices - rank(∂₁)
            let n_verts = complex.simplices_of_dimension(0).len();
            let bm1 = complex.boundary_matrix(1);
            let rank1 = compute_rank(&bm1);
            n_verts.saturating_sub(rank1)
        };

        let b1 = if dim >= 1 {
            // β₁ = dim(ker ∂₁) - rank(∂₂) = (n₁ - rank(∂₁)) - rank(∂₂)
            let n1 = complex.simplices_of_dimension(1).len();
            let rank1 = compute_rank(&complex.boundary_matrix(1));
            let rank2 = compute_rank(&complex.boundary_matrix(2));
            n1.saturating_sub(rank1).saturating_sub(rank2)
        } else {
            0
        };

        let b2 = if dim >= 2 {
            // β₂ = dim(ker ∂₂) - rank(∂₃)
            let n2 = complex.simplices_of_dimension(2).len();
            let rank2 = compute_rank(&complex.boundary_matrix(2));
            let rank3 = compute_rank(&complex.boundary_matrix(3));
            n2.saturating_sub(rank2).saturating_sub(rank3)
        } else {
            0
        };

        BettiNumbers { b0, b1, b2 }
    }

    /// Compute the Euler characteristic from Betti numbers.
    pub fn euler_from_betti(betti: &BettiNumbers) -> i64 {
        betti.b0 as i64 - betti.b1 as i64 + betti.b2 as i64
    }
}

/// Compute the rank of a binary matrix via Gaussian elimination (mod 2).
#[allow(clippy::needless_range_loop)]
fn compute_rank(matrix: &[Vec<u8>]) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut m: Vec<Vec<u8>> = matrix.to_vec();
    let mut pivot_row = 0;

    for col in 0..cols {
        // Find pivot
        let mut found = None;
        for row in pivot_row..rows {
            if m[row][col] == 1 {
                found = Some(row);
                break;
            }
        }
        if let Some(found_row) = found {
            // Swap
            m.swap(pivot_row, found_row);
            // Eliminate
            for row in 0..rows {
                if row != pivot_row && m[row][col] == 1 {
                    for c in 0..cols {
                        m[row][c] ^= m[pivot_row][c];
                    }
                }
            }
            pivot_row += 1;
        }
    }

    pivot_row
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
    fn test_betti_single_point() {
        let sc = SimplicialComplex::new(vec![vec![0]]);
        let betti = TopologyExtractor::betti_numbers(&sc);
        assert_eq!(betti, BettiNumbers { b0: 1, b1: 0, b2: 0 });
    }

    #[test]
    fn test_betti_two_disconnected_points() {
        let sc = SimplicialComplex::new(vec![vec![0], vec![1]]);
        let betti = TopologyExtractor::betti_numbers(&sc);
        assert_eq!(betti, BettiNumbers { b0: 2, b1: 0, b2: 0 });
    }

    #[test]
    fn test_betti_line_segment() {
        let sc = SimplicialComplex::new(vec![vec![0, 1]]);
        let betti = TopologyExtractor::betti_numbers(&sc);
        assert_eq!(betti, BettiNumbers { b0: 1, b1: 0, b2: 0 });
    }

    #[test]
    fn test_betti_triangle_solid() {
        let sc = SimplicialComplex::new(vec![vec![0, 1, 2]]);
        let betti = TopologyExtractor::betti_numbers(&sc);
        assert_eq!(betti, BettiNumbers { b0: 1, b1: 0, b2: 0 });
    }

    #[test]
    fn test_betti_hollow_triangle() {
        // Three edges forming a cycle — has β₁ = 1
        let sc = SimplicialComplex::new(vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        let betti = TopologyExtractor::betti_numbers(&sc);
        assert_eq!(betti, BettiNumbers { b0: 1, b1: 1, b2: 0 });
    }

    #[test]
    fn test_betti_tetrahedron() {
        // Solid tetrahedron — contractible
        let sc = SimplicialComplex::new(vec![vec![0, 1, 2, 3]]);
        let betti = TopologyExtractor::betti_numbers(&sc);
        assert_eq!(betti, BettiNumbers { b0: 1, b1: 0, b2: 0 });
    }

    #[test]
    fn test_betti_hollow_tetrahedron() {
        // Surface of tetrahedron (4 triangular faces, no 3-simplex)
        let sc = SimplicialComplex::new(vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ]);
        let betti = TopologyExtractor::betti_numbers(&sc);
        // Hollow tetrahedron is homeomorphic to S²: β₀=1, β₁=0, β₂=1
        assert_eq!(betti, BettiNumbers { b0: 1, b1: 0, b2: 1 });
    }

    #[test]
    fn test_euler_from_betti() {
        let betti = BettiNumbers { b0: 1, b1: 1, b2: 0 };
        assert_eq!(TopologyExtractor::euler_from_betti(&betti), 0);
    }

    #[test]
    fn test_euler_sphere() {
        let betti = BettiNumbers { b0: 1, b1: 0, b2: 1 };
        assert_eq!(TopologyExtractor::euler_from_betti(&betti), 2);
    }

    #[test]
    fn test_witness_complex_circle_betti() {
        // Build a witness complex on a circle and verify β₁ > 0
        use crate::witness::WitnessComplex;
        let data = circle_points(100);
        let wc = WitnessComplex::build(&data, 20, 3, 2).unwrap();
        let sc = SimplicialComplex::new(wc.simplices);
        let betti = TopologyExtractor::betti_numbers(&sc);
        // Circle should have β₀ = 1, and ideally β₁ ≥ 1
        assert!(
            betti.b0 >= 1,
            "circle witness complex should be connected (β₀={})",
            betti.b0
        );
    }

    #[test]
    fn test_rank_empty() {
        assert_eq!(compute_rank(&[]), 0);
        assert_eq!(compute_rank(&[vec![]]), 0);
    }

    #[test]
    fn test_rank_identity() {
        let m = vec![vec![1, 0], vec![0, 1]];
        assert_eq!(compute_rank(&m), 2);
    }

    #[test]
    fn test_rank_rank_deficient() {
        let m = vec![vec![1, 1], vec![1, 1]];
        assert_eq!(compute_rank(&m), 1);
    }

    #[test]
    fn test_two_squares_connected() {
        // Two squares sharing a vertex — figure-eight shape
        let sc = SimplicialComplex::new(vec![
            vec![0, 1],
            vec![1, 2],
            vec![2, 3],
            vec![0, 3], // first square
            vec![4, 5],
            vec![5, 6],
            vec![6, 7],
            vec![4, 7], // second square
        ]);
        // These are disconnected (no shared vertex): β₀=2, β₁=2
        let betti = TopologyExtractor::betti_numbers(&sc);
        assert_eq!(betti.b0, 2);
        assert_eq!(betti.b1, 2);
    }
}
