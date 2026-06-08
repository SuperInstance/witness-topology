use serde::{Deserialize, Serialize};

/// A simplicial complex: a collection of simplices closed under taking faces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplicialComplex {
    pub simplices: Vec<Vec<usize>>,
    pub dimension: usize,
    pub n_vertices: usize,
}

impl SimplicialComplex {
    /// Build a simplicial complex from a list of simplices.
    ///
    /// Automatically adds missing faces and computes dimension and vertex count.
    pub fn new(simplices: Vec<Vec<usize>>) -> Self {
        // Ensure closure under faces
        let mut all: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();

        for s in &simplices {
            add_all_faces(s, &mut all);
        }

        let simplices: Vec<Vec<usize>> = all.into_iter().collect();
        let dimension = simplices.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
        let n_vertices = simplices
            .iter()
            .flat_map(|s| s.iter().copied())
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        Self {
            simplices,
            dimension,
            n_vertices,
        }
    }

    /// Get simplices of a specific dimension.
    /// Dimension d means simplices with d+1 vertices.
    pub fn simplices_of_dimension(&self, d: usize) -> Vec<&Vec<usize>> {
        self.simplices
            .iter()
            .filter(|s| s.len() == d + 1)
            .collect()
    }

    /// Compute the boundary matrix for dimension d.
    ///
    /// Returns a matrix where entry (i,j) is 1 if the i-th (d-1)-simplex is a face
    /// of the j-th d-simplex (mod 2).
    pub fn boundary_matrix(&self, d: usize) -> Vec<Vec<u8>> {
        let faces = if d == 0 {
            vec![]
        } else {
            self.simplices_of_dimension(d - 1)
                .into_iter()
                .cloned()
                .collect::<Vec<_>>()
        };
        let d_simplices: Vec<Vec<usize>> = self
            .simplices_of_dimension(d)
            .into_iter()
            .cloned()
            .collect();

        if d == 0 || faces.is_empty() || d_simplices.is_empty() {
            return vec![];
        }

        let mut matrix = vec![vec![0u8; d_simplices.len()]; faces.len()];

        for (j, simplex) in d_simplices.iter().enumerate() {
            // Each face of simplex is obtained by removing one vertex
            for skip in 0..simplex.len() {
                let mut face: Vec<usize> = simplex.clone();
                face.remove(skip);
                face.sort();
                // Find this face in our face list
                if let Some(i) = faces.iter().position(|f| *f == face) {
                    matrix[i][j] ^= 1; // mod 2
                }
            }
        }

        matrix
    }

    /// Compute the Euler characteristic: χ = Σ(-1)^d * n_d
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi: i64 = 0;
        let mut d = 0;
        loop {
            let count = self.simplices_of_dimension(d).len();
            if count == 0 && d > self.dimension {
                break;
            }
            if d % 2 == 0 {
                chi += count as i64;
            } else {
                chi -= count as i64;
            }
            d += 1;
        }
        chi
    }
}

/// Add all faces of a simplex recursively.
fn add_all_faces(simplex: &[usize], set: &mut std::collections::BTreeSet<Vec<usize>>) {
    let mut sorted = simplex.to_vec();
    sorted.sort();
    set.insert(sorted.clone());

    if sorted.len() <= 1 {
        return;
    }

    // Add all subsets obtained by removing one element
    for i in 0..sorted.len() {
        let mut face: Vec<usize> = sorted.clone();
        face.remove(i);
        add_all_faces(&face, set);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_complex() {
        // A single triangle: 3 vertices, 3 edges, 1 face
        let sc = SimplicialComplex::new(vec![vec![0, 1, 2]]);
        assert_eq!(sc.dimension, 2);
        assert_eq!(sc.n_vertices, 3);
        assert_eq!(sc.simplices.len(), 7); // 3 vertices + 3 edges + 1 face
    }

    #[test]
    fn test_edge_complex() {
        let sc = SimplicialComplex::new(vec![vec![0, 1]]);
        assert_eq!(sc.dimension, 1);
        assert_eq!(sc.simplices.len(), 3); // 2 vertices + 1 edge
    }

    #[test]
    fn test_single_vertex() {
        let sc = SimplicialComplex::new(vec![vec![0]]);
        assert_eq!(sc.dimension, 0);
        assert_eq!(sc.n_vertices, 1);
        assert_eq!(sc.simplices.len(), 1);
    }

    #[test]
    fn test_empty_complex() {
        let sc = SimplicialComplex::new(vec![]);
        assert_eq!(sc.dimension, 0);
    }

    #[test]
    fn test_euler_characteristic_triangle() {
        // Triangle: χ = V - E + F = 3 - 3 + 1 = 1
        let sc = SimplicialComplex::new(vec![vec![0, 1, 2]]);
        assert_eq!(sc.euler_characteristic(), 1);
    }

    #[test]
    fn test_euler_characteristic_two_triangles() {
        // Two triangles sharing an edge (square divided diagonally)
        let sc = SimplicialComplex::new(vec![vec![0, 1, 2], vec![1, 2, 3]]);
        // V=4, E=5, F=2 → χ = 4-5+2 = 1
        assert_eq!(sc.euler_characteristic(), 1);
    }

    #[test]
    fn test_euler_characteristic_hollow_triangle() {
        // Just edges, no face: χ = V - E = 3 - 3 = 0 (a cycle, Betti1=1)
        let sc = SimplicialComplex::new(vec![vec![0, 1], vec![1, 2], vec![0, 2]]);
        assert_eq!(sc.euler_characteristic(), 0);
    }

    #[test]
    fn test_boundary_matrix_d1() {
        let sc = SimplicialComplex::new(vec![vec![0, 1]]);
        let bm = sc.boundary_matrix(1);
        // 2 faces (vertices), 1 edge
        assert_eq!(bm.len(), 2);
        assert_eq!(bm[0].len(), 1);
    }

    #[test]
    fn test_simplices_of_dimension() {
        let sc = SimplicialComplex::new(vec![vec![0, 1, 2]]);
        assert_eq!(sc.simplices_of_dimension(0).len(), 3);
        assert_eq!(sc.simplices_of_dimension(1).len(), 3);
        assert_eq!(sc.simplices_of_dimension(2).len(), 1);
    }

    #[test]
    fn test_tetrahedron() {
        let sc = SimplicialComplex::new(vec![vec![0, 1, 2, 3]]);
        // V=4, E=6, F=4, T=1 → total = 15
        assert_eq!(sc.simplices.len(), 15);
        assert_eq!(sc.dimension, 3);
        // χ = 4 - 6 + 4 - 1 = 1
        assert_eq!(sc.euler_characteristic(), 1);
    }
}
