//! Mapper graph construction for high-dimensional behavioral summary.
//!
//! Implements the Mapper algorithm: project data through a filter function,
//! bin the resulting values, cluster within overlapping bins, and build a graph
//! connecting overlapping clusters.

use crate::{PointCloud, MapperGraph};

/// Configuration for Mapper graph construction.
#[derive(Debug, Clone)]
pub struct MapperConfig {
    /// Number of intervals to divide the filter range into.
    pub num_intervals: usize,
    /// Overlap fraction between adjacent intervals (0.0 to <1.0).
    pub overlap: f64,
    /// Maximum distance for single-linkage clustering within bins.
    pub cluster_epsilon: f64,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            num_intervals: 10,
            overlap: 0.3,
            cluster_epsilon: 0.5,
        }
    }
}

/// Build a Mapper graph using a filter function applied to each point.
///
/// The filter function maps each point to a scalar value. Points are binned
/// by filter value into overlapping intervals, clustered within each interval,
/// and connected if clusters share points.
pub fn build_mapper_graph<F>(
    cloud: &PointCloud,
    filter_fn: F,
    config: &MapperConfig,
) -> MapperGraph
where
    F: Fn(&[f64]) -> f64,
{
    let n = cloud.len();
    if n == 0 {
        return MapperGraph::new(vec![], vec![]);
    }

    // Compute filter values
    let filter_values: Vec<f64> = cloud.points.iter().map(|p| filter_fn(p)).collect();

    let f_min = filter_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let f_max = filter_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (f_max - f_min).abs() < 1e-15 {
        // All points have same filter value
        let node = (0..n).collect();
        return MapperGraph::new(vec![node], vec![]);
    }

    // Create overlapping intervals
    let interval_width = (f_max - f_min) / (config.num_intervals as f64 * (1.0 - config.overlap) + config.overlap);
    let step = interval_width * (1.0 - config.overlap);

    let mut intervals: Vec<(f64, f64)> = Vec::new();
    let mut start = f_min;
    for _ in 0..config.num_intervals {
        let end = start + interval_width;
        intervals.push((start, end));
        start += step;
    }

    // For each interval, find points in range, cluster them
    let mut all_clusters: Vec<Vec<usize>> = Vec::new();
    let mut cluster_interval: Vec<usize> = Vec::new(); // which interval each cluster belongs to

    for (interval_idx, (lo, hi)) in intervals.iter().enumerate() {
        let points_in_bin: Vec<usize> = (0..n)
            .filter(|&i| filter_values[i] >= *lo - 1e-10 && filter_values[i] <= *hi + 1e-10)
            .collect();

        if points_in_bin.is_empty() {
            continue;
        }

        // Single-linkage clustering
        let clusters = single_linkage_cluster(cloud, &points_in_bin, config.cluster_epsilon);

        for cluster in clusters {
            cluster_interval.push(interval_idx);
            all_clusters.push(cluster);
        }
    }

    // Build edges: clusters connected if they share points
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let num_clusters = all_clusters.len();

    for i in 0..num_clusters {
        for j in (i + 1)..num_clusters {
            // Only check clusters from overlapping intervals
            let interval_diff = (cluster_interval[i] as i64 - cluster_interval[j] as i64).unsigned_abs();
            if interval_diff <= 1 {
                let shared = all_clusters[i].iter()
                    .any(|p| all_clusters[j].contains(p));
                if shared {
                    edges.push((i, j));
                }
            }
        }
    }

    MapperGraph::new(all_clusters, edges)
}

/// Single-linkage clustering of a subset of points.
fn single_linkage_cluster(
    cloud: &PointCloud,
    point_indices: &[usize],
    epsilon: f64,
) -> Vec<Vec<usize>> {
    let n = point_indices.len();
    if n == 0 {
        return vec![];
    }

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let d = cloud.distance(point_indices[i], point_indices[j]);
            if d <= epsilon {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    // Group by root
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(point_indices[i]);
    }

    groups.into_values().collect()
}

/// Convenience: build a Mapper graph using the first coordinate as filter.
pub fn mapper_graph_first_coord(cloud: &PointCloud, config: &MapperConfig) -> MapperGraph {
    build_mapper_graph(cloud, |p| p[0], config)
}

/// Convenience: build a Mapper graph using L2 norm as filter.
pub fn mapper_graph_norm(cloud: &PointCloud, config: &MapperConfig) -> MapperGraph {
    build_mapper_graph(cloud, |p| p.iter().map(|x| x * x).sum::<f64>().sqrt(), config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapper_two_clusters() {
        let mut points = Vec::new();
        for i in 0..10 {
            points.push(vec![-5.0 + i as f64 * 0.1, 0.0]);
        }
        for i in 0..10 {
            points.push(vec![5.0 + i as f64 * 0.1, 0.0]);
        }
        let cloud = PointCloud::from_points(points);
        let config = MapperConfig {
            num_intervals: 5,
            overlap: 0.3,
            cluster_epsilon: 1.0,
        };
        let graph = build_mapper_graph(&cloud, |p| p[0], &config);
        assert!(graph.node_count() >= 2);
        // Two clusters should produce at least 2 nodes
    }

    #[test]
    fn test_mapper_single_point() {
        let cloud = PointCloud::from_points(vec![vec![1.0, 2.0]]);
        let config = MapperConfig::default();
        let graph = mapper_graph_first_coord(&cloud, &config);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_mapper_connected_components() {
        // Line of closely spaced points
        let points: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1, 0.0]).collect();
        let cloud = PointCloud::from_points(points);
        let config = MapperConfig {
            num_intervals: 3,
            overlap: 0.3,
            cluster_epsilon: 0.5,
        };
        let graph = mapper_graph_first_coord(&cloud, &config);
        // Should be one connected component
        assert_eq!(graph.connected_components(), 1);
    }

    #[test]
    fn test_mapper_all_same_filter() {
        let cloud = PointCloud::from_points(vec![vec![1.0], vec![1.0], vec![1.0]]);
        let config = MapperConfig::default();
        let graph = mapper_graph_first_coord(&cloud, &config);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_mapper_2d_data() {
        // Points in two blobs
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(vec![i as f64 * 0.1, j as f64 * 0.1]);
            }
        }
        for i in 0..5 {
            for j in 0..5 {
                points.push(vec![10.0 + i as f64 * 0.1, j as f64 * 0.1]);
            }
        }
        let cloud = PointCloud::from_points(points);
        let config = MapperConfig {
            num_intervals: 10,
            overlap: 0.3,
            cluster_epsilon: 1.0,
        };
        let graph = mapper_graph_norm(&cloud, &config);
        assert!(graph.node_count() >= 2);
    }
}
