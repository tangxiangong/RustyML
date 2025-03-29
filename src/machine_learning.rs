/// Represents different distance calculation methods used in various machine learning algorithms.
///
/// This enum defines common distance metrics that can be used in clustering algorithms,
/// nearest neighbor searches, and other applications where distance between points is relevant.
/// # Variants
///
/// * `Euclidean` - Euclidean distance (L2 norm), calculated as the square root of the sum of squared differences between corresponding coordinates.
/// * `Manhattan` - Manhattan distance (L1 norm), calculated as the sum of absolute differences between corresponding coordinates.
/// * `Minkowski` - A generalized metric that includes both Euclidean and Manhattan distances as special cases. Requires an additional parameter p (not implemented in this enum).

#[derive(Debug, Clone, PartialEq)]
pub enum DistanceCalculationMetric {
    /// Euclidean distance (L2 norm) - the straight-line distance between two points.
    Euclidean,
    /// Manhattan distance (L1 norm) - the sum of absolute differences between coordinates.
    Manhattan,
    /// Minkowski distance - a generalized metric that includes both Euclidean and Manhattan distances.
    Minkowski,
}

pub mod linear_regression;
pub mod logistic_regression;
pub mod kmeans;
pub mod knn;
pub mod meanshift;
pub mod dbscan;
pub mod decision_tree;