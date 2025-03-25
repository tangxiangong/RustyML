use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

/// # K-Nearest Neighbors (KNN) Classifier
///
/// A non-parametric classification algorithm that classifies new data points
/// based on the majority class of its k nearest neighbors.
///
/// ## Type Parameters
///
/// * `T` - The type of target values. Must implement `Clone`, `Hash`, and `Eq` traits.
///
/// ## Fields
///
/// * `k` - Number of neighbors to consider for classification
/// * `x_train` - Training data features as a 2D array
/// * `y_train` - Training data labels/targets
/// * `weights` - Weight function for neighbor votes. Options: "uniform"(default), "distance"
/// * `metric` - Distance metric used for finding neighbors. Options: "euclidean"(default), "manhattan", "minkowski"
pub struct KNN<T> {
    k: usize,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<T>>,
    weights: String,
    metric: String,
}

impl<T: Clone + std::hash::Hash + Eq> Default for KNN<T> {
    /// Creates a new KNN classifier with default parameters:
    /// * k = 5
    /// * weights = "uniform"
    /// * metric = "euclidean"
    fn default() -> Self {
        KNN {
            k: 5,
            x_train: None,
            y_train: None,
            weights: "uniform".to_string(),
            metric: "euclidean".to_string(),
        }
    }
}

impl<T: Clone + std::hash::Hash + Eq> KNN<T> {
    /// Creates a new KNN classifier with the specified parameters
    ///
    /// # Arguments
    ///
    /// * `k` - Number of neighbors to use for classification
    /// * `weights` - Weighting strategy for neighbor votes ("uniform" or "distance")
    /// * `metric` - Distance metric to use ("euclidean", "manhattan" or "minkowski")
    ///
    /// # Returns
    ///
    /// A new KNN classifier instance
    pub fn new(k: usize, weights: &str, metric: &str) -> Self {
        KNN {
            k,
            x_train: None,
            y_train: None,
            weights: weights.to_string(),
            metric: metric.to_string(),
        }
    }

    /// Fits the KNN classifier to the training data
    ///
    /// # Arguments
    ///
    /// * `x` - Training features as a 2D array (samples × features)
    /// * `y` - Training targets/labels as a 1D array
    /// 
    /// # Notes
    /// 
    /// KNN is a lazy learning algorithm, and the calculation is done in the prediction phase.
    pub fn fit(&mut self, x: Array2<f64>, y: Array1<T>) {
        self.x_train = Some(x);
        self.y_train = Some(y);
    }

    /// Predicts the class labels for the provided data points
    ///
    /// # Arguments
    ///
    /// * `x` - Data points to classify as a 2D array (samples × features)
    ///
    /// # Returns
    ///
    /// A vector containing the predicted class labels
    pub fn predict(&self, x: ArrayView2<f64>) -> Vec<T> {
        let x_train = self.x_train.as_ref().expect("Model not trained");
        let y_train = self.y_train.as_ref().expect("Model not trained");

        let mut predictions = Vec::with_capacity(x.nrows());

        for i in 0..x.nrows() {
            let sample = x.row(i);
            predictions.push(self.predict_one(sample, x_train.view(), y_train));
        }

        predictions
    }

    /// Calculates the distance between two points based on the selected metric
    ///
    /// # Arguments
    ///
    /// * `a` - First point as a 1D array
    /// * `b` - Second point as a 1D array
    ///
    /// # Returns
    ///
    /// The calculated distance between points `a` and `b`
    fn calculate_distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        use crate::math::{squared_euclidean_distance, manhattan_distance, minkowski_distance};
        let a = a.insert_axis(ndarray::Axis(0));
        let b = b.insert_axis(ndarray::Axis(0));
        match self.metric.as_str() {
            "euclidean" => {
                squared_euclidean_distance(&a, &b).sqrt()
            },

            "manhattan" => manhattan_distance(&a, &b),
            "minkowski" => minkowski_distance(&a, &b, 3.0),
            _ => {
                // Default to Euclidean distance
                squared_euclidean_distance(&a, &b).sqrt()
            }
        }
    }

    /// Predicts the class for a single data point
    ///
    /// # Arguments
    ///
    /// * `x` - The data point to classify as a 1D array
    /// * `x_train` - Training data features
    /// * `y_train` - Training data labels
    ///
    /// # Returns
    ///
    /// The predicted class for the data point
    fn predict_one(&self, x: ArrayView1<f64>, x_train: ArrayView2<f64>, y_train: &Array1<T>) -> T {
        let n_samples = x_train.nrows();

        // Calculate distances to all training samples
        let mut distances = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let distance = self.calculate_distance(x, x_train.row(i));
            distances.push((distance, i));
        }

        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Get k nearest neighbors
        let k_neighbors: Vec<_> = distances.iter().take(self.k).collect();

        // Calculate based on weight strategy
        match self.weights.as_str() {
            "uniform" => {
                // Count class occurrences
                let mut class_counts: HashMap<&T, usize> = HashMap::new();
                for &(_, idx) in k_neighbors {
                    let class = &y_train[idx];
                    *class_counts.entry(class).or_insert(0) += 1;
                }

                // Find the most common class
                class_counts
                    .iter()
                    .max_by_key(|&(_, &count)| count)
                    .map(|(class, _)| (*class).clone())
                    .unwrap()
            },
            "distance" => {
                // Weight by inverse distance
                let mut class_weights: HashMap<&T, f64> = HashMap::new();
                for &(distance, idx) in k_neighbors {
                    // Avoid division by zero
                    let weight = if distance > 0.0 { 1.0 / distance } else { f64::MAX };
                    let class = &y_train[idx];
                    *class_weights.entry(class).or_insert(0.0) += weight;
                }

                // Find the class with highest weight
                class_weights
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(class, _)| (*class).clone())
                    .unwrap()
            },
            _ => {
                // Default to uniform weights
                let mut class_counts: HashMap<&T, usize> = HashMap::new();
                for &(_, idx) in k_neighbors {
                    let class = &y_train[idx];
                    *class_counts.entry(class).or_insert(0) += 1;
                }

                class_counts
                    .iter()
                    .max_by_key(|&(_, &count)| count)
                    .map(|(class, _)| (*class).clone())
                    .unwrap()
            }
        }
    }

    /// Returns the number of neighbors (k) used in the KNN algorithm
    ///
    /// # Returns
    ///
    /// The value of k, representing how many nearest neighbors are considered for predictions
    pub fn get_k(&self) -> usize {
        self.k
    }

    /// Returns the weighting strategy used in the KNN algorithm
    ///
    /// # Returns
    ///
    /// The weight function name, either "uniform" or "distance"
    pub fn get_weights(&self) -> &str {
        &self.weights
    }

    /// Returns the distance metric used for calculating point similarities
    ///
    /// # Returns
    ///
    /// The metric name, such as "euclidean" or "manhattan"
    pub fn get_metric(&self) -> &str {
        &self.metric
    }

    /// Returns a reference to the training features if available
    ///
    /// # Returns
    ///
    /// Some reference to the training data features if the model has been trained,
    /// or None if the model hasn't been trained yet
    pub fn get_x_train(&self) -> Option<&Array2<f64>> {
        self.x_train.as_ref()
    }

    /// Returns a reference to the training labels if available
    ///
    /// # Returns
    ///
    /// Some reference to the training data labels if the model has been trained,
    /// or None if the model hasn't been trained yet
    pub fn get_y_train(&self) -> Option<&Array1<T>> {
        self.y_train.as_ref()
    }
}