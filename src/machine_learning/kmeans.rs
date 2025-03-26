use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;
use rand::SeedableRng;
use std::ops::AddAssign;
use crate::ModelError;

/// KMeans clustering algorithm implementation.
///
/// This struct implements the K-Means clustering algorithm, which partitions
/// n observations into k clusters where each observation belongs to the cluster
/// with the nearest mean (centroid).
///
/// # Fields
///
/// * `n_clusters` - Number of clusters to form
/// * `max_iter` - Maximum number of iterations for a single run
/// * `tol` - Tolerance for declaring convergence
/// * `random_seed` - Optional seed for random number generation
/// * `centroids` - Computed cluster centers after fitting
/// * `labels` - Cluster labels for training data after fitting
/// * `inertia` - Sum of squared distances to closest centroid after fitting
/// * `n_iter` - Number of iterations the algorithm ran for after fitting
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Number of clusters
    n_clusters: usize,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence threshold
    tol: f64,
    /// Random seed
    random_seed: Option<u64>,
    /// Cluster centers
    centroids: Option<Array2<f64>>,
    /// Labels of each sample point (clustering result)
    labels: Option<Array1<usize>>,
    /// Sum of squared distances of samples to their closest cluster center
    inertia: Option<f64>,
    /// Actual number of iterations
    n_iter: Option<usize>,
}

impl Default for KMeans {
    fn default() -> Self {
        KMeans::new(8, 300, 1e-4, None)
    }
}

impl KMeans {
    /// Creates a new KMeans instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `n_clusters` - Number of clusters to form
    /// * `max_iter` - Maximum number of iterations for the algorithm
    /// * `tol` - Convergence tolerance, the algorithm stops when the centroids move less than this value
    /// * `random_seed` - Optional seed for random number generation to ensure reproducibility
    ///
    /// # Returns
    ///
    /// A new KMeans instance with the specified configuration
    pub fn new(n_clusters: usize,
               max_iterations: usize,
               tolerance: f64,
               random_seed: Option<u64>
    ) -> Self {
        KMeans {
            n_clusters,
            max_iter: max_iterations,
            tol: tolerance,
            random_seed,
            centroids: None,
            labels: None,
            inertia: None,
            n_iter: None,
        }
    }

    /// Returns the number of clusters (k) that the KMeans algorithm will use.
    ///
    /// # Returns
    ///
    /// The number of clusters as a `usize`.
    pub fn get_n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Returns the maximum number of iterations allowed for the KMeans algorithm.
    ///
    /// # Returns
    ///
    /// The maximum number of iterations as a `usize`.
    pub fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the tolerance for declaring convergence in the KMeans algorithm.
    /// Convergence is declared when the change in inertia is less than this value.
    ///
    /// # Returns
    ///
    /// The convergence tolerance as a `f64`.
    pub fn get_tol(&self) -> f64 {
        self.tol
    }

    /// Returns the random seed used for centroid initialization.
    ///
    /// # Returns
    ///
    /// - `Ok(seed)` - The random seed as a `u64` if it has been set.
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet and no seed is available.
    pub fn get_random_seed(&self) -> Result<u64, ModelError> {
        match self.random_seed {
            Some(seed) => Ok(seed),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the cluster centroids if the model has been fitted.
    ///
    /// # Returns
    ///
    /// An optional reference to the cluster centroids as a 2D array,
    /// or None if the model has not been fitted
    pub fn get_centroids(&self) -> Result<&Array2<f64>, ModelError> {
        match self.centroids.as_ref() {
            Some(centroids) => Ok(centroids),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the cluster assignments for the training data if the model has been fitted.
    ///
    /// # Returns
    ///
    /// An optional reference to the cluster assignments as a 1D array,
    /// or None if the model has not been fitted
    pub fn get_labels(&self) -> Result<&Array1<usize>, ModelError> {
        match self.labels.as_ref() {
            Some(labels) => Ok(labels),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the sum of squared distances of samples to their closest centroid.
    ///
    /// # Returns
    ///
    /// An optional float value representing the inertia,
    /// or None if the model has not been fitted
    pub fn get_inertia(&self) -> Result<f64, ModelError> {
        match self.inertia {
            Some(inertia) => Ok(inertia),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the number of iterations the algorithm ran for during fitting.
    ///
    /// # Returns
    ///
    /// An optional value representing the number of iterations,
    /// or None if the model has not been fitted
    pub fn get_n_iter(&self) -> Result<usize, ModelError> {
        match self.n_iter {
            Some(n_iter) => Ok(n_iter),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Finds the closest centroid to a given data point and returns its index and distance.
    ///
    /// # Arguments
    ///
    /// * `x` - Data point as a 2D array view
    ///
    /// # Returns
    ///
    /// A tuple containing the index of the closest centroid and the squared distance to it
    fn closest_centroid(&self, x: &ArrayView2<f64>) -> (usize, f64) {
        use crate::math::squared_euclidean_distance;
        let centroids = self.centroids.as_ref().unwrap();

        let mut min_dist = f64::MAX;
        let mut min_idx = 0;

        for (i, centroid) in centroids.outer_iter().enumerate() {
            let centroid_shaped = centroid.to_shape((1, centroid.len())).unwrap();
            let centroid_view = centroid_shaped.view();
            let dist = squared_euclidean_distance(&x, &centroid_view);
            if dist < min_dist {
                min_dist = dist;
                min_idx = i;
            }
        }

        (min_idx, min_dist)
    }

    /// Initializes cluster centroids using random selection from the data points.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data as a 2D array
    fn init_centroids(&mut self, data: &Array2<f64>) {
        use crate::math::squared_euclidean_distance;
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // Initialize cluster centers matrix
        let mut centroids = Array2::<f64>::zeros((self.n_clusters, n_features));

        let mut rng = match self.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(0),
        };


        // k-means++ initialization method

        // Randomly select the first center point
        let first_center_idx = rng.random_range(0..n_samples);
        centroids.row_mut(0).assign(&data.row(first_center_idx));

        // Select the remaining center points
        for k in 1..self.n_clusters {
            // Calculate the distance from each point to the nearest center
            let mut distances = Vec::with_capacity(n_samples);
            let mut total_dist = 0.0;

            for i in 0..n_samples {
                let row_i = data.row(i);
                let sample_shaped = row_i.to_shape((1, n_features)).unwrap();
                let sample_view = sample_shaped.view();

                // Find the closest already selected center point
                let mut min_dist = f64::MAX;

                for j in 0..k {
                    let row_j = centroids.row(j);
                    let centroid_shaped = row_j.to_shape((1, n_features)).unwrap();
                    let centroid_view = centroid_shaped.view();
                    let dist = squared_euclidean_distance(&sample_view, &centroid_view);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }

                distances.push(min_dist);
                total_dist += min_dist;
            }

            // Use roulette wheel selection to choose the next center point
            let mut cumulative_dist = 0.0;
            let choice = rng.random::<f64>() * total_dist;

            for (i, dist) in distances.iter().enumerate() {
                cumulative_dist += dist;
                if cumulative_dist >= choice {
                    centroids.row_mut(k).assign(&data.row(i));
                    break;
                }
            }
        }

        self.centroids = Some(centroids);
    }

    /// Fits the KMeans model to the training data.
    ///
    /// This method computes cluster centroids and assigns each data point to its closest centroid.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data as a 2D array where each row is a sample
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn fit(&mut self, data: &Array2<f64>) -> &mut Self {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        // Initialize cluster centers
        self.init_centroids(data);

        let mut labels = Array1::<usize>::zeros(n_samples);
        let mut old_inertia = f64::MAX;
        let mut inertia;
        let mut iter_count = 0;

        // Main iteration loop
        for i in 0..self.max_iter {
            // Assign sample points to the nearest cluster center
            inertia = 0.0;

            for (idx, sample) in data.outer_iter().enumerate() {
                let sample_shaped = sample.to_shape((1, n_features)).unwrap();
                let sample_view = sample_shaped.view();
                let (closest_idx, dist) = self.closest_centroid(&sample_view);
                labels[idx] = closest_idx;
                inertia += dist;
            }

            // Check for convergence
            if (old_inertia - inertia).abs() < self.tol * old_inertia {
                iter_count = i;
                break;
            }

            old_inertia = inertia;
            iter_count = i;

            // Update cluster centers
            let mut new_centroids = Array2::<f64>::zeros((self.n_clusters, n_features));
            let mut counts = vec![0; self.n_clusters];

            for (idx, sample) in data.outer_iter().enumerate() {
                let cluster_idx = labels[idx];
                new_centroids.row_mut(cluster_idx).add_assign(&sample);
                counts[cluster_idx] += 1;
            }

            // Calculate the mean of each cluster as a new center
            for (idx, count) in counts.iter().enumerate() {
                if *count > 0 {
                    let count_f = *count as f64;
                    new_centroids.row_mut(idx).mapv_inplace(|x| x / count_f);
                }
            }

            // Handle empty clusters
            for (idx, count) in counts.iter().enumerate() {
                if *count == 0 {
                    // If a cluster is empty, select the point furthest from the current centers as the new center
                    let mut max_dist = -1.0;
                    let mut farthest_idx = 0;

                    for (sample_idx, sample) in data.outer_iter().enumerate() {
                        let row_data = sample;
                        let sample_shaped = row_data.to_shape((1, n_features)).unwrap();
                        let sample_view = sample_shaped.view();
                        let (_, dist) = self.closest_centroid(&sample_view);

                        if dist > max_dist {
                            max_dist = dist;
                            farthest_idx = sample_idx;
                        }
                    }

                    new_centroids.row_mut(idx).assign(&data.row(farthest_idx));
                }
            }

            self.centroids = Some(new_centroids);
        }

        self.labels = Some(labels);
        self.inertia = Some(old_inertia);
        self.n_iter = Some(iter_count + 1);

        // print training info
        println!("KMeans model training finished at iteration {}, avg_cost: {}",
                 iter_count + 1, old_inertia / n_samples as f64);

        self
    }

    /// Predicts the closest cluster for each sample in the input data.
    ///
    /// # Arguments
    ///
    /// * `data` - New data points for which to predict cluster assignments
    ///
    /// # Returns
    ///
    /// An array of cluster indices for each input data point
    pub fn predict(&self, data: &Array2<f64>) -> Result<Array1<usize>, ModelError> {
        if self.centroids.is_none(){
            return Err(ModelError::NotFitted);
        }
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let mut labels = Array1::<usize>::zeros(n_samples);

        for (idx, sample) in data.outer_iter().enumerate() {
            let row_data = sample;
            let sample_shaped = row_data.to_shape((1, n_features)).unwrap();
            let sample_view = sample_shaped.view();
            let (closest_idx, _) = self.closest_centroid(&sample_view);
            labels[idx] = closest_idx;
        }

        Ok(labels)
    }

    /// Fits the model and predicts cluster indices for the input data.
    ///
    /// This is equivalent to calling `fit` followed by `predict`, but more efficient.
    ///
    /// # Arguments
    ///
    /// * `data` - Training data as a 2D array
    ///
    /// # Returns
    ///
    /// An array of cluster indices for each input data point
    pub fn fit_predict(&mut self, data: &Array2<f64>) -> Array1<usize> {
        self.fit(data);
        self.labels.clone().unwrap()
    }
}