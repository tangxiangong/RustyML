use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::rng;
use std::collections::HashMap;
use crate::ModelError;

/// Mean Shift clustering algorithm implementation.
///
/// Mean Shift is a centroid-based clustering algorithm that works by iteratively shifting
/// data points towards areas of higher density. Each data point moves in the direction of
/// the mean of points within its current window until convergence. The algorithm does not
/// require specifying the number of clusters in advance.
///
/// # Fields
///
/// * `bandwidth` - The kernel bandwidth parameter that determines the search radius. Larger values lead to fewer clusters.
/// * `max_iter` - Maximum number of iterations to prevent infinite loops.
/// * `tol` - Convergence tolerance threshold. Points are considered converged when they move less than this value.
/// * `bin_seeding` - Whether to use bin seeding strategy for faster algorithm execution.
/// * `cluster_all` - Whether to assign all points to clusters, including potential noise.
///
/// # Examples
///
/// ```
/// use rustyml::machine_learning::meanshift::MeanShift;
/// use ndarray::Array2;
///
/// // Create a 2D dataset
/// let data = Array2::<f64>::from_shape_vec((10, 2),
///     vec![1.0, 2.0, 1.1, 2.2, 0.9, 1.9, 1.0, 2.1,
///          10.0, 10.0, 10.2, 9.9, 10.1, 10.0, 9.9, 9.8,
///          5.0, 5.0, 5.1, 4.9]).unwrap();
///
/// // Create a MeanShift instance with default parameters
/// let mut ms = MeanShift::default();
///
/// // Fit the model and predict cluster labels
/// let labels = ms.fit_predict(&data);
///
/// // Get the cluster centers
/// let centers = ms.get_cluster_centers().unwrap();
/// ```
///
/// # Notes
///
/// * If unsure about an appropriate bandwidth value, use the `estimate_bandwidth` function.
/// * The bandwidth parameter significantly affects algorithm performance and should be chosen carefully based on data characteristics.
/// * For large datasets, setting `bin_seeding = true` can improve performance.
#[derive(Debug, Clone)]
pub struct MeanShift {
    /// Bandwidth parameter that controls the kernel width
    bandwidth: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence threshold
    tol: f64,
    /// Whether to use bin seeding for initialization
    bin_seeding: bool,
    /// Whether to assign all points to clusters
    cluster_all: bool,
    /// Number of samples per cluster center
    n_samples_per_center: Option<Array1<usize>>,
    /// Cluster centers
    cluster_centers: Option<Array2<f64>>,
    /// Cluster labels for each sample
    labels: Option<Array1<usize>>,
    /// Actual number of iterations
    n_iter: Option<usize>,
}

impl Default for MeanShift {
    fn default() -> Self {
        Self::new(
            1.0,
            None,
            None,
            None,
            None,
        )
    }
}


impl MeanShift {
    /// Creates a new MeanShift instance with the specified parameters.
    ///
    /// # Parameters
    /// * `bandwidth` - The bandwidth parameter that determines the size of the kernel.
    /// * `max_iter` - The maximum number of iterations for the mean shift algorithm.
    /// * `tol` - The convergence threshold for the algorithm.
    /// * `bin_seeding` - Whether to use bin seeding for initialization.
    /// * `cluster_all` - Whether to assign all points to clusters, even those far from any centroid.
    ///
    /// # Returns
    /// * `Self` - A new MeanShift instance.
    pub fn new(
        bandwidth: f64,
        max_iter: Option<usize>,
        tol: Option<f64>,
        bin_seeding: Option<bool>,
        cluster_all: Option<bool>,
    ) -> Self {
        MeanShift {
            bandwidth,
            max_iter: max_iter.unwrap_or(300),
            tol: tol.unwrap_or(1e-3),
            bin_seeding: bin_seeding.unwrap_or(false),
            cluster_all: cluster_all.unwrap_or(true),
            n_samples_per_center: None,
            cluster_centers: None,
            labels: None,
            n_iter: None,
        }
    }

    /// Gets the cluster centers found by the algorithm.
    ///
    /// # Returns
    /// * `Ok(Array2<f64>)` - A Result containing the cluster centers as a ndarray `Array2<f64>`
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_cluster_centers(&self) -> Result<Array2<f64>, ModelError> {
        match self.cluster_centers.as_ref() {
            Some(centers) => Ok(centers.clone()),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the cluster labels assigned to each data point.
    ///
    /// # Returns
    /// - `Ok(Array1<usize>)` - A Result containing the cluster labels as a ndarray `Array1<usize>`
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_labels(&self) -> Result<Array1<usize>, ModelError> {
        match self.labels.as_ref() {
            Some(labels) => Ok(labels.clone()),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the number of iterations the algorithm performed.
    ///
    /// # Returns
    /// - `Ok(usize)` - A Result containing the number of iterations or an error
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_n_iter(&self) -> Result<usize, ModelError> {
        match self.n_iter.as_ref() {
            Some(n_iter) => Ok(*n_iter),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the number of samples per cluster center.
    ///
    /// # Returns
    /// - `Ok(Array1<usize>)` - A Result containing the number of samples per center as a ndarray `Array1<usize>`
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_n_samples_per_center(&self) -> Result<Array1<usize>, ModelError> {
        match self.n_samples_per_center.as_ref() {
            Some(n_samples_per_center) => Ok(n_samples_per_center.clone()),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the bandwidth parameter value.
    ///
    /// # Returns
    /// * `f64` - The bandwidth value.
    pub fn get_bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Gets the maximum number of iterations.
    ///
    /// # Returns
    /// * `usize` - The maximum number of iterations.
    pub fn get_max_iter(&self) -> usize {
        self.max_iter
    }

    /// Gets the convergence tolerance.
    ///
    /// # Returns
    /// * `f64` - The tolerance value.
    pub fn get_tol(&self) -> f64 {
        self.tol
    }

    /// Gets the bin seeding setting.
    ///
    /// # Returns
    /// * `bool` - A boolean indicating whether bin seeding is enabled.
    pub fn get_bin_seeding(&self) -> bool {
        self.bin_seeding
    }

    /// Gets the cluster_all setting.
    ///
    /// # Returns
    /// * `bool` - A boolean indicating whether all points are assigned to clusters.
    pub fn get_cluster_all(&self) -> bool {
        self.cluster_all
    }

    /// Calculates the squared Euclidean distance between two 1D arrays
    ///
    /// This method converts two 1D arrays to 2D array views with a single row
    /// and then uses the `squared_euclidean_distance` function to calculate
    /// the distance between them.
    ///
    /// # Arguments
    ///
    /// * `x` - First vector as a 1D array
    /// * `y` - Second vector as a 1D array
    ///
    /// # Returns
    ///
    /// * `f64` - The squared Euclidean distance between the two vectors
    fn calculate_distance(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        use crate::math::squared_euclidean_distance;
        // Convert 1D arrays to 2D array views
        let x_2d = x.view().insert_axis(ndarray::Axis(0));
        let y_2d = y.view().insert_axis(ndarray::Axis(0));

        // Use the provided function to calculate the distance
        squared_euclidean_distance(&x_2d, &y_2d)
    }

    /// Fits the MeanShift clustering model to the input data.
    ///
    /// # Parameters
    /// * `x` - The input data as a ndarray `Array2<f64>` where each row is a sample.
    ///
    /// # Returns
    /// - `Ok(&mut Self)` - A mutable reference to the fitted model
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self, ModelError> {
        if self.bandwidth <= 0.0 {
            return Err(ModelError::InputValidationError("bandwidth must be positive".to_string()));
        }

        if self.max_iter <= 0 {
            return Err(ModelError::InputValidationError("max_iter must be positive".to_string()));
        }

        if self.tol <= 0.0 {
            return Err(ModelError::InputValidationError("tol must be positive".to_string()));
        }

        use crate::math::gaussian_kernel;
        use super::preliminary_check;

        preliminary_check(&x, None)?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        // Initialize seed points
        let seeds: Vec<usize> = if self.bin_seeding {
            self.get_bin_seeds(x)
        } else {
            // Randomly select points as initial seeds
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = rng();
            indices.shuffle(&mut rng);
            // Limit number of seeds to avoid excessive computation
            let max_seeds = n_samples.min(100);
            indices[..max_seeds].to_vec()
        };

        // Perform mean shift for each seed point
        let mut centers: Vec<Array1<f64>> = Vec::new();

        for seed_idx in seeds {
            let mut center = x.row(seed_idx).to_owned();
            let mut completed_iterations = 0;

            loop {
                let mut new_center = Array1::zeros(n_features);
                let mut weight_sum = 0.0;

                // Calculate distance from each point to current center
                let mut distances = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let point = x.row(i).to_owned();
                    let dist = self.calculate_distance(&center, &point);
                    distances.push(dist);
                }

                // Convert to ndarray
                let distances = Array1::from(distances);

                // Apply Gaussian kernel
                let mut weights = Array1::zeros(n_samples);
                let gamma = 1.0 / (2.0 * self.bandwidth.powi(2));
                let zero_matrix = Array2::<f64>::zeros((1, 1));
                let zero_view = zero_matrix.view();

                for i in 0..n_samples {
                    // turn to 2D View
                    let dist_matrix = Array2::<f64>::from_elem((1, 1), distances[i]);
                    let dist_view = dist_matrix.view();

                    // use function `gaussian_kernel`
                    weights[i] = gaussian_kernel(&zero_view, &dist_view, gamma);
                }

                // Calculate weighted average
                for i in 0..n_samples {
                    let weight = weights[i];
                    if weight > 0.0 {
                        let point = x.row(i);
                        for j in 0..n_features {
                            new_center[j] += point[j] * weight;
                        }
                        weight_sum += weight;
                    }
                }

                // Normalize
                if weight_sum > 0.0 {
                    new_center.mapv_inplace(|x| x / weight_sum);
                }

                // Check convergence
                let shift = self.calculate_distance(&center, &new_center).sqrt();
                center = new_center;

                completed_iterations += 1;

                if shift < self.tol || completed_iterations >= self.max_iter {
                    self.n_iter = Some(completed_iterations);
                    break;
                }
            }

            // Record found center
            centers.push(center);
        }

        // Merge similar centers
        let mut unique_centers: Vec<Array1<f64>> = Vec::new();
        let mut center_counts: Vec<usize> = Vec::new();

        for center in centers {
            let mut is_unique = true;

            for (i, unique_center) in unique_centers.iter().enumerate() {
                let distance = self.calculate_distance(&center, unique_center).sqrt();
                if distance < self.bandwidth {
                    // Update existing center (weighted average)
                    let count = center_counts[i];
                    let new_count = count + 1;
                    let updated_center = unique_centers[i].clone() * (count as f64 / new_count as f64) +
                        center.clone() * (1.0 / new_count as f64);
                    unique_centers[i] = updated_center;
                    center_counts[i] = new_count;
                    is_unique = false;
                    break;
                }
            }

            if is_unique {
                unique_centers.push(center);
                center_counts.push(1);
            }
        }


        // Create cluster_centers array
        let n_clusters = unique_centers.len();
        let mut cluster_centers = Array2::zeros((n_clusters, n_features));
        for i in 0..n_clusters {
            cluster_centers.row_mut(i).assign(&unique_centers[i]);
        }

        // Assign cluster labels to each data point
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let point = x.row(i).to_owned();
            let mut min_dist = f64::INFINITY;
            let mut label = 0;

            for (j, center) in unique_centers.iter().enumerate() {
                let dist = self.calculate_distance(&point, center);
                if dist < min_dist {
                    min_dist = dist;
                    label = j;
                }
            }

            // If not cluster_all and distance is too far, mark as outlier
            if !self.cluster_all && min_dist > self.bandwidth.powi(2) {
                labels[i] = n_clusters; // Use n_clusters as outlier label
            } else {
                labels[i] = label;
            }
        }

        self.cluster_centers = Some(cluster_centers);
        self.labels = Some(labels);
        self.n_samples_per_center = Some(Array1::from(center_counts));

        // print training info
        println!("Mean shift model training finished at iteration {}, number of clusters: {}",
                self.n_iter.unwrap_or(0), n_clusters);

        Ok(self)
    }

    /// Predicts cluster labels for the input data.
    ///
    /// # Parameters
    /// * `x` - The input data as a ndarray `Array2<f64>` where each row is a sample.
    ///
    /// # Returns
    /// - `Ok(Array1<usize>)` - containing the predicted cluster labels.
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, ModelError> {
        if let Some(centers) = &self.cluster_centers {
            let n_samples = x.shape()[0];
            let n_clusters = centers.shape()[0];

            let mut labels = Array1::zeros(n_samples);

            for i in 0..n_samples {
                let point = x.row(i).to_owned();
                let mut min_dist = f64::INFINITY;
                let mut label = 0;

                for j in 0..n_clusters {
                    let center = centers.row(j).to_owned();
                    let dist = self.calculate_distance(&point, &center);
                    if dist < min_dist {
                        min_dist = dist;
                        label = j;
                    }
                }

                // If not cluster_all and distance is too far, mark as outlier
                if !self.cluster_all && min_dist > self.bandwidth.powi(2) {
                    labels[i] = n_clusters; // Use n_clusters as outlier label
                } else {
                    labels[i] = label;
                }
            }

            Ok(labels)
        } else {
            Err(ModelError::NotFitted)
        }
    }

    /// Fits the model to the input data and predicts cluster labels.
    ///
    /// # Parameters
    /// * `x` - The input data as a ndarray `Array2<f64>` where each row is a sample.
    ///
    /// # Returns
    /// - `Ok(Array1<usize>)` - containing the predicted cluster labels.
    /// - `Err(ModelError::InputValidationError(&str))` - Input does not match expectation
    pub fn fit_predict(&mut self, x: &Array2<f64>) -> Result<Array1<usize>, ModelError> {
        self.fit(x)?;
        Ok(self.labels.clone().unwrap())
    }

    /// Generates initial seeds for the clustering algorithm using binning.
    ///
    /// # Parameters
    /// * `x` - The input data as a ndarray Array2<f64> where each row is a sample.
    ///
    /// # Returns
    /// * `Vec<usize>` - A vector of indices representing the initial seed points.
    fn get_bin_seeds(&self, x: &Array2<f64>) -> Vec<usize> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        // Calculate min and max for each feature
        let mut mins = Vec::with_capacity(n_features);
        let mut maxs = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let min = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            mins.push(min);
            maxs.push(max);
        }

        // Create grid
        let bin_size = self.bandwidth;
        let mut bins: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();

        for i in 0..n_samples {
            let point = x.row(i);
            let mut bin_index = Vec::with_capacity(n_features);

            for j in 0..n_features {
                let idx = ((point[j] - mins[j]) / bin_size).floor() as i64;
                bin_index.push(idx);
            }

            bins.entry(bin_index).or_insert_with(Vec::new).push(i);
        }

        // Select one point from each grid cell as seed
        let mut seeds = Vec::new();
        for (_, indices) in bins {
            if !indices.is_empty() {
                seeds.push(indices[0]);
            }
        }

        seeds
    }
}

/// Estimates the bandwidth to use with the MeanShift algorithm.
///
/// The bandwidth is estimated based on the pairwise distances between a subset of points.
///
/// # Parameters
/// * `x` - The input data as a ndarray `Array2<f64>` where each row is a sample.
/// * `quantile` - The quantile of the pairwise distances to use as the bandwidth.
/// * `n_samples` - The number of samples to use for the distance calculation.
/// * `random_state` - Seed for random number generation.
///
/// # Returns
/// * `f64` - The estimated bandwidth.
pub fn estimate_bandwidth(
    x: &Array2<f64>,
    quantile: Option<f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>
) -> f64 {
    use rand::SeedableRng;

    let quantile = quantile.unwrap_or(0.3);
    if quantile <= 0.0 || quantile >= 1.0 {
        panic!("quantile should be in range ]0, 1[");
    }

    let (n_samples_total, _) = x.dim();
    let n_samples = n_samples.unwrap_or(n_samples_total);

    let mut rng = match random_state {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => {
            let mut thread_rng = rand::rng();
            rand::rngs::StdRng::from_rng(&mut thread_rng)
        },
    };

    // If we have fewer samples than requested, use all samples
    let x_samples = if n_samples >= n_samples_total {
        x.clone()
    } else {
        // Random sampling
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..n_samples_total).collect();
        indices.shuffle(&mut rng);
        let indices = &indices[..n_samples];

        let mut samples = Array2::zeros((n_samples, x.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            samples.row_mut(i).assign(&x.row(idx));
        }
        samples
    };

    // Compute distances between all pairs of points
    let mut distances = Vec::new();
    for i in 0..n_samples {
        let point_i = x_samples.row(i);
        for j in (i+1)..n_samples {
            let point_j = x_samples.row(j);
            // Euclidean distance
            let dist = point_i.iter()
                .zip(point_j.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            distances.push(dist);
        }
    }

    // Sort distances and select the value at the specified quantile
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = (distances.len() as f64 * quantile) as usize;
    distances.get(k).copied().unwrap_or(0.0)
}