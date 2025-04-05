use crate::ModelError;
pub use crate::machine_learning::svc::KernelType;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::eigh::Eigh;
use rayon::prelude::*;

/// # A Kernel Principal Component Analysis implementation.
///
/// KernelPCA performs a non-linear dimensionality reduction by using
/// kernel methods to map the data into a higher dimensional space where
/// linear PCA is performed.
///
/// ## Fields
///
/// * `kernel` - The kernel function type used to compute the kernel matrix
/// * `n_components` - The number of components to extract
/// * `eigenvalues` - Eigenvalues in descending order (available after fitting)
/// * `eigenvectors` - Corresponding normalized eigenvectors (available after fitting)
/// * `x_fit` - Training data, used for subsequent transformation of new data
/// * `row_means` - Mean of each row in the training kernel matrix (used for centering)
/// * `total_mean` - Overall mean of the training kernel matrix
///
/// ## Example
///
/// ```
/// use ndarray::{Array2, arr2};
/// use rustyml::utility::kernel_pca::{KernelPCA, KernelType};
///
/// // Create some sample data
/// let data = arr2(&[
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0],
/// ]);
///
/// // Create a KernelPCA model with RBF kernel and 2 components
/// let mut kpca = KernelPCA::new(
///     KernelType::RBF { gamma: 0.1 },
///     2
/// );
///
/// // Fit the model and transform the data
/// let transformed = kpca.fit_transform(&data).unwrap();
///
/// // The transformed data now has 2 columns instead of 3
/// assert_eq!(transformed.ncols(), 2);
/// assert_eq!(transformed.nrows(), 4);
///
/// // We can also transform new data
/// let new_data = arr2(&[
///     [2.0, 3.0, 4.0],
///     [5.0, 6.0, 7.0],
/// ]);
/// let new_transformed = kpca.transform(&new_data).unwrap();
/// assert_eq!(new_transformed.ncols(), 2);
/// assert_eq!(new_transformed.nrows(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct KernelPCA {
    kernel: KernelType,
    n_components: usize,
    eigenvalues: Option<Array1<f64>>, // Eigenvalues in descending order
    eigenvectors: Option<Array2<f64>>, // Corresponding normalized eigenvectors, each column is an eigenvector
    x_fit: Option<Array2<f64>>, // Training data, used for subsequent transformation of new data
    row_means: Option<Array1<f64>>, // Mean of each row in the training kernel matrix (used for centering)
    total_mean: Option<f64>,        // Overall mean of the training kernel matrix
}

/// Calculates the kernel value between two samples based on the specified kernel type.
///
/// This function computes the kernel function value between two feature vectors
/// according to the provided kernel type.
///
/// # Parameters
///
/// * `x` - First feature vector
/// * `y` - Second feature vector
/// * `kernel` - Kernel type configuration
///
/// # Returns
///
/// * `f64` - The computed kernel value as a floating-point number
pub fn compute_kernel(x: &ArrayView1<f64>, y: &ArrayView1<f64>, kernel: &KernelType) -> f64 {
    match kernel {
        KernelType::Linear => x.dot(y),
        KernelType::Poly {
            degree,
            gamma,
            coef0,
        } => (gamma * x.dot(y) + coef0).powi(*degree as i32),
        KernelType::RBF { gamma } => {
            let diff = x - y;
            let norm_sq = diff.dot(&diff);
            (-gamma * norm_sq).exp()
        }
        KernelType::Sigmoid { gamma, coef0 } => (gamma * x.dot(y) + coef0).tanh(),
    }
}

/// Default implementation for KernelPCA.
///
/// Creates a KernelPCA instance with default parameters:
/// - Linear kernel
/// - 2 components
impl Default for KernelPCA {
    fn default() -> Self {
        KernelPCA {
            kernel: KernelType::Linear,
            n_components: 2,
            eigenvalues: None,
            eigenvectors: None,
            x_fit: None,
            row_means: None,
            total_mean: None,
        }
    }
}

impl KernelPCA {
    /// Creates a new KernelPCA model with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `kernel` - The kernel function type to use
    /// * `n_components` - The number of components to extract
    ///
    /// # Returns
    ///
    /// * `Self` - A new KernelPCA instance
    pub fn new(kernel: KernelType, n_components: usize) -> Self {
        KernelPCA {
            kernel,
            n_components,
            eigenvalues: None,
            eigenvectors: None,
            x_fit: None,
            row_means: None,
            total_mean: None,
        }
    }

    /// Gets the kernel type used by this KernelPCA instance.
    ///
    /// # Returns
    ///
    /// * `&KernelType` - A reference to the kernel type
    pub fn get_kernel(&self) -> &KernelType {
        &self.kernel
    }

    /// Gets the number of components this KernelPCA instance extracts.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of components
    pub fn get_n_components(&self) -> usize {
        self.n_components
    }

    /// Gets the eigenvalues computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - containing a reference to the eigenvalues if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_eigenvalues(&self) -> Result<&Array1<f64>, ModelError> {
        match self.eigenvalues.as_ref() {
            Some(eigenvalues) => Ok(eigenvalues),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the eigenvectors computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - containing a reference to the eigenvectors if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_eigenvectors(&self) -> Result<&Array2<f64>, ModelError> {
        match self.eigenvectors.as_ref() {
            Some(eigenvectors) => Ok(eigenvectors),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the training data saved during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - containing a reference to the training data if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_x_fit(&self) -> Result<&Array2<f64>, ModelError> {
        match self.x_fit.as_ref() {
            Some(x_fit) => Ok(x_fit),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the row means computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - containing a reference to the row means if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_row_means(&self) -> Result<&Array1<f64>, ModelError> {
        match self.row_means.as_ref() {
            Some(row_means) => Ok(row_means),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the total mean computed during fitting.
    ///
    /// # Returns
    ///
    /// - `Ok(f64)` - containing the total mean if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - if the model hasn't been fitted
    pub fn get_total_mean(&self) -> Result<f64, ModelError> {
        match self.total_mean {
            Some(total_mean) => Ok(total_mean),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Fits the KernelPCA model to the input data.
    ///
    /// This method computes the kernel matrix, centers it, and performs eigendecomposition
    /// to extract the principal components.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - containing the transformed training data (projection of the input data)
    /// - `Err(Box<dyn std::error::Error>)` - if there are validation errors or computation errors
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input data is empty
    /// - n_components is 0
    /// - n_components is greater than the number of samples
    /// - Eigendecomposition fails
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self, Box<dyn std::error::Error>> {
        if x.is_empty() {
            return Err(Box::new(ModelError::InputValidationError(
                "Input data cannot be empty".to_string(),
            )));
        }

        if self.n_components <= 0 {
            return Err(Box::new(ModelError::InputValidationError(format!(
                "n_components={} must be greater than 0",
                self.n_components
            ))));
        }

        if x.nrows() < self.n_components {
            return Err(Box::new(ModelError::InputValidationError(format!(
                "n_components={} must be less than the number of samples={}",
                self.n_components,
                x.nrows()
            ))));
        }

        let n_samples = x.nrows();
        // Save a clone of the training data
        self.x_fit = Some(x.clone());

        // Calculate the kernel matrix: k_matrix[i, j] = kernel(x[i], x[j])
        let mut k_matrix = Array2::<f64>::zeros((n_samples, n_samples));

        // Use Rayon to compute kernel values for upper triangular matrix in parallel, avoiding direct modification of k_matrix in closure
        let kernel_vals: Vec<((usize, usize), f64)> = (0..n_samples)
            .into_par_iter()
            .flat_map(|i| {
                let mut local_results = Vec::new();
                for j in i..n_samples {
                    let k_val = compute_kernel(&x.row(i), &x.row(j), &self.kernel);
                    local_results.push(((i, j), k_val));
                }
                local_results
            })
            .collect();

        // Fill the kernel matrix
        for ((i, j), k_val) in kernel_vals {
            k_matrix[[i, j]] = k_val;
            k_matrix[[j, i]] = k_val;
        }

        // Calculate row means and total mean
        let row_means = k_matrix.mean_axis(Axis(1)).unwrap();
        let total_mean = k_matrix.mean().unwrap();
        self.row_means = Some(row_means.clone());
        self.total_mean = Some(total_mean);

        // Center the k_matrix: k_centered[i,j] = k_matrix[i,j] - row_means[i] - row_means[j] + total_mean
        let mut k_centered = k_matrix.clone();

        // Similarly avoid direct modification of k_centered in parallel closure
        let centered_vals: Vec<((usize, usize), f64)> = (0..n_samples)
            .into_par_iter()
            .flat_map(|i| {
                let mut local_results = Vec::new();
                for j in 0..n_samples {
                    let centered_val = k_matrix[[i, j]] - row_means[i] - row_means[j] + total_mean;
                    local_results.push(((i, j), centered_val));
                }
                local_results
            })
            .collect();

        // Fill the centered matrix
        for ((i, j), centered_val) in centered_vals {
            k_centered[[i, j]] = centered_val;
        }

        // Perform eigendecomposition on the centered kernel matrix (use eigh for symmetric matrices)
        let eig_result = k_centered.eigh(ndarray_linalg::UPLO::Lower)?;
        let eigenvalues_all = eig_result.0;
        let eigenvectors_all = eig_result.1; // Each column is an eigenvector

        // Collect and map eigenvalues and eigenvectors in parallel
        let mut eig_pairs: Vec<(f64, Array1<f64>)> = eigenvalues_all
            .iter()
            .cloned()
            .zip(eigenvectors_all.columns())
            .par_bridge() // Convert to parallel iterator
            .map(|(val, vec)| (val, vec.to_owned()))
            .collect();

        // Sorting is still sequential since parallel sorting might be unstable
        eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Select the top n_components eigenvalues and corresponding eigenvectors, and normalize them
        let mut selected_eigenvalues = Array1::<f64>::zeros(self.n_components);
        let mut selected_eigenvectors = Array2::<f64>::zeros((n_samples, self.n_components));

        // This part could also be parallelized, but since n_components is typically small, parallel benefit is limited
        for i in 0..self.n_components {
            selected_eigenvalues[i] = eig_pairs[i].0;
            // If the eigenvalue is large enough, normalize: normalized_vec = eigenvector / sqrt(eigenvalue)
            let norm_factor = if eig_pairs[i].0 > 1e-10 {
                eig_pairs[i].0.sqrt()
            } else {
                1.0
            };
            let normalized_vec = &eig_pairs[i].1 / norm_factor;
            selected_eigenvectors.column_mut(i).assign(&normalized_vec);
        }

        self.eigenvalues = Some(selected_eigenvalues);
        self.eigenvectors = Some(selected_eigenvectors);

        // Return the projection of training data in the new space (each row corresponds to n_components features of a sample)
        Ok(self)
    }

    /// Transforms new data using the fitted KernelPCA model.
    ///
    /// Projects new data points into the principal component space.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data matrix where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` containing the transformed data (projection of the input data)
    /// - `Err(Box<dyn std::error::Error>)` if the model hasn't been fitted or computation errors occur
    ///
    /// # Errors
    ///
    /// Returns an error if the model hasn't been fitted
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        // Model must be fitted first
        if self.x_fit.is_none()
            || self.eigenvectors.is_none()
            || self.row_means.is_none()
            || self.total_mean.is_none()
        {
            return Err(Box::new(ModelError::NotFitted));
        }
        let x_fit = self.x_fit.as_ref().unwrap();
        let n_train = x_fit.nrows();
        let n_new = x.nrows();
        let mut k_new = Array2::<f64>::zeros((n_train, n_new));

        // Calculate the kernel matrix between training data and new data in parallel
        let kernel_vals: Vec<((usize, usize), f64)> = (0..n_train)
            .into_par_iter()
            .flat_map(|i| {
                let mut local_results = Vec::new();
                for j in 0..n_new {
                    let k_val = compute_kernel(&x_fit.row(i), &x.row(j), &self.kernel);
                    local_results.push(((i, j), k_val));
                }
                local_results
            })
            .collect();

        // Fill the kernel matrix
        for ((i, j), k_val) in kernel_vals {
            k_new[[i, j]] = k_val;
        }

        // Center the new data kernel matrix: k_new_centered[i,j] = k_new[i,j] - row_means[i] - mean(k_new[:,j]) + total_mean
        let row_means = self.row_means.as_ref().unwrap();
        let total_mean = self.total_mean.unwrap();
        let mut k_new_centered = Array2::<f64>::zeros((n_train, n_new));

        // Precompute column means in parallel
        let column_means: Vec<f64> = (0..n_new)
            .into_par_iter()
            .map(|j| k_new.column(j).mean().unwrap())
            .collect();

        // Compute centered values in parallel
        let centered_vals: Vec<((usize, usize), f64)> = (0..n_new)
            .into_par_iter()
            .flat_map(|j| {
                let mean_new = column_means[j];
                let mut local_results = Vec::new();
                for i in 0..n_train {
                    let centered_val = k_new[[i, j]] - row_means[i] - mean_new + total_mean;
                    local_results.push(((i, j), centered_val));
                }
                local_results
            })
            .collect();

        // Fill the centered matrix
        for ((i, j), centered_val) in centered_vals {
            k_new_centered[[i, j]] = centered_val;
        }

        // Project using eigenvectors saved during training: projection = eigenvectors^T * k_new_centered
        let eigenvectors = self.eigenvectors.as_ref().unwrap(); // shape (n_train, n_components)
        let transformed = eigenvectors.t().dot(&k_new_centered);
        // The resulting shape is (n_components, n_new), transpose to return (n_new, n_components)
        Ok(transformed.t().to_owned())
    }

    /// Fits the model to the data and then transforms it.
    ///
    /// This is a convenience method that calls fit() followed by transform()
    /// on the same data.
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(Array2<f64>)` - containing the transformed data (projection of the input data)
    /// - `Err(Box<dyn std::error::Error>)` - if there are validation errors or computation errors
    ///
    /// # Errors
    ///
    /// Returns an error if fit() or transform() fails
    pub fn fit_transform(
        &mut self,
        x: &Array2<f64>,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        self.fit(x)?;
        match self.transform(x) {
            Ok(transformed) => Ok(transformed),
            Err(err) => Err(err),
        }
    }
}
