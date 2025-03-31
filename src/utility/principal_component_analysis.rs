use ndarray::{Array1, Array2};
use std::error::Error;
use ndarray_linalg::Eigh;
use crate::ModelError;

/// # PCA structure for implementing Principal Component Analysis
///
/// This structure provides functionality for dimensionality reduction using PCA.
/// It allows fitting a model to data, transforming data into principal component space,
/// and retrieving various statistics about the decomposition.
///
/// ## Fields
///
/// * `n_components` - Number of principal components to keep in the model
/// * `components` - Principal axes in feature space, representing the directions of maximum variance Shape is (n_components, n_features)
/// * `mean` - Mean of each feature in the training data, used for centering
/// * `explained_variance` - Amount of variance explained by each component
/// * `explained_variance_ratio` - Percentage of variance explained by each component
/// * `singular_values` - Singular values corresponding to each component
///
/// ## Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use rustyml::utility::principal_component_analysis::PCA;
///
/// // Create some sample data (3 samples, 4 features)
/// let data = Array2::from_shape_vec((3, 4), vec![
///     1.0, 2.0, 3.0, 4.0,
///     2.0, 3.0, 4.0, 5.0,
///     3.0, 4.0, 5.0, 6.0
/// ]).unwrap();
///
/// // Create a PCA model with 2 components
/// let mut pca = PCA::new(2);
///
/// // Fit the model to the data
/// pca.fit(&data).expect("Failed to fit PCA model");
///
/// // Transform the data to the principal component space
/// let transformed = pca.transform(&data).expect("Failed to transform data");
/// println!("Transformed data:\n{:?}", transformed);
///
/// // Get the explained variance ratio
/// let variance_ratio = pca.get_explained_variance_ratio().expect("Model not fitted");
/// println!("Explained variance ratio: {:?}", variance_ratio);
///
/// // Transform back to original space
/// let reconstructed = pca.inverse_transform(&transformed).expect("Failed to inverse transform");
/// println!("Reconstructed data:\n{:?}", reconstructed);
/// ```
///
/// # Common use cases
///
/// * Dimensionality reduction: Reduce high-dimensional data to a lower-dimensional space
/// * Data visualization: Project data to 2 or 3 dimensions for visualization
/// * Feature extraction: Extract the most important features from the dataset
/// * Noise filtering: Remove noise by discarding components with low variance
#[derive(Debug, Clone)]
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
    singular_values: Option<Array1<f64>>,
}

impl Default for PCA {
    fn default() -> Self {
        // Default to 2 components which is common for visualization purposes
        Self {
            n_components: 2,
            components: None,
            mean: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
        }
    }
}

impl PCA {
    /// Creates a new PCA instance
    ///
    /// # Parameters
    ///
    /// * `n_components` - Number of principal components to keep
    ///
    /// # Returns
    ///
    /// * `Self` - A new PCA instance with the specified number of components
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            components: None,
            mean: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
        }
    }

    /// Fits the PCA model
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix, where rows are samples and columns are features
    ///
    /// # Returns
    ///
    /// - `Ok(&mut Self)` - The instance
    /// - `Err(Box<dyn std::error::Error>)` - If something goes wrong
    ///
    /// # Implementation Details
    ///
    /// - Computes the mean of each feature
    /// - Centers the data by subtracting the mean
    /// - Computes the covariance matrix
    /// - Calculates eigenvalues and eigenvectors
    /// - Sorts components by explained variance
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self, Box<dyn Error>> {
        use crate::machine_learning::preliminary_check;

        preliminary_check(&x, None)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if self.n_components <= 0 {
            return Err(Box::new(ModelError::InputValidationError("Number of components must be positive.".to_string())));
        }

        // Calculate mean
        let mut mean = Array1::<f64>::zeros(n_features);
        for i in 0..n_features {
            mean[i] = x.column(i).mean().unwrap_or(0.0);
        }

        // Center the data
        let mut x_centered = x.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                x_centered[[i, j]] -= mean[j];
            }
        }

        // Calculate covariance matrix
        let cov = x_centered.t().dot(&x_centered) / (n_samples as f64 - 1.0);

        // Calculate eigenvalues and eigenvectors
        let (eigvals, eigvecs) = cov.eigh(ndarray_linalg::UPLO::Upper)?;

        // Sort by eigenvalue in descending order
        let mut indices: Vec<usize> = (0..eigvals.len()).collect();
        indices.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap());

        // Get top n_components eigenvectors
        let n = self.n_components.min(n_features);
        let mut components = Array2::<f64>::zeros((n, n_features));
        let mut explained_variance = Array1::<f64>::zeros(n);
        let mut singular_values = Array1::<f64>::zeros(n);

        for i in 0..n {
            let idx = indices[i];
            explained_variance[i] = eigvals[idx];
            singular_values[i] = eigvals[idx].sqrt() * ((n_samples as f64 - 1.0).sqrt());
            components.row_mut(i).assign(&eigvecs.column(idx));
        }

        // Calculate explained variance ratio
        let total_var = eigvals.sum();
        let explained_variance_ratio = explained_variance.map(|&v| v / total_var);

        println!("Finish PCA");

        self.components = Some(components);
        self.mean = Some(mean);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.singular_values = Some(singular_values);

        Ok(self)
    }

    /// Gets the components matrix
    ///
    /// # Returns
    ///
    /// - `Ok(&Array2<f64>)` - The components matrix if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_components(&self) -> Result<&Array2<f64>, ModelError> {
        match self.components.as_ref() {
            Some(components) => Ok(components),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the explained variance
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The explained variance array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_explained_variance(&self) -> Result<&Array1<f64>, ModelError> {
        match self.explained_variance.as_ref() {
            Some(explained_variance) => Ok(explained_variance),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the explained variance ratio
    ///
    /// # Returns
    ///
    /// - `Ok(&Array1<f64>)` - The explained variance ratio array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_explained_variance_ratio(&self) -> Result<&Array1<f64>, ModelError> {
        match self.explained_variance_ratio.as_ref() {
            Some(explained_variance_ratio) => Ok(explained_variance_ratio),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Gets the singular values
    ///
    /// # Returns
    ///
    /// * `Ok(&Array1<f64>)` - The singular values array if fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_singular_values(&self) -> Result<&Array1<f64>, ModelError> {
        match self.singular_values.as_ref() {
            Some(singular_values) => Ok(singular_values),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Transforms data into principal component space
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix to transform
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, Box<dyn Error>>` - The transformed data if successful
    ///
    /// # Errors
    ///
    /// Returns an error if the model hasn't been fitted yet
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let components = self.components.as_ref().ok_or(ModelError::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(ModelError::NotFitted)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Center the data
        let mut x_centered = x.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                x_centered[[i, j]] -= mean[j];
            }
        }

        // Transform to principal component space
        let transformed = x_centered.dot(&components.t());

        Ok(transformed)
    }

    /// Fits the model and transforms the data
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, Box<dyn Error>>` - The transformed data if successful
    pub fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Transforms data from principal component space back to original feature space
    ///
    /// # Parameters
    ///
    /// * `x` - The input data matrix in principal component space
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f64>, Box<dyn Error>>` - The reconstructed data in original space
    ///
    /// # Errors
    ///
    /// Returns an error if the model hasn't been fitted yet
    pub fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
        let components = self.components.as_ref().ok_or("PCA model not fitted yet")?;
        let mean = self.mean.as_ref().ok_or("PCA model not fitted yet")?;

        let n_samples = x.nrows();

        // Transform back to original feature space
        let x_orig = x.dot(components);

        // Add mean back
        let mut x_restored = x_orig.clone();
        for i in 0..n_samples {
            for j in 0..mean.len() {
                x_restored[[i, j]] += mean[j];
            }
        }

        Ok(x_restored)
    }
}

/// Standardizes data to have zero mean and unit variance
///
/// This function transforms input data by subtracting the mean and dividing
/// by the standard deviation for each feature, resulting in standardized data
/// where each feature has a mean of 0 and a standard deviation of 1.
///
/// # Parameters
///
/// * `x` - A 2D array where rows represent samples and columns represent features
///
/// # Returns
///
/// * `Array2<f64>` - A standardized 2D array with the same shape as the input
///
/// # Implementation Details
///
/// - Calculates mean and standard deviation for each feature column
/// - Handles cases where standard deviation is zero (or very small) by setting it to 1.0
/// - Applies the z-score transformation: (x - mean) / std_dev
pub fn standardize(x: &Array2<f64>) -> Array2<f64> {
    use crate::math::standard_deviation;

    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Calculate mean for each column
    let mut means = Array1::<f64>::zeros(n_features);
    for i in 0..n_features {
        means[i] = x.column(i).mean().unwrap_or(0.0);
    }

    let mut stds = Array1::<f64>::zeros(n_features);
    for i in 0..n_features {
        let col_vec: Vec<f64> = x.column(i).iter().cloned().collect();
        stds[i] = standard_deviation(&col_vec);

        // Handle case where standard deviation is zero
        if stds[i] < 1e-10 {
            stds[i] = 1.0;
        }
    }

    // Standardize data
    let mut x_std = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x_std[[i, j]] = (x[[i, j]] - means[j]) / stds[j];
        }
    }

    x_std
}