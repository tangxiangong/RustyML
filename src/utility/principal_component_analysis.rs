use crate::ModelError;
use ndarray::{Array1, Array2};
use ndarray_linalg::Eigh;
use rayon::prelude::*;
use std::error::Error;

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
            return Err(Box::new(ModelError::InputValidationError(
                "Number of components must be positive.".to_string(),
            )));
        }

        // Calculate mean using parallel iteration
        let mean: Array1<f64> = (0..n_features)
            .into_par_iter()
            .map(|i| x.column(i).mean().unwrap_or(0.0))
            .collect::<Vec<f64>>()
            .into();

        // Center the data in parallel
        let mut x_centered = x.clone();
        x_centered
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(_, mut row)| {
                for j in 0..n_features {
                    row[j] -= mean[j];
                }
            });

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

            let mut row = components.row_mut(i);
            for (j, &val) in eigvecs.column(idx).iter().enumerate() {
                row[j] = val;
            }
        }

        // Calculate explained variance ratio
        let total_var = eigvals.sum();
        let explained_variance_ratio = explained_variance.map(|&v| v / total_var);

        self.components = Some(components);
        self.mean = Some(mean);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.singular_values = Some(singular_values);

        Ok(self)
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

        // Use ndarray's vectorized operations for centering
        // This creates a view instead of cloning the entire array
        let x_centered = x
            .view()
            .outer_iter()
            .into_par_iter()
            .map(|row| {
                // Subtract the mean from each row
                let mut centered_row = row.to_owned();
                for (i, &m) in mean.iter().enumerate() {
                    centered_row[i] -= m;
                }
                centered_row
            })
            .collect::<Vec<_>>();

        // Convert the vector collection back to Array2
        let x_centered = Array2::from_shape_vec(
            (x.nrows(), x.ncols()),
            x_centered
                .into_iter()
                .flat_map(|row| row.into_iter().collect::<Vec<_>>())
                .collect(),
        )?;

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

        // Transform back to original feature space
        let mut x_restored = x.dot(components);
        let n_features = mean.len();

        x_restored
            .axis_chunks_iter_mut(ndarray::Axis(0), 1)
            .into_par_iter()
            .for_each(|mut chunk| {
                for j in 0..n_features {
                    chunk[[0, j]] += mean[j];
                }
            });

        Ok(x_restored)
    }
}
