use ndarray::{Array1, Array2};

/// # Principal Component Analysis (PCA)
///
/// This module provides an implementation of Principal Component Analysis (PCA),
/// a dimensionality reduction technique that transforms high-dimensional data
/// into a lower-dimensional space while preserving maximum variance.
///
/// PCA works by identifying the principal components - orthogonal axes that
/// capture the most variance in the data. These components are ordered by the
/// amount of variance they explain.
///
/// ## Features
///
/// - Dimensionality reduction with controllable number of components
/// - Computation of explained variance and explained variance ratio
/// - Transformation of data to and from the principal component space
/// - Access to components, singular values, and other PCA attributes
///
/// ## Example
///
/// ```rust
/// use ndarray::Array2;
/// use rustyml::utility::principal_component_analysis::PCA;
///
/// // Create a new PCA model with 2 components
/// let mut pca = PCA::new(2);
///
/// // Sample data matrix (5 samples, 3 features)
/// let data = Array2::from_shape_vec((5, 3), vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
///     10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0,
/// ]).unwrap();
///
/// // Fit the PCA model and transform the data
/// let transformed = pca.fit_transform(&data).unwrap();
///
/// // Get the explained variance ratio
/// let variance_ratio = pca.get_explained_variance_ratio().unwrap();
/// println!("Explained variance ratio: {:?}", variance_ratio);
/// ```
///
/// ## Mathematical Background
///
/// PCA computes the covariance matrix of the data, performs eigendecomposition,
/// and projects the data onto the eigenvectors (principal components) associated
/// with the largest eigenvalues.
pub mod principal_component_analysis;

/// # Train Test Split
///
/// This module provides functionality for splitting datasets into training and test sets,
/// which is a fundamental preprocessing step in machine learning workflows.
///
/// The train-test split allows for model evaluation on unseen data, helping to assess
/// how well a model generalizes beyond the training data. This implementation supports
/// random splitting with optional seed specification for reproducibility.
///
/// ## Features
///
/// - Split feature matrices and target vectors into training and test sets
/// - Configurable test set size (proportion of data allocated to testing)
/// - Optional random seed for reproducible splits
/// - Input validation to ensure data consistency
///
/// ## Example
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use rustyml::utility::train_test_split::train_test_split;
///
/// // Create sample data
/// let features = Array2::from_shape_vec((5, 2),
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
/// let targets = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
///
/// // Split data with 40% test size and fixed random seed
/// let (x_train, x_test, y_train, y_test) =
///     train_test_split(features, targets, Some(0.4), Some(42)).unwrap();
///
/// // Now we can use these sets for model training and evaluation
/// ```
///
/// ## Implementation Details
///
/// The splitting algorithm uses random shuffling of indices to ensure
/// unbiased dataset partitioning while maintaining the relationship between
/// features and their corresponding targets.
pub mod train_test_split;

/// # Kernel Principal Component Analysis (Kernel PCA)
///
/// This module implements Kernel Principal Component Analysis, a non-linear dimensionality
/// reduction technique that uses kernel methods to extend PCA to non-linear transformations.
///
/// Kernel PCA first maps the data into a higher dimensional feature space via a kernel function,
/// then performs standard PCA in this feature space. This allows capturing non-linear
/// relationships in the data that standard PCA cannot detect.
///
/// ## Features
///
/// - Support for various kernel functions (RBF, polynomial, linear, etc.)
/// - Configurable number of components to retain
/// - Methods for fitting models and transforming new data
/// - Access to eigenvalues and eigenvectors of the kernel matrix
///
/// ## Example Usage
///
/// ```
/// use rustyml::utility::kernel_pca::{KernelPCA, KernelType};
/// use ndarray::Array2;
///
/// // Create a new KernelPCA instance with RBF kernel and 2 components
/// let mut kpca = KernelPCA::new(KernelType::RBF{gamma: 1.0}, 2);
///
/// // Fit the model to your data and transform it
/// let data = Array2::from_shape_vec((5, 3), vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
///     10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0,
/// ]).unwrap();
/// let transformed = kpca.fit_transform(&data).unwrap();
/// ```
///
/// ## Mathematical Background
///
/// Kernel PCA works by:
/// 1. Computing a kernel matrix K using the chosen kernel function
/// 2. Centering the kernel matrix in the feature space
/// 3. Finding the eigenvalues and eigenvectors of the centered kernel matrix
/// 4. Projecting data onto the principal components in the feature space
pub mod kernel_pca;

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