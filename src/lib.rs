/// Error types that can occur during model operations
#[derive(Debug)]
pub enum ModelError {
    /// Indicates that the model has not been fitted yet
    NotFitted,
    /// indicates the input data provided does not meet the expected format, type, or validation rules.
    InputValidationError(String),
    /// indicates that there is something wrong with the tree
    TreeError(&'static str),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::NotFitted => write!(f, "Model has not been fitted. Parameters are unavailable."),
            ModelError::InputValidationError(msg) => write!(f, "Input validation error: {}", msg),
            ModelError::TreeError(msg) => write!(f, "Tree structure error: {}", msg),
        }
    }
}

/// Implements the standard error trait for ModelError
impl std::error::Error for ModelError {}

/// # Module `math` contains mathematical utility functions for statistical operations and model evaluation.
///
/// # Included formula
///
/// This module provides implementations of common statistical measures used in machine learning:
/// - Sum of square total (SST) for measuring data variability
/// - Sum of squared errors (SSE) for evaluating prediction errors
/// - R-squared (R²) score for assessing model fit quality
/// - Sigmoid function for logistic regression and neural networks
/// - Logistic loss (log loss) for binary classification models
/// - Accuracy score for classification model evaluation
/// - Calculate the squared Euclidean distance between two points
/// - Calculate the Manhattan distance between two points
/// - Calculate the Minkowski distance between two points
/// - Calculate the Gaussian kernel (RBF kernel)
/// - Calculates the entropy of a label set
/// - Calculates the Gini impurity of a label set
/// - Calculates the information gain when splitting a dataset
/// - Calculates the gain ratio for a dataset split
/// - Calculates the Mean Squared Error (MSE) of a set of values
/// - Calculates the leaf node adjustment factor c(n)
/// - Calculates the standard deviation of a set of values
/// 
/// These functions are particularly useful for regression model evaluation and
/// performance assessment in machine learning applications.
///
/// # Examples
///
/// ```
/// use rustyml::math::{sum_of_squared_errors, r2_score};
///
/// // Example data
/// let predicted = vec![2.1, 3.8, 5.2, 7.1];
/// let actual = vec![2.0, 4.0, 5.0, 7.0];
///
/// // Calculate error metrics
/// let sse = sum_of_squared_errors(&predicted, &actual);
/// let r2 = r2_score(&predicted, &actual);
/// ```
pub mod math;

#[cfg(test)]
mod math_module_test;

/// Module `machine_learning` provides implementations of various machine learning algorithms and models.
///
/// This module includes a collection of supervised and unsupervised learning algorithms
/// that can be used for tasks such as classification, regression, and clustering:
///
/// # Supervised Learning Algorithms
///
/// ## Classification
/// - **LogisticRegression**: Binary classification using logistic regression with gradient descent optimization
/// - **KNN**: K-Nearest Neighbors classifier with customizable distance metrics and weighting strategies
/// - **DecisionTree**: Decision tree classifier with various splitting criteria and pruning options
///
/// ## Regression
/// - **LinearRegression**: Simple and multivariate linear regression with optional intercept fitting
///
/// # Unsupervised Learning Algorithms
///
/// ## Clustering
/// - **KMeans**: K-means clustering with customizable initialization and convergence criteria
/// - **DBSCAN**: Density-based spatial clustering of applications with noise
/// - **MeanShift**: Non-parametric clustering that finds clusters by identifying density modes
///
/// ## Dimensionality Reduction
/// - **PCA**: Principal Component Analysis for linear dimensionality reduction
///
/// ## Anomaly Detection
/// - **IsolationForest**: Isolation Forest for identifying anomalies and outliers in the data
///
/// # Utility Functions
/// - **estimate_bandwidth**: Estimates the bandwidth parameter for MeanShift clustering
/// - **generate_polynomial_features**: Creates polynomial features for enhancing model complexity
/// - **standardize**: Standardizes features by removing the mean and scaling to unit variance
///
/// # Examples
///
/// ```
/// use rustyml::machine_learning::linear_regression::LinearRegression;
/// use ndarray::{Array1, Array2, array};
///
/// // Create a linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6);
///
/// // Prepare training data
/// let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
/// let raw_y = vec![6.0, 9.0, 12.0];
///
/// // Convert Vec to ndarray types
/// let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
/// let y = Array1::from_vec(raw_y);
///
/// // Train the model
/// model.fit(&x, &y).unwrap();
///
/// // Make predictions
/// let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
/// let predictions = model.predict(&new_data);
///
/// // Since Clone is implemented, the model can be easily cloned
/// let model_copy = model.clone();
///
/// // Since Debug is implemented, detailed model information can be printed
/// println!("{:?}", model);
/// ```
pub mod machine_learning;

#[cfg(test)]
mod machine_learning_test;

/// A convenience module that re-exports the most commonly used types and traits from this crate.
///
/// This module provides a single import point for frequently used items from this library's machine learning modules,
/// allowing users to import multiple items with a single `use` statement.
///
/// # Examples
///
/// ```
/// // Import all common items
/// use rustyml::prelude::*;
///
/// // Now you can use items like DBSCAN, KMeans, DecisionTree, etc. directly
/// ```
///
/// # Available Components
///
/// This prelude exports the following machine learning algorithms and utilities:
/// - Clustering: `DBSCAN`, `KMeans`, `MeanShift`
/// - Classification: `KNN`, `DecisionTree`, `LogisticRegression`
/// - Regression: `LinearRegression`
/// - Dimensionality Reduction: `PCA`
/// - Anomaly Detection: `IsolationForest`
/// - Common parameters and metrics: `DistanceCalculationMetric`, `WeightingStrategy`, etc.
pub mod prelude;