/// Error types that can occur during model operations
#[derive(Debug)]
pub enum ModelError {
    /// Indicates that the model has not been fitted yet
    NotFitted,
    /// indicates the input data provided  does not meet the expected format, type, or validation rules.
    InputValidationError(&'static str),
    /// indicates that there is something wrong with decision tree
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

/// Module `math` contains mathematical utility functions for statistical operations and model evaluation.
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
/// 
/// These functions are particularly useful for regression model evaluation and
/// performance assessment in machine learning applications.
///
/// # Examples
///
/// ```
/// use rust_ai::math::{sum_of_squared_errors, r2_score};
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

pub mod machine_learning;

#[cfg(test)]
mod machine_learning_test;