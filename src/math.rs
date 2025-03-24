use ndarray::ArrayView2;

/// Calculates the Sum of Square Total
///
/// SST measures the total variability in the data, computed as the sum of squared
/// differences between each actual value and the mean of all values.
///
/// # Parameters
/// * `values` - A slice of observed values
///
/// # Returns
/// * The Sum of Square Total (SST)
///
/// # Examples
/// ```
///  use rust_ai::math::sum_of_square_total;
/// let values = vec![2.0, 4.0, 6.0, 8.0];
/// let sst = sum_of_square_total(&values);
/// ```
pub fn sum_of_square_total(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    // Calculate the mean
    let mean = values.iter().sum::<f64>() / values.len() as f64;

    // Calculate sum of squared differences from the mean
    values.iter()
        .map(|&value| (value - mean).powi(2))
        .sum()
}

/// Calculate the sum of squared errors
///
/// # Parameters
/// * `predicted` - Predicted values vector (y')
/// * `actual` - Actual values vector (y)
///
/// # Returns
/// * Sum of squared errors sum((predicted_i - actual_i)^2)
///
/// # Panic
/// The function will panic if the vectors have different lengths
/// # Example
/// ```
/// use rust_ai::math::sum_of_squared_errors;
///
/// let predicted = vec![1.0, 2.0, 3.0, 4.0];
/// let actual = vec![1.2, 1.8, 3.3, 3.9];
///
/// // Calculate the sum of squared errors
/// let sse = sum_of_squared_errors(&predicted, &actual);
/// ```

pub fn sum_of_squared_errors(predicted: &[f64], actual: &[f64]) -> f64 {
    // Ensure both vectors have the same length
    assert_eq!(predicted.len(), actual.len(), "Vectors must have the same length");
    // Calculate the sum of squared errors
    predicted.iter()
        .zip(actual.iter())
        .map(|(p, a)| {
            let diff = p - a;
            diff * diff // Calculate square
        })
        .sum()
}

/// Calculate the R-squared score
///
/// R² = 1 - (SSE / SST)
///
/// # Arguments
/// * `predicted` - Array of predicted values
/// * `actual` - Array of actual values
///
/// # Returns
/// * `f64` - R-squared value, typically ranges from 0 to 1
///
/// # Notes
/// - Returns 0 if SST is 0 (when all actual values are identical)
/// - R-squared can theoretically be negative, indicating that the model performs worse than simply predicting the mean
/// - A value close to 1 indicates a good fit
/// # Example
/// ```
/// use rust_ai::math::{r2_score, sum_of_squared_errors, sum_of_square_total};
///
/// let actual = vec![3.0, 2.0, 5.0, 7.0, 9.0];
/// let predicted = vec![2.8, 1.9, 5.2, 7.5, 8.9];
///
/// // Calculate R-squared
/// let r2 = r2_score(&predicted, &actual);
/// ```

pub fn r2_score(predicted: &[f64], actual: &[f64]) -> f64 {
    let sse = sum_of_squared_errors(predicted, actual);
    let sst = sum_of_square_total(actual);

    // Prevent division by zero (when all actual values are identical)
    if sst == 0.0 {
        return 0.0;
    }

    1.0 - (sse / sst)
}

/// Calculate the sigmoid function value for a given input
///
/// The sigmoid function transforms any real-valued number into the range (0, 1).
/// It is defined as: σ(z) = 1 / (1 + e^(-z))
///
/// # Parameters
/// * `z` - The input value (can be any real number)
///
/// # Returns
/// * A value between 0 and 1, representing the sigmoid of the input
///
/// # Examples
/// ```
/// use rust_ai::math::sigmoid;
///
/// assert_eq!(sigmoid(0.0), 0.5);
/// assert!(sigmoid(10.0) > 0.999);
/// assert!(sigmoid(-10.0) < 0.001);
/// ```
///
/// # Mathematical properties
/// * sigmoid(0) = 0.5
/// * As z approaches positive infinity, sigmoid(z) approaches 1
/// * As z approaches negative infinity, sigmoid(z) approaches 0
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Calculate logistic regression loss (log loss)
///
/// This function computes the average cross-entropy loss for logistic regression,
/// applying sigmoid to raw predictions before calculating loss.
///
/// # Parameters
/// * `raw_predictions` - Vector of raw model outputs (logits, before sigmoid)
/// * `actual_labels` - Vector of actual binary labels (0 or 1)
///
/// # Returns
/// * Average logistic regression loss
///
/// # Example
/// ```
/// use rust_ai::math::logistic_loss;
///
/// let raw_predictions = vec![1.2, -0.5, 2.1, -1.8];
/// let actual_labels = vec![1.0, 0.0, 1.0, 0.0];
///
/// let loss = logistic_loss(&raw_predictions, &actual_labels);
/// ```
pub fn logistic_loss(logits: &[f64], actual_labels: &[f64]) -> f64 {
    assert_eq!(logits.len(), actual_labels.len(), 
               "Raw predictions and actual labels must have the same length");

    // Validate that all labels are either 0 or 1
    for (i, &label) in actual_labels.iter().enumerate() {
        assert!(label == 0.0 || label == 1.0,
                "Label at index {} must be either 0 or 1, found: {}", i, label);
    }

    let mut total_loss = 0.0;
    for (&x, &y) in logits.iter().zip(actual_labels.iter()) {
        let loss = x.max(0.0) - x * y + (1.0 + (-x.abs()).exp()).ln();
        total_loss += loss;
    }
    total_loss / logits.len() as f64
}

/// Calculate the accuracy of a classification model
///
/// Accuracy is defined as the proportion of correctly predicted samples
/// over the total number of samples.
/// For binary classification, predicted and actual values should be 0.0 or 1.0.
/// For multi-class classification, values should be numeric class labels.
///
/// # Arguments
///
/// * `predicted` - Array of predicted class labels
/// * `actual` - Array of actual class labels
///
/// # Returns
///
/// The accuracy score between 0.0 and 1.0
///
/// # Panics
///
/// Panics if the input arrays have different lengths
///
/// # Examples
///
/// ```
/// use rust_ai::math::accuracy;
///
/// // Binary classification example
/// let predicted = vec![1.0, 0.0, 1.0, 1.0, 0.0];
/// let actual = vec![1.0, 0.0, 0.0, 1.0, 0.0];
///
/// let acc = accuracy(&predicted, &actual);
/// assert_eq!(acc, 0.8); // 4 out of 5 predictions are correct
/// ```
pub fn accuracy(predicted: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(
        predicted.len(),
        actual.len(),
        "Predicted and actual arrays must have the same length"
    );

    if predicted.is_empty() {
        return 0.0;
    }

    let correct_predictions = predicted
        .iter()
        .zip(actual.iter())
        .filter(|&(p, a)| (p - a).abs() < f64::EPSILON)
        .count();

    correct_predictions as f64 / predicted.len() as f64
}

/// Calculate the squared Euclidean distance between two points
///
/// Computes the sum of squared differences between corresponding elements
/// of two 2D arrays, representing the squared Euclidean distance.
///
/// # Arguments
///
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
///
/// # Returns
///
/// The squared Euclidean distance as a f64 value
///
/// # Example
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use rust_ai::math::squared_euclidean_distance;
///
/// let a = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
/// let b = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();
///
/// let dist = squared_euclidean_distance(&a.view(), &b.view());
/// assert_eq!(dist, 27.0); // (4-1)² + (5-2)² + (6-3)² = 9 + 9 + 9 = 27
/// ```
pub fn squared_euclidean_distance(x1: &ArrayView2<f64>, x2: &ArrayView2<f64>) -> f64 {
    let diff = x1 - x2;
    diff.iter().map(|&x| x * x).sum()
}

/// Calculate the Manhattan distance between two points
///
/// Computes the sum of absolute differences between corresponding elements
/// of two 2D arrays, representing the Manhattan (L1) distance.
///
/// # Arguments
///
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
///
/// # Returns
///
/// The Manhattan distance as a f64 value
///
/// # Example
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use rust_ai::math::manhattan_distance;
///
/// let a = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
/// let b = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();
///
/// let dist = manhattan_distance(&a.view(), &b.view());
/// assert_eq!(dist, 9.0); // |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
/// ```
pub fn manhattan_distance(x1: &ArrayView2<f64>, x2: &ArrayView2<f64>) -> f64 {
    let diff = x1 - x2;
    diff.iter().map(|&x| x.abs()).sum()
}

/// Calculate the Minkowski distance between two points
///
/// Computes the p-norm between corresponding elements of two 2D arrays,
/// representing the Minkowski distance with parameter p.
///
/// # Arguments
///
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
/// * `p` - The order of the norm (p ≥ 1.0)
///
/// # Returns
///
/// The Minkowski distance as a f64 value
///
/// # Example
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use rust_ai::math::minkowski_distance;
///
/// let a = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
/// let b = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();
///
/// // p=1 is Manhattan distance
/// let dist_p1 = minkowski_distance(&a.view(), &b.view(), 1.0);
/// assert!((dist_p1 - 9.0).abs() < 1e-10);
///
/// // p=2 is Euclidean distance
/// let dist_p2 = minkowski_distance(&a.view(), &b.view(), 2.0);
/// assert!((dist_p2 - 5.196152).abs() < 1e-6); // sqrt((4-1)² + (5-2)² + (6-3)²) = sqrt(27) ≈ 5.196
/// ```
pub fn minkowski_distance(x1: &ArrayView2<f64>, x2: &ArrayView2<f64>, p: f64) -> f64 {
    assert!(p >= 1.0, "p must be greater than or equal to 1.0");

    let diff = x1 - x2;
    let sum: f64 = diff.iter().map(|&x| x.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}