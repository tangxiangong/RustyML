use std::collections::HashMap;
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
/// # Parameters
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
/// # Parameters
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
/// # Parameters
///
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
///
/// # Returns
///
/// The squared Euclidean distance as a f64 value
pub fn squared_euclidean_distance(x1: &ArrayView2<f64>, x2: &ArrayView2<f64>) -> f64 {
    let diff = x1 - x2;
    diff.iter().map(|&x| x * x).sum()
}

/// Calculate the Manhattan distance between two points
///
/// Computes the sum of absolute differences between corresponding elements
/// of two 2D arrays, representing the Manhattan (L1) distance.
///
/// # Parameters
///
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
///
/// # Returns
///
/// The Manhattan distance as a f64 value
pub fn manhattan_distance(x1: &ArrayView2<f64>, x2: &ArrayView2<f64>) -> f64 {
    let diff = x1 - x2;
    diff.iter().map(|&x| x.abs()).sum()
}

/// Calculate the Minkowski distance between two points
///
/// Computes the p-norm between corresponding elements of two 2D arrays,
/// representing the Minkowski distance with parameter p.
///
/// # Parameters
///
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
/// * `p` - The order of the norm (p ≥ 1.0)
///
/// # Returns
///
/// The Minkowski distance as a f64 value
pub fn minkowski_distance(x1: &ArrayView2<f64>, x2: &ArrayView2<f64>, p: f64) -> f64 {
    assert!(p >= 1.0, "p must be greater than or equal to 1.0");

    let diff = x1 - x2;
    let sum: f64 = diff.iter().map(|&x| x.abs().powf(p)).sum();
    sum.powf(1.0 / p)
}

/// Calculate the Gaussian kernel (RBF kernel)
///
/// # Parameters
/// * `x1` - View of the first input vector/matrix
/// * `x2` - View of the second input vector/matrix
/// * `gamma` - Kernel width parameter, controls the "width" of the Gaussian function
///
/// # Returns
/// The Gaussian kernel value between the two points
pub fn gaussian_kernel(x1: &ArrayView2<f64>, x2: &ArrayView2<f64>, gamma: f64) -> f64 {
    // Calculate the squared Euclidean distance between the two points
    let squared_distance = squared_euclidean_distance(x1, x2);

    // Apply the Gaussian formula: K(x, y) = exp(-gamma * ||x - y||²)
    (-gamma * squared_distance).exp()
}

/// Calculates the entropy of a label set.
///
/// Entropy is a measure of impurity in a dataset, commonly used in decision tree algorithms
/// like ID3 and C4.5. It quantifies the uncertainty or randomness in the data distribution.
///
/// # Parameters
///
/// * `y` - A slice of f64 values representing class labels
///
/// # Returns
///
/// * The entropy value of the given dataset. Lower values indicate more homogeneous data.
///
/// # Notes
///
/// * The function handles floating point labels by rounding to 3 decimal places for counting.
/// * Returns 0.0 for empty datasets.
pub fn entropy(y: &[f64]) -> f64 {
    let total_samples = y.len() as f64;
    if total_samples == 0.0 {
        return 0.0;
    }

    // Count samples for each class
    let mut class_counts = HashMap::new();
    for &value in y {
        // Convert float to integer representation
        let key = (value * 1000.0).round() as i64; // Preserve 3 decimal places precision
        *class_counts.entry(key).or_insert(0) += 1;
    }

    // Calculate entropy
    -class_counts.values()
        .map(|&count| {
            let p = count as f64 / total_samples;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Calculates the Gini impurity of a label set.
///
/// Gini impurity is a measure of how often a randomly chosen element from the set would be
/// incorrectly labeled if it was randomly labeled according to the distribution of labels
/// in the subset. It is commonly used in decision tree algorithms like CART.
///
/// # Parameters
///
/// * `y` - A slice of f64 values representing class labels
///
/// # Returns
///
/// * The Gini impurity value of the given dataset. The value ranges from 0.0 (pure) to 1.0 (impure).
///
/// # Notes
///
/// * The function handles floating point labels by rounding to 3 decimal places for counting.
/// * Returns 0.0 for empty datasets.
pub fn gini(y: &[f64]) -> f64 {
    let total_samples = y.len() as f64;
    if total_samples == 0.0 {
        return 0.0;
    }

    // Count samples for each class
    let mut class_counts = HashMap::new();
    for &value in y {
        // Convert float to integer representation
        let key = (value * 1000.0).round() as i64; // Preserve 3 decimal places precision
        *class_counts.entry(key).or_insert(0) += 1;
    }

    // Calculate Gini impurity
    1.0 - class_counts.values()
        .map(|&count| {
            let p = count as f64 / total_samples;
            p * p
        })
        .sum::<f64>()
}

/// Calculates the information gain when splitting a dataset.
///
/// Information gain measures the reduction in entropy (or impurity) achieved by
/// splitting the dataset into two parts. It is commonly used in decision tree algorithms
/// like ID3 and C4.5 to select the best feature for splitting.
///
/// # Parameters
///
/// * `y` - A slice of f64 values representing class labels in the parent node
/// * `left_y` - A slice of f64 values representing class labels in the left child node
/// * `right_y` - A slice of f64 values representing class labels in the right child node
///
/// # Returns
///
/// * The information gain value. Higher values indicate a more useful split.
///
/// # Notes
///
/// * This function uses the entropy function to calculate impurity at each node.
pub fn information_gain(y: &[f64], left_y: &[f64], right_y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let n_left = left_y.len() as f64;
    let n_right = right_y.len() as f64;

    let e = entropy(y);
    let e_left = entropy(left_y);
    let e_right = entropy(right_y);

    // Information gain = parent entropy - weighted sum of child entropies
    e - (n_left / n) * e_left - (n_right / n) * e_right
}

/// Calculates the gain ratio for a dataset split.
///
/// Gain ratio is an extension of information gain that reduces the bias towards
/// features with a large number of values. It is used in the C4.5 algorithm to
/// normalize information gain by the entropy of the split itself.
///
/// # Parameters
///
/// * `y` - A slice of f64 values representing class labels in the parent node
/// * `left_y` - A slice of f64 values representing class labels in the left child node
/// * `right_y` - A slice of f64 values representing class labels in the right child node
///
/// # Returns
///
/// * The gain ratio value. Higher values indicate a more useful split.
///
/// # Notes
///
/// * Returns 0.0 if the split information is zero to avoid division by zero.
/// * This function uses the information_gain function as part of its calculation.
pub fn gain_ratio(y: &[f64], left_y: &[f64], right_y: &[f64]) -> f64 {
    let info_gain = information_gain(y, left_y, right_y);
    let n = y.len() as f64;
    let n_left = left_y.len() as f64;
    let n_right = right_y.len() as f64;

    // Split information
    let split_info = -(n_left / n) * (n_left / n).log2() - (n_right / n) * (n_right / n).log2();

    if split_info == 0.0 {
        0.0
    } else {
        info_gain / split_info
    }
}

/// Calculates the Mean Squared Error (MSE) of a set of values.
///
/// The MSE is calculated as the average of the squared differences between each value
/// and the mean of all values. It represents the variance of the dataset.
///
/// # Arguments
///
/// * `y` - A slice of f64 values for which to calculate the MSE
///
/// # Returns
///
/// * `f64` - The mean squared error as an f64 value, returns 0.0 if the input slice is empty
pub fn mean_squared_error(y: &[f64]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }

    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let variance = y.iter()
        .map(|&val| (val - mean).powi(2))
        .sum::<f64>() / y.len() as f64;

    variance
}