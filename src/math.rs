use std::collections::HashMap;
use ndarray::{Array1, ArrayView2};
use crate::ModelError;

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
///
/// ```
/// use rustyml::math::sum_of_square_total;
///
/// let values = [1.0, 2.0, 3.0];
/// let sst = sum_of_square_total(&values);
/// // Mean is 2.0, so SST = (1-2)^2 + (2-2)^2 + (3-2)^2 = 1 + 0 + 1 = 2.0
/// assert!((sst - 2.0).abs() < 1e-5);
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
/// # Panics
/// The function will panic if the vectors have different lengths
///
/// # Examples
///
/// ```
/// use rustyml::math::sum_of_squared_errors;
///
/// let predicted = [2.0, 3.0];
/// let actual = [1.0, 3.0];
/// let sse = sum_of_squared_errors(&predicted, &actual);
/// // (2-1)^2 + (3-3)^2 = 1 + 0 = 1
/// assert!((sse - 1.0).abs() < 1e-6);
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
///
/// # Examples
///
/// ```
/// use rustyml::math::r2_score;
///
/// let predicted = [2.0, 3.0, 4.0];
/// let actual = [1.0, 3.0, 5.0];
/// let r2 = r2_score(&predicted, &actual);
/// // For actual values [1,3,5], mean=3, SSE = 1+0+1 = 2, SST = 4+0+4 = 8, so R2 = 1 - (2/8) = 0.75
/// assert!((r2 - 0.75).abs() < 1e-6);
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
/// # Mathematical properties
/// * sigmoid(0) = 0.5
/// * As z approaches positive infinity, sigmoid(z) approaches 1
/// * As z approaches negative infinity, sigmoid(z) approaches 0
///
/// # Examples
///
/// ```
/// use rustyml::math::sigmoid;
///
/// let value = sigmoid(0.0);
/// // sigmoid(0) = 0.5
/// assert!((value - 0.5).abs() < 1e-6);
/// ```
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Calculate logistic regression loss (log loss)
///
/// This function computes the average cross-entropy loss for logistic regression,
/// applying sigmoid to raw predictions before calculating loss.
///
/// # Parameters
/// * `logits` - Vector of raw model outputs (logits, before sigmoid)
/// * `actual_labels` - Vector of actual binary labels (0 or 1)
///
/// # Returns
/// * Average logistic regression loss
///
/// # Examples
///
/// ```
/// use rustyml::math::logistic_loss;
///
/// let logits = [0.0, 2.0, -1.0];
/// let actual_labels = [0.0, 1.0, 0.0];
/// let loss = logistic_loss(&logits, &actual_labels);
/// // Expected average loss is approximately 0.37778
/// assert!((loss - 0.37778).abs() < 1e-5);
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
/// # Parameters
/// * `predicted` - Array of predicted class labels
/// * `actual` - Array of actual class labels
///
/// # Returns
/// * The accuracy score between 0.0 and 1.0
///
/// # Panic
/// If the input arrays have different lengths
///
/// # Examples
///
/// ```
/// use rustyml::math::accuracy;
///
/// let predicted = [0.0, 1.0, 1.0];
/// let actual = [0.0, 0.0, 1.0];
/// let acc = accuracy(&predicted, &actual);
///
/// // Two out of three predictions are correct: accuracy = 2/3 ≈ 0.6666666666666667
/// assert!((acc - 0.6666666666666667).abs() < 1e-6);
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
/// # Parameters
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
///
/// # Returns
/// * The squared Euclidean distance as a f64 value
///
/// # Examples
///
/// ```
/// use rustyml::math::squared_euclidean_distance;
///
/// use ndarray::array;
/// let a = array![[1.0, 2.0]];
/// let b = array![[4.0, 6.0]];
/// let distance = squared_euclidean_distance(&a.view(), &b.view());
/// // (1-4)^2 + (2-6)^2 = 9 + 16 = 25
/// assert!((distance - 25.0).abs() < 1e-6);
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
/// # Parameters
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
///
/// # Returns
/// * The Manhattan distance as a f64 value
///
/// # Examples
///
/// ```
/// use rustyml::math::manhattan_distance;
///
/// use ndarray::array;
/// let a = array![[1.0, 2.0]];
/// let b = array![[4.0, 6.0]];
/// let distance = manhattan_distance(&a.view(), &b.view());
/// // |1-4| + |2-6| = 3 + 4 = 7
/// assert!((distance - 7.0).abs() < 1e-6);
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
/// # Parameters
/// * `x1` - First point as a 2D array view
/// * `x2` - Second point as a 2D array view
/// * `p` - The order of the norm (p ≥ 1.0)
///
/// # Returns
/// * The Minkowski distance as a f64 value
///
/// # Examples
///
/// ```
/// use rustyml::math::minkowski_distance;
///
/// use ndarray::array;
/// let a = array![[1.0, 2.0]];
/// let b = array![[4.0, 6.0]];
/// let distance = minkowski_distance(&a.view(), &b.view(), 3.0);
/// // Expected distance is approximately 4.497
/// assert!((distance - 4.497).abs() < 1e-3);
/// ```
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
/// * The Gaussian kernel value between the two points
///
/// # Examples
///
/// ```
/// use rustyml::math::gaussian_kernel;
///
/// use ndarray::array;
/// let a = array![[1.0, 2.0]];
/// let b = array![[1.0, 2.0]];
/// let gamma = 0.5;
/// let kernel = gaussian_kernel(&a.view(), &b.view(), gamma);
/// // For identical points, squared distance is 0, so kernel = exp(0) = 1.0
/// assert!((kernel - 1.0).abs() < 1e-6);
/// ```
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
/// * `y` - A slice of f64 values representing class labels
///
/// # Returns
/// * The entropy value of the given dataset. Lower values indicate more homogeneous data.
///
/// # Notes
/// * The function handles floating point labels by rounding to 3 decimal places for counting.
/// * Returns 0.0 for empty datasets.
///
/// # Examples
///
/// ```
/// use rustyml::math::entropy;
///
/// let labels = [0.0, 1.0, 1.0, 0.0];
/// let ent = entropy(&labels);
/// // For two classes with equal frequency, entropy = 1.0
/// assert!((ent - 1.0).abs() < 1e-6);
/// ```
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
/// * `y` - A slice of f64 values representing class labels
///
/// # Returns
/// * The Gini impurity value of the given dataset. The value ranges from 0.0 (pure) to 1.0 (impure).
///
/// # Notes
/// * The function handles floating point labels by rounding to 3 decimal places for counting.
/// * Returns 0.0 for empty datasets.
///
/// # Examples
///
/// ```
/// use rustyml::math::gini;
///
/// let labels = [0.0, 0.0, 1.0, 1.0];
/// let gini_val = gini(&labels);
/// // For two classes with equal frequency, Gini = 1 - (0.5^2 + 0.5^2) = 0.5
/// assert!((gini_val - 0.5).abs() < 1e-6);
/// ```
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
/// * `y` - A slice of f64 values representing class labels in the parent node
/// * `left_y` - A slice of f64 values representing class labels in the left child node
/// * `right_y` - A slice of f64 values representing class labels in the right child node
///
/// # Returns
/// * The information gain value. Higher values indicate a more useful split.
///
/// # Notes
/// * This function uses the entropy function to calculate impurity at each node.
///
/// # Examples
///
/// ```
/// use rustyml::math::information_gain;
///
/// let parent = [0.0, 0.0, 1.0, 1.0];
/// let left = [0.0, 0.0];
/// let right = [1.0, 1.0];
/// let ig = information_gain(&parent, &left, &right);
/// // Entropy(parent)=1.0, Entropy(left)=Entropy(right)=0, so IG = 1.0
/// assert!((ig - 1.0).abs() < 1e-6);
/// ```
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
/// * `y` - A slice of f64 values representing class labels in the parent node
/// * `left_y` - A slice of f64 values representing class labels in the left child node
/// * `right_y` - A slice of f64 values representing class labels in the right child node
///
/// # Returns
/// * The gain ratio value. Higher values indicate a more useful split.
///
/// # Notes
/// * Returns 0.0 if the split information is zero to avoid division by zero.
/// * This function uses the information_gain function as part of its calculation.
///
/// # Examples
///
/// ```
/// use rustyml::math::gain_ratio;
///
/// let parent = [0.0, 0.0, 1.0, 1.0];
/// let left = [0.0, 0.0];
/// let right = [1.0, 1.0];
/// let gr = gain_ratio(&parent, &left, &right);
/// // With equal splits, gain ratio should be 1.0
/// assert!((gr - 1.0).abs() < 1e-6);
/// ```
pub fn gain_ratio(y: &[f64], left_y: &[f64], right_y: &[f64]) -> f64 {
    if left_y.is_empty() || right_y.is_empty() {
        return 0.0;
    }

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
/// * `y` - A slice of f64 values for which to calculate the MSE
///
/// # Returns
/// * `f64` - The mean squared error as a f64 value, returns 0.0 if the input slice is empty
///
/// # Examples
///
/// ```
/// use rustyml::math::mean_squared_error;
///
/// let values = [1.0, 2.0, 3.0];
/// let mse = mean_squared_error(&values);
/// // Mean is 2.0, so MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 ≈ 0.66667
/// assert!((mse - 0.6666667).abs() < 1e-6);
/// ```
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

/// Calculates the leaf node adjustment factor c(n)
///
/// # Parameters
/// * `n` - Number of samples
///
/// # Returns
/// * `f64` - The adjustment factor
///
/// # Notes
/// Formula: c(n) = 2 * (H(n-1)) - (2*(n-1)/n)
/// where H(n-1) can be approximated by ln(n-1) + gamma, gamma is Euler's constant
///
/// # Examples
///
/// ```
/// use rustyml::math::average_path_length_factor;
///
/// let factor = average_path_length_factor(10.0);
/// // For n = 10, the factor is approximately 3.748
/// assert!((factor - 3.748).abs() < 0.01);
/// ```
pub fn average_path_length_factor(n: f64) -> f64 {
    if n <= 1.0 {
        0.0
    } else {
        let gamma = 0.57721566; // Euler's constant
        2.0 * ((n - 1.0).ln() + gamma) - 2.0 * (n - 1.0) / n
    }
}

/// Calculates the standard deviation of a set of values
///
/// # Parameters
/// * `values` - A slice of f64 values
///
/// # Returns
/// * `f64` - The standard deviation. Returns 0.0 if the slice is empty or contains only one element.
///
/// # Examples
///
/// ```
/// use rustyml::math::standard_deviation;
///
/// let values = [1.0, 2.0, 3.0];
/// let std_dev = standard_deviation(&values);
/// // Population standard deviation for [1,2,3] is approximately 0.8165
/// assert!((std_dev - 0.8165).abs() < 1e-4);
/// ```
pub fn standard_deviation(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }

    // Calculate the mean of the values
    let mean = values.iter().sum::<f64>() / n as f64;

    // Calculate the variance using n (population variance)
    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n as f64;

    // Return the population standard deviation
    variance.sqrt()
}

/// # Confusion Matrix for binary classification evaluation
///
/// A confusion matrix is a table that is often used to describe the performance of a classification model
/// on a set of test data for which the true values are known. It allows visualization of the performance
/// of an algorithm and identification of common types of errors.
///
/// ## Fields
///
/// * `tp` - True Positive: The number of correct positive predictions when the actual class is positive
/// * `fp` - False Positive: The number of incorrect positive predictions when the actual class is negative (Type I error)
/// * `tn` - True Negative: The number of correct negative predictions when the actual class is negative
/// * `fn_` - False Negative: The number of incorrect negative predictions when the actual class is positive (Type II error)
///
/// ## Performance Metrics
///
/// This implementation provides methods to calculate common performance metrics including:
/// accuracy, precision, recall, specificity, and F1 score.
///
/// ## Example
///
/// ```
/// use ndarray::arr1;
/// use rustyml::math::ConfusionMatrix;
///
/// // Create arrays for predicted and actual values
/// let predicted = arr1(&[0.9, 0.2, 0.8, 0.1, 0.7]);
/// let actual = arr1(&[1.0, 0.0, 1.0, 0.0, 1.0]);
///
/// // Create confusion matrix
/// let cm = ConfusionMatrix::new(&predicted, &actual).unwrap();
///
/// // Calculate performance metrics
/// println!("Accuracy: {:.2}", cm.accuracy());
/// println!("Precision: {:.2}", cm.precision());
/// println!("Recall: {:.2}", cm.recall());
/// println!("F1 Score: {:.2}", cm.f1_score());
///
/// // Get the confusion matrix components
/// let (tp, fp, tn, fn_) = cm.get_counts();
/// println!("TP: {}, FP: {}, TN: {}, FN: {}", tp, fp, tn, fn_);
///
/// // Print full summary
/// println!("{}", cm.summary());
/// ```
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// True Positive (TP): The number of correct positive predictions
    /// when the actual class is positive (correctly identified positive cases)
    tp: usize,

    /// False Positive (FP): The number of incorrect positive predictions
    /// when the actual class is negative (Type I error, false alarm)
    fp: usize,

    /// True Negative (TN): The number of correct negative predictions
    /// when the actual class is negative (correctly identified negative cases)
    tn: usize,

    /// False Negative (FN): The number of incorrect negative predictions
    /// when the actual class is positive (Type II error, miss)
    fn_: usize,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix
    ///
    /// # Parameters
    ///
    /// * `predicted` - Array of predicted labels, values >= 0.5 are considered positive class
    /// * `actual` - Array of actual labels, values >= 0.5 are considered positive class
    ///
    /// # Returns
    ///
    /// - `Ok(Self)` - A new confusion matrix if input arrays have the same length
    /// - `Err(ModelError::InputValidationError)` - Input does not match expectation
    pub fn new(predicted: &Array1<f64>, actual: &Array1<f64>) -> Result<Self, ModelError> {
        if predicted.len() != actual.len() {
            return Err(ModelError::InputValidationError(
                format!("The length of prediction and actual labels do not match, prediction length: {}, actual label length: {}",
                    predicted.len(), actual.len()
                )
            ));
        }

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for (p, a) in predicted.iter().zip(actual.iter()) {
            let p_binary = if *p >= 0.5 { 1.0 } else { 0.0 };
            let a_binary = if *a >= 0.5 { 1.0 } else { 0.0 };

            match (p_binary, a_binary) {
                (1.0, 1.0) => tp += 1,
                (1.0, 0.0) => fp += 1,
                (0.0, 1.0) => fn_ += 1,
                (0.0, 0.0) => tn += 1,
                _ => unreachable!(), // Should not happen as we explicitly convert to binary values
            }
        }

        Ok(Self { tp, fp, tn, fn_ })
    }

    /// Get the components of the confusion matrix
    ///
    /// # Returns
    ///
    /// A tuple containing the four basic components of the confusion matrix:
    /// * `tp` - True Positive count
    /// * `fp` - False Positive count
    /// * `tn` - True Negative count
    /// * `fn_` - False Negative count
    pub fn get_counts(&self) -> (usize, usize, usize, usize) {
        (self.tp, self.fp, self.tn, self.fn_)
    }

    /// Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    ///
    /// Accuracy measures the proportion of correct predictions among the total number of cases examined.
    ///
    /// # Returns
    ///
    /// * `f64` - A float value between 0.0 and 1.0 representing the accuracy. Returns 0.0 if there are no predictions (empty matrix).
    pub fn accuracy(&self) -> f64 {
        let total = self.tp + self.tn + self.fp + self.fn_;
        if total == 0 {
            return 0.0;
        }
        (self.tp + self.tn) as f64 / total as f64
    }

    /// Calculate error rate: (FP + FN) / (TP + TN + FP + FN) = 1 - Accuracy
    ///
    /// Error rate measures the proportion of incorrect predictions among the total number of cases examined.
    ///
    /// # Returns
    ///
    /// * `f64` - A float value between 0.0 and 1.0 representing the error rate. Returns 1.0 if there are no predictions (empty matrix).
    pub fn error_rate(&self) -> f64 {
        1.0 - self.accuracy()
    }

    /// Calculate precision: TP / (TP + FP)
    ///
    /// Precision measures the proportion of positive identifications that were actually correct.
    /// It answers the question: "Of all the instances predicted as positive, how many were actually positive?"
    ///
    /// # Returns
    ///
    /// * `f64` - A float value between 0.0 and 1.0 representing precision. Returns 0.0 if there are no positive predictions (TP + FP = 0).
    pub fn precision(&self) -> f64 {
        if self.tp + self.fp == 0 {
            return 0.0;
        }
        self.tp as f64 / (self.tp + self.fp) as f64
    }

    /// Calculate recall (sensitivity): TP / (TP + FN)
    ///
    /// Recall measures the proportion of actual positives that were correctly identified.
    /// It answers the question: "Of all the actual positive instances, how many were correctly predicted?"
    ///
    /// # Returns
    ///
    /// * `f64` - A float value between 0.0 and 1.0 representing recall. Returns 1.0 if there are no actual positive instances (TP + FN = 0).
    pub fn recall(&self) -> f64 {
        if self.tp + self.fn_ == 0 {
            return 1.0;
        }
        self.tp as f64 / (self.tp + self.fn_) as f64
    }

    /// Calculate specificity: TN / (TN + FP)
    ///
    /// Specificity measures the proportion of actual negatives that were correctly identified.
    /// It answers the question: "Of all the actual negative instances, how many were correctly predicted?"
    ///
    /// # Returns
    ///
    /// * `f64` - A float value between 0.0 and 1.0 representing specificity. Returns 1.0 if there are no actual negative instances (TN + FP = 0).
    pub fn specificity(&self) -> f64 {
        if self.tn + self.fp == 0 {
            return 1.0;
        }
        self.tn as f64 / (self.tn + self.fp) as f64
    }

    /// Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)
    ///
    /// F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics.
    /// It's particularly useful when you want to balance precision and recall and there's an uneven class distribution.
    ///
    /// # Returns
    ///
    /// * `f64` - A float value between 0.0 and 1.0 representing the F1 score. Returns 0.0 if either precision or recall is 0.0.
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall == 0.0 {
            return 0.0;
        }

        2.0 * (precision * recall) / (precision + recall)
    }

    /// Generate a formatted summary of the confusion matrix and all performance metrics
    ///
    /// # Returns
    ///
    /// * `String` - A formatted string containing a visual representation of the confusion matrix and all calculated performance metrics with 4 decimal places of precision.
    pub fn summary(&self) -> String {
        format!(
            "Confusion Matrix:\n\
        |                | Predicted Positive | Predicted Negative |\n\
        |----------------|-------------------|--------------------|\n\
        | Actual Positive | TP: {}           | FN: {}             |\n\
        | Actual Negative | FP: {}           | TN: {}             |\n\
        \n\
        Performance Metrics:\n\
        - Accuracy: {:.4}\n\
        - Error Rate: {:.4}\n\
        - Precision: {:.4}\n\
        - Recall: {:.4}\n\
        - Specificity: {:.4}\n\
        - F1 Score: {:.4}",
            self.tp, self.fn_, self.fp, self.tn,
            self.accuracy(), self.error_rate(),
            self.precision(), self.recall(),
            self.specificity(), self.f1_score()
        )
    }
}