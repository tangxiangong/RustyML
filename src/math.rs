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