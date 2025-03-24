use crate::math::*;
use std::f64::EPSILON;

#[test]
fn test_sum_of_square_total() {
    // Standard case
    let values = vec![2.0, 4.0, 6.0, 8.0];
    // Mean is 5.0, so SST = (2-5)² + (4-5)² + (6-5)² + (8-5)² = 9 + 1 + 1 + 9 = 20
    let expected = 20.0;
    assert!((sum_of_square_total(&values) - expected).abs() < EPSILON);

    // Case where all values are identical
    let same_values = vec![5.0, 5.0, 5.0, 5.0];
    // Mean is 5.0, so SST = 0
    assert!((sum_of_square_total(&same_values) - 0.0).abs() < EPSILON);

    // Empty array case
    let empty = Vec::<f64>::new();
    assert_eq!(sum_of_square_total(&empty), 0.0);

    // Single value case
    let single = vec![10.0];
    assert!((sum_of_square_total(&single) - 0.0).abs() < EPSILON);
}

#[test]
fn test_sum_of_squared_errors() {
    // Standard case
    let predicted = vec![1.0, 2.0, 3.0, 4.0];
    let actual = vec![1.2, 1.8, 3.3, 3.9];
    // SSE = (1-1.2)² + (2-1.8)² + (3-3.3)² + (4-3.9)²
    // SSE = 0.04 + 0.04 + 0.09 + 0.01 = 0.18
    let expected = 0.18;
    assert!((sum_of_squared_errors(&predicted, &actual) - expected).abs() < 0.0001);

    // Case with perfect prediction
    let perfect_prediction = vec![5.0, 10.0, 15.0];
    let actual_values = vec![5.0, 10.0, 15.0];
    assert!((sum_of_squared_errors(&perfect_prediction, &actual_values) - 0.0).abs() < EPSILON);

    // Empty arrays case
    let empty_pred = Vec::<f64>::new();
    let empty_actual = Vec::<f64>::new();
    assert!((sum_of_squared_errors(&empty_pred, &empty_actual) - 0.0).abs() < EPSILON);
}

#[test]
#[should_panic(expected = "Vectors must have the same length")]
fn test_sum_of_squared_errors_different_lengths() {
    // Different length arrays should panic
    let predicted = vec![1.0, 2.0, 3.0];
    let actual = vec![1.2, 1.8];
    sum_of_squared_errors(&predicted, &actual);
}

#[test]
fn test_r2_score() {
    // Standard case
    let actual = vec![3.0, 2.0, 5.0, 7.0, 9.0];
    let predicted = vec![2.8, 1.9, 5.2, 7.5, 8.9];
    // Should be close to 1 but not exactly 1 since predictions are close but not exact
    let r2 = r2_score(&predicted, &actual);
    assert!(r2 > 0.95); // Verify R² is close to 1

    // Perfect prediction
    let perfect_actual = vec![1.0, 2.0, 3.0];
    let perfect_predicted = vec![1.0, 2.0, 3.0];
    assert!((r2_score(&perfect_predicted, &perfect_actual) - 1.0).abs() < EPSILON);

    // When prediction is always the mean, R² should be 0
    let mean_actual = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // mean is 3
    let mean_predicted = vec![3.0, 3.0, 3.0, 3.0, 3.0];
    assert!((r2_score(&mean_predicted, &mean_actual) - 0.0).abs() < EPSILON);

    // When all actual values are the same, R² should be 0
    let same_actual = vec![7.0, 7.0, 7.0];
    let same_predicted = vec![6.0, 7.0, 8.0];
    assert!((r2_score(&same_predicted, &same_actual) - 0.0).abs() < EPSILON);
}

#[test]
fn test_sigmoid() {
    // Test the midpoint - sigmoid(0) should be exactly 0.5
    assert!((sigmoid(0.0) - 0.5).abs() < EPSILON);

    // Test extreme positive input - should approach 1.0
    assert!((sigmoid(10.0) - 1.0).abs() < 0.0001);

    // Test extreme negative input - should approach 0.0
    assert!((sigmoid(-10.0) - 0.0).abs() < 0.0001);

    // Test symmetry property: sigmoid(-z) = 1 - sigmoid(z)
    let z = 2.5;
    assert!((sigmoid(-z) - (1.0 - sigmoid(z))).abs() < EPSILON);

    // Test a few known values
    assert!((sigmoid(1.0) - 0.7310585786300049).abs() < EPSILON);
    assert!((sigmoid(-1.0) - 0.2689414213699951).abs() < EPSILON);
}

#[test]
fn test_logistic_loss() {
    // Test case 1: Very good predictions
    let predictions1 = vec![10.0, -10.0];  // Very confident predictions
    let labels1 = vec![1.0, 0.0];          // Actual labels
    let loss1 = logistic_loss(&predictions1, &labels1);
    println!("Loss1: {}", loss1);
    assert!(loss1 < 0.01);  // Loss should be small

    // Test case 2: Completely wrong predictions
    let predictions2 = vec![10.0, -10.0];
    let labels2 = vec![0.0, 1.0];
    let loss2 = logistic_loss(&predictions2, &labels2);
    println!("Loss2: {}", loss2);
    assert!(loss2 > 9.0);

    // Test case 3: Uncertain predictions
    let predictions3 = vec![0.1, -0.1, 0.0];
    let labels3 = vec![1.0, 0.0, 1.0];
    let loss3 = logistic_loss(&predictions3, &labels3);
    println!("Loss3: {}", loss3);
    assert!(loss3 > 0.5 && loss3 < 1.0);
}

#[test]
#[should_panic(expected = "must be either 0 or 1")]
fn test_logistic_loss_invalid_labels() {
    let predictions = vec![1.0, 2.0];
    let labels = vec![1.0, 0.5]; // Invalid label value
    logistic_loss(&predictions, &labels); // Should panic
}

#[test]
#[should_panic(expected = "Raw predictions and actual labels must have the same length")]
fn test_logistic_loss_different_lengths() {
    // Different length arrays should panic
    let predictions = vec![0.1, 0.2, 0.3];
    let labels = vec![0.0, 1.0];
    logistic_loss(&predictions, &labels);
}

#[test]
fn test_accuracy() {
    // Test case 1: Binary classification with perfect prediction
    let predicted = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    let actual = vec![1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(accuracy(&predicted, &actual), 1.0);

    // Test case 2: Binary classification with some errors
    let predicted = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let actual = vec![1.0, 0.0, 0.0, 1.0, 1.0];
    assert_eq!(accuracy(&predicted, &actual), 0.6);

    // Test case 3: Multi-class classification
    let predicted = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0];
    let actual = vec![0.0, 1.0, 2.0, 2.0, 1.0, 1.0];
    assert_eq!(accuracy(&predicted, &actual), 2.0/3.0); // 4 out of 6 correct

    // Test case 4: Edge case - empty arrays
    let empty_vec: Vec<f64> = vec![];
    assert_eq!(accuracy(&empty_vec, &empty_vec), 0.0);

    // Test case 5: Floating point equality with very small differences
    // (should be considered equal due to epsilon comparison)
    let predicted = vec![0.0000001, 1.0];
    let actual = vec![0.0, 1.0];
    assert!(accuracy(&predicted, &actual) < 1.0); // Should fail equality due to floating point precision

    // Test case 6: All predictions are wrong
    let predicted = vec![1.0, 1.0, 0.0, 0.0];
    let actual = vec![0.0, 0.0, 1.0, 1.0];
    assert_eq!(accuracy(&predicted, &actual), 0.0);
}

#[test]
#[should_panic(expected = "Predicted and actual arrays must have the same length")]
fn test_accuracy_with_different_length_arrays() {
    let predicted = vec![1.0, 0.0, 1.0];
    let actual = vec![1.0, 0.0];
    accuracy(&predicted, &actual); // This should panic
}
