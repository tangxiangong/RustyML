use crate::math::{sum_of_square_total, sum_of_squared_errors, r2_score};
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
