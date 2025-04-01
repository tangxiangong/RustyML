use ndarray::array;
use crate::metric::*;

#[test]
fn test_mse_empty() {
    let empty: Vec<f64> = vec![];
    assert_eq!(mean_squared_error(&empty), 0.0);
}

#[test]
fn test_root_mean_squared_error() {
    // Test basic calculation
    let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = root_mean_squared_error(&predictions, &targets);
    assert!((result - 0.0).abs() < f64::EPSILON, "Expected 0.0, got {}", result);

    // Test calculation with constant error
    let predictions = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = root_mean_squared_error(&predictions, &targets);
    assert!((result - 1.0).abs() < f64::EPSILON, "Expected 1.0, got {}", result);

    // Test more complex example
    let predictions = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = root_mean_squared_error(&predictions, &targets);
    assert!((result - 0.5).abs() < f64::EPSILON, "Expected 0.5, got {}", result);

    // Test negative values
    let predictions = vec![-1.0, -2.0, -3.0];
    let targets = vec![1.0, 2.0, 3.0];
    let result = root_mean_squared_error(&predictions, &targets);
    assert!((result - 4.320493798938574).abs() < f64::EPSILON, "Expected 4.320493798938574, got {}", result);
}

#[test]
#[should_panic(expected = "Input arrays cannot be empty")]
fn test_rmse_empty_arrays() {
    // Test empty arrays
    let empty: Vec<f64> = vec![];
    root_mean_squared_error(&empty, &empty);
}

#[test]
#[should_panic(expected = "Prediction and target arrays must have the same length")]
fn test_rmse_mismatched_lengths() {
    // Test arrays with mismatched lengths
    let predictions = vec![1.0, 2.0, 3.0];
    let targets = vec![1.0, 2.0];
    root_mean_squared_error(&predictions, &targets);
}

#[test]
fn test_mean_absolute_error() {
    // Test identical values (zero error)
    let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = mean_absolute_error(&predictions, &targets);
    assert!((result - 0.0).abs() < f64::EPSILON, "Expected 0.0, got {}", result);

    // Test constant error
    let predictions = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = mean_absolute_error(&predictions, &targets);
    assert!((result - 1.0).abs() < f64::EPSILON, "Expected 1.0, got {}", result);

    // Test mixed positive and negative errors
    let predictions = vec![1.0, 3.0, 2.0];
    let targets = vec![2.0, 1.0, 3.0];
    let result = mean_absolute_error(&predictions, &targets);
    assert!((result - 1.3333333333333333).abs() < f64::EPSILON, "Expected 1.3333333333333333, got {}", result);

    // Test negative values
    let predictions = vec![-1.0, -2.0, -3.0];
    let targets = vec![1.0, 2.0, 3.0];
    let result = mean_absolute_error(&predictions, &targets);
    assert!((result - 4.0).abs() < f64::EPSILON, "Expected 4.0, got {}", result);

    // Test decimal values
    let predictions = vec![1.5, 2.5, 3.5];
    let targets = vec![1.0, 2.0, 3.0];
    let result = mean_absolute_error(&predictions, &targets);
    assert!((result - 0.5).abs() < f64::EPSILON, "Expected 0.5, got {}", result);
}

#[test]
#[should_panic(expected = "Input arrays cannot be empty")]
fn test_mae_empty_arrays() {
    // Test empty arrays
    let empty: Vec<f64> = vec![];
    mean_absolute_error(&empty, &empty);
}

#[test]
#[should_panic(expected = "Prediction and target arrays must have the same length")]
fn test_mae_mismatched_lengths() {
    // Test arrays with mismatched lengths
    let predictions = vec![1.0, 2.0, 3.0];
    let targets = vec![1.0, 2.0];
    mean_absolute_error(&predictions, &targets);
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
    assert!((r2_score(&perfect_predicted, &perfect_actual) - 1.0).abs() < f64::EPSILON);

    // When prediction is always the mean, R² should be 0
    let mean_actual = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // mean is 3
    let mean_predicted = vec![3.0, 3.0, 3.0, 3.0, 3.0];
    assert!((r2_score(&mean_predicted, &mean_actual) - 0.0).abs() < f64::EPSILON);

    // When all actual values are the same, R² should be 0
    let same_actual = vec![7.0, 7.0, 7.0];
    let same_predicted = vec![6.0, 7.0, 8.0];
    assert!((r2_score(&same_predicted, &same_actual) - 0.0).abs() < f64::EPSILON);
}


fn assert_float_eq(a: f64, b: f64) {
    assert!((a - b).abs() < f64::EPSILON, "Expected {}, got {}", b, a);
}

#[test]
fn test_confusion_matrix_new() {
    // Test perfect classification
    let pred = array![1.0, 0.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);

    assert_eq!(cm.get_counts(), (2, 0, 2, 0));

    // Test various classification error scenarios
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);

    assert_eq!(cm.get_counts(), (1, 1, 1, 1));

    // Test probability threshold (0.5)
    let pred = array![0.6, 0.4, 0.7, 0.3];
    let actual = array![0.9, 0.1, 0.2, 0.8];
    let cm = ConfusionMatrix::new(&pred, &actual);

    assert_eq!(cm.get_counts(), (1, 1, 1, 1));
}

#[test]
#[should_panic(expected = "Predicted and actual arrays must have the same length")]
fn test_confusion_matrix_new_error() {
    // Test case with mismatched input lengths
    let pred = array![1.0, 0.0, 1.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let _result = ConfusionMatrix::new(&pred, &actual);
}

#[test]
fn test_get_counts() {
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);

    let counts = cm.get_counts();
    assert_eq!(counts, (1, 1, 1, 1));
}

#[test]
fn test_confusion_matrix_accuracy() {
    // Perfect classification
    let pred = array![1.0, 0.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.accuracy(), 1.0);

    // 50% correct
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.accuracy(), 0.5);

    // All incorrect
    let pred = array![1.0, 1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.accuracy(), 0.0);
}

#[test]
fn test_error_rate() {
    // Perfect classification
    let pred = array![1.0, 0.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.error_rate(), 0.0);

    // 50% errors
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.error_rate(), 0.5);

    // All errors
    let pred = array![1.0, 1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.error_rate(), 1.0);
}

#[test]
fn test_precision() {
    // Perfect precision
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.precision(), 1.0);

    // Partial precision
    let pred = array![1.0, 1.0, 1.0, 0.0];
    let actual = array![1.0, 0.0, 1.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.precision(), 2.0/3.0);

    // No true positives
    let pred = array![1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.precision(), 0.0);
}

#[test]
fn test_recall() {
    // Perfect recall
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.recall(), 1.0);

    // Partial recall
    let pred = array![1.0, 0.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.recall(), 0.5);

    // No actual positives
    let pred = array![0.0, 0.0, 0.0];
    let actual = array![0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.recall(), 1.0); // Note: when there are no actual positives, recall is 1.0 (avoiding 0/0 case)
}

#[test]
fn test_specificity() {
    // Perfect specificity
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.specificity(), 1.0);

    // Partial specificity
    let pred = array![1.0, 1.0, 0.0, 1.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.specificity(), 0.5);

    // No actual negatives
    let pred = array![1.0, 1.0, 1.0];
    let actual = array![1.0, 1.0, 1.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.specificity(), 1.0); // Note: when there are no actual negatives, specificity is 1.0 (avoiding 0/0 case)
}

#[test]
fn test_f1_score() {
    // Perfect F1 score
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 1.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.f1_score(), 1.0);

    // F1 score calculation example
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    // precision = 1/2, recall = 1/2, F1 = 2*1/2*1/2 / (1/2 + 1/2) = 1/2
    assert_float_eq(cm.f1_score(), 0.5);

    // Precision or recall is 0
    let pred = array![1.0, 1.0, 1.0];
    let actual = array![0.0, 0.0, 0.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    assert_float_eq(cm.f1_score(), 0.0);
}

#[test]
fn test_summary() {
    let pred = array![1.0, 1.0, 0.0, 0.0];
    let actual = array![1.0, 0.0, 0.0, 1.0];
    let cm = ConfusionMatrix::new(&pred, &actual);
    let summary = cm.summary();

    // Check if the summary string contains all necessary information
    assert!(summary.contains("TP:"));
    assert!(summary.contains("FP:"));
    assert!(summary.contains("TN:"));
    assert!(summary.contains("FN:"));
    assert!(summary.contains("Accuracy:"));
    assert!(summary.contains("Error Rate:"));
    assert!(summary.contains("Precision:"));
    assert!(summary.contains("Recall:"));
    assert!(summary.contains("Specificity:"));
    assert!(summary.contains("F1 Score:"));
}