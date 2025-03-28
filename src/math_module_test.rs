use crate::math::*;
use ndarray::Array2;

#[test]
fn test_sum_of_square_total() {
    // Standard case
    let values = vec![2.0, 4.0, 6.0, 8.0];
    // Mean is 5.0, so SST = (2-5)² + (4-5)² + (6-5)² + (8-5)² = 9 + 1 + 1 + 9 = 20
    let expected = 20.0;
    assert!((sum_of_square_total(&values) - expected).abs() < f64::EPSILON);

    // Case where all values are identical
    let same_values = vec![5.0, 5.0, 5.0, 5.0];
    // Mean is 5.0, so SST = 0
    assert!((sum_of_square_total(&same_values) - 0.0).abs() < f64::EPSILON);

    // Empty array case
    let empty = Vec::<f64>::new();
    assert_eq!(sum_of_square_total(&empty), 0.0);

    // Single value case
    let single = vec![10.0];
    assert!((sum_of_square_total(&single) - 0.0).abs() < f64::EPSILON);
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
    assert!((sum_of_squared_errors(&perfect_prediction, &actual_values) - 0.0).abs() < f64::EPSILON);

    // Empty arrays case
    let empty_pred = Vec::<f64>::new();
    let empty_actual = Vec::<f64>::new();
    assert!((sum_of_squared_errors(&empty_pred, &empty_actual) - 0.0).abs() < f64::EPSILON);
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

#[test]
fn test_sigmoid() {
    // Test the midpoint - sigmoid(0) should be exactly 0.5
    assert!((sigmoid(0.0) - 0.5).abs() < f64::EPSILON);

    // Test extreme positive input - should approach 1.0
    assert!((sigmoid(10.0) - 1.0).abs() < 0.0001);

    // Test extreme negative input - should approach 0.0
    assert!((sigmoid(-10.0) - 0.0).abs() < 0.0001);

    // Test symmetry property: sigmoid(-z) = 1 - sigmoid(z)
    let z = 2.5;
    assert!((sigmoid(-z) - (1.0 - sigmoid(z))).abs() < f64::EPSILON);

    // Test a few known values
    assert!((sigmoid(1.0) - 0.7310585786300049).abs() < f64::EPSILON);
    assert!((sigmoid(-1.0) - 0.2689414213699951).abs() < f64::EPSILON);
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

#[test]
fn test_squared_distance() {
    use ndarray::array;
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[2.0, 3.0], [4.0, 5.0]];

    // (1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2 = 1 + 1 + 1 + 1 = 4
    let expected = 4.0;

    let result = squared_euclidean_distance(&a.view(), &b.view());
    assert_eq!(result, expected);

    let zero_distance = squared_euclidean_distance(&a.view(), &a.view());
    assert_eq!(zero_distance, 0.0);

    let c = array![[1.0, 2.0]];
    let d = array![[2.0, 4.0]];
    let expected_cd = (1.0_f64 - 2.0_f64).powi(2) + (2.0_f64 - 4.0_f64).powi(2);
    let result_cd = squared_euclidean_distance(&c.view(), &d.view());
    assert_eq!(result_cd, expected_cd);
}

#[test]
fn test_manhattan_distance() {
    // Basic test case
    let a = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let b = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();

    let dist = manhattan_distance(&a.view(), &b.view());
    assert_eq!(dist, 9.0); // |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9

    // Test case with negative values
    let c = Array2::<f64>::from_shape_vec((1, 3), vec![-1.0, -2.0, -3.0]).unwrap();
    let d = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();

    let dist2 = manhattan_distance(&c.view(), &d.view());
    assert_eq!(dist2, 12.0); // |1-(-1)| + |2-(-2)| + |3-(-3)| = 2 + 4 + 6 = 12

    // Test with zero vector
    let zero = Array2::<f64>::zeros((1, 3));
    let e = Array2::<f64>::from_shape_vec((1, 3), vec![5.0, 5.0, 5.0]).unwrap();

    let dist3 = manhattan_distance(&zero.view(), &e.view());
    assert_eq!(dist3, 15.0); // |0-5| + |0-5| + |0-5| = 15
}

#[test]
fn test_minkowski_distance() {
    let a = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let b = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();

    // p=1 should be equivalent to Manhattan distance
    let dist_p1 = minkowski_distance(&a.view(), &b.view(), 1.0);
    assert!((dist_p1 - 9.0).abs() < 1e-10);

    // p=2 should be equivalent to Euclidean distance
    let dist_p2 = minkowski_distance(&a.view(), &b.view(), 2.0);
    let expected = (3.0f64.powi(2) + 3.0f64.powi(2) + 3.0f64.powi(2)).sqrt();
    assert!((dist_p2 - expected).abs() < 1e-10);
    assert!((dist_p2 - 5.196152).abs() < 1e-6);

    // Test with p=3
    let dist_p3 = minkowski_distance(&a.view(), &b.view(), 3.0);
    let expected_p3 = (3.0f64.powi(3) + 3.0f64.powi(3) + 3.0f64.powi(3)).powf(1.0/3.0);
    assert!((dist_p3 - expected_p3).abs() < 1e-10);

    // Test with vectors containing zeros
    let c = Array2::<f64>::from_shape_vec((1, 2), vec![3.0, 0.0]).unwrap();
    let d = Array2::<f64>::from_shape_vec((1, 2), vec![0.0, 4.0]).unwrap();

    let dist_p1_cd = minkowski_distance(&c.view(), &d.view(), 1.0);
    assert_eq!(dist_p1_cd, 7.0); // |0-3| + |4-0| = 3 + 4 = 7

    let dist_p2_cd = minkowski_distance(&c.view(), &d.view(), 2.0);
    assert!((dist_p2_cd - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
}

#[test]
#[should_panic(expected = "p must be greater than or equal to 1.0")]
fn test_minkowski_distance_invalid_p() {
    let a = Array2::<f64>::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
    let b = Array2::<f64>::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap();

    // p < 1.0 should panic
    minkowski_distance(&a.view(), &b.view(), 0.5);
}

#[test]
fn test_entropy_empty() {
    let empty: Vec<f64> = vec![];
    assert_eq!(entropy(&empty), 0.0);
}

#[test]
fn test_entropy_homogeneous() {
    // Dataset with only one class, entropy should be 0
    let homogeneous = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    assert!(entropy(&homogeneous).abs() < f64::EPSILON);
}

#[test]
fn test_entropy_balanced_binary() {
    // Balanced binary dataset, entropy should be 1
    let balanced = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    assert!((entropy(&balanced) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_entropy_unbalanced() {
    // Unbalanced dataset
    let unbalanced = vec![0.0, 1.0, 1.0, 1.0];
    // Probabilities: 0.0 -> 1/4, 1.0 -> 3/4
    // Entropy = -(0.25*log2(0.25) + 0.75*log2(0.75))
    let expected = -(0.25 * 0.25f64.log2() + 0.75 * 0.75f64.log2());
    assert!((entropy(&unbalanced) - expected).abs() < f64::EPSILON);
}

#[test]
fn test_entropy_multiclass() {
    // Multi-class dataset
    let multiclass = vec![0.0, 1.0, 2.0, 3.0];
    // Each class has probability 0.25
    // Entropy = -(4 * 0.25*log2(0.25))
    let expected = -(4.0 * 0.25 * 0.25f64.log2());
    assert!((entropy(&multiclass) - expected).abs() < f64::EPSILON);
}

#[test]
fn test_gini_empty() {
    let empty: Vec<f64> = vec![];
    assert_eq!(gini(&empty), 0.0);
}

#[test]
fn test_gini_homogeneous() {
    // Homogeneous dataset, Gini impurity should be 0
    let homogeneous = vec![1.0, 1.0, 1.0, 1.0];
    assert!(gini(&homogeneous).abs() < f64::EPSILON);
}

#[test]
fn test_gini_balanced_binary() {
    // Balanced binary dataset, Gini impurity should be 0.5
    let balanced = vec![0.0, 0.0, 1.0, 1.0];
    // Gini = 1 - (0.5^2 + 0.5^2) = 0.5
    assert!((gini(&balanced) - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_gini_unbalanced() {
    // Unbalanced dataset
    let unbalanced = vec![0.0, 1.0, 1.0, 1.0];
    // Probabilities: 0.0 -> 1/4, 1.0 -> 3/4
    // Gini = 1 - (0.25^2 + 0.75^2) = 1 - (0.0625 + 0.5625) = 0.375
    let expected = 1.0 - (0.25f64.powi(2) + 0.75f64.powi(2));
    assert!((gini(&unbalanced) - expected).abs() < f64::EPSILON);
}

#[test]
fn test_information_gain() {
    let parent = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];  // 3 zeros, 3 ones
    let left = vec![0.0, 0.0, 0.0];                   // 3 zeros
    let right = vec![1.0, 1.0, 1.0];                  // 3 ones

    // Parent entropy = 1.0
    // Left child entropy = 0.0
    // Right child entropy = 0.0
    // Information gain = 1.0 - (3/6)*0.0 - (3/6)*0.0 = 1.0
    assert!((information_gain(&parent, &left, &right) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_information_gain_no_improvement() {
    let parent = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];  // 3 zeros, 3 ones
    let left = vec![0.0, 0.0, 1.0];                   // 2 zeros, 1 one
    let right = vec![0.0, 1.0, 1.0];                  // 1 zero, 2 ones

    // Left and right child nodes have similar entropy distribution to parent
    // So information gain should be close to 0
    let gain = information_gain(&parent, &left, &right);
    assert!(gain < 0.1);  // Allow for small error
}

#[test]
fn test_gain_ratio() {
    let parent = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];  // 3 zeros, 3 ones
    let left = vec![0.0, 0.0, 0.0];                   // 3 zeros
    let right = vec![1.0, 1.0, 1.0];                  // 3 ones

    // Information gain = 1.0
    // Split info = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0
    // Gain ratio = 1.0/1.0 = 1.0
    assert!((gain_ratio(&parent, &left, &right) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_gain_ratio_zero_split_info() {
    // Test case where split info is zero
    let parent = vec![0.0, 0.0, 1.0, 1.0];
    let left = vec![0.0, 0.0, 1.0, 1.0];  // All samples in left node
    let right: Vec<f64> = vec![];         // Right node empty

    // Split info should be 0, function should return 0
    assert_eq!(gain_ratio(&parent, &left, &right), 0.0);
}

#[test]
fn test_mse_empty() {
    let empty: Vec<f64> = vec![];
    assert_eq!(mean_squared_error(&empty), 0.0);
}

#[test]
fn test_mse_constant() {
    // All values are the same, MSE should be 0
    let constant = vec![5.0, 5.0, 5.0, 5.0];
    assert!(mean_squared_error(&constant).abs() < f64::EPSILON);
}

#[test]
fn test_mse_simple() {
    // Simple case: [1, 2, 3]
    // Mean = 2
    // MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
    let values = vec![1.0, 2.0, 3.0];
    let expected = 2.0 / 3.0;
    assert!((mean_squared_error(&values) - expected).abs() < f64::EPSILON);
}

#[test]
fn test_mse_with_variance() {
    // Case with larger variance
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    // Mean = 30
    // MSE = ((10-30)^2 + (20-30)^2 + (30-30)^2 + (40-30)^2 + (50-30)^2) / 5
    //     = (400 + 100 + 0 + 100 + 400) / 5 = 1000 / 5 = 200
    let expected = 200.0;
    assert!((mean_squared_error(&values) - expected).abs() < f64::EPSILON);
}