use crate::machine_learning::isolation_forest::IsolationForest;
use ndarray::{Array2};

#[test]
fn test_isolation_forest_new() {
    // Test creation with default parameters
    let forest = IsolationForest::default();
    assert_eq!(forest.get_n_estimators(), 100); // Assuming default value is 100
    assert_eq!(forest.get_max_samples(), 256); // Assuming default value is 256
    assert_eq!(forest.get_max_depth(), 8); // Assuming default value is 8
    assert_eq!(forest.get_random_state(), None);

    // Test creation with custom parameters
    let forest = IsolationForest::new(50, 100, Some(10), Some(42));
    assert_eq!(forest.get_n_estimators(), 50);
    assert_eq!(forest.get_max_samples(), 100);
    assert_eq!(forest.get_max_depth(), 10);
    assert_eq!(forest.get_random_state(), Some(42));
}

#[test]
fn test_isolation_forest_getters() {
    let forest = IsolationForest::new(50, 100, Some(10), Some(42));

    assert_eq!(forest.get_n_estimators(), 50);
    assert_eq!(forest.get_max_samples(), 100);
    assert_eq!(forest.get_max_depth(), 10);
    assert_eq!(forest.get_random_state(), Some(42));

    // Test that get_trees returns an error before model is fitted
    match forest.get_trees() {
        Err(_) => assert!(true), // Expected failure
        Ok(_) => panic!("Should return an error before fitting"),
    }
}

#[test]
fn test_isolation_forest_fit() {
    // Create a 2D sample data
    let data = Array2::from_shape_vec((10, 2), vec![
        0.0, 0.0,
        0.1, 0.1,
        0.2, 0.2,
        0.3, 0.3,
        0.4, 0.4,
        0.5, 0.5,
        0.6, 0.6,
        0.7, 0.7,
        0.8, 0.8,
        10.0, 10.0, // Outlier
    ]).unwrap();

    let mut forest = IsolationForest::new(10, 8, Some(5), Some(42));
    forest.fit(&data);

    // After fitting, we should be able to get the trees
    assert!(forest.get_trees().is_ok());
    if let Ok(trees) = forest.get_trees() {
        assert_eq!(trees.len(), 10); // Should have n_estimators trees
    }
}

#[test]
fn test_isolation_forest_anomaly_score() {
    // Create a 2D sample data with one clear outlier
    let data = Array2::from_shape_vec((10, 2), vec![
        0.0, 0.0,
        0.1, 0.1,
        0.2, 0.2,
        0.3, 0.3,
        0.4, 0.4,
        0.5, 0.5,
        0.6, 0.6,
        0.7, 0.7,
        0.8, 0.8,
        10.0, 10.0, // Outlier
    ]).unwrap();

    let mut forest = IsolationForest::new(100, 8, Some(8), Some(42));
    forest.fit(&data);

    // Test anomaly score for a normal point
    let normal_point = vec![0.5, 0.5];
    let normal_score = forest.anomaly_score(&normal_point).unwrap();

    // Test anomaly score for an outlier point
    let anomaly_point = vec![10.0, 10.0];
    let anomaly_score = forest.anomaly_score(&anomaly_point).unwrap();

    // The outlier's score should be higher than the normal point's
    assert!(anomaly_score > normal_score);
}

#[test]
fn test_isolation_forest_predict() {
    // Create a 2D sample training data
    let train_data = Array2::from_shape_vec((10, 2), vec![
        0.0, 0.0,
        0.1, 0.1,
        0.2, 0.2,
        0.3, 0.3,
        0.4, 0.4,
        0.5, 0.5,
        0.6, 0.6,
        0.7, 0.7,
        0.8, 0.8,
        0.9, 0.9,
    ]).unwrap();

    let mut forest = IsolationForest::new(100, 8, Some(8), Some(42));
    forest.fit(&train_data);

    // Create test data with both normal and anomaly points
    let test_data = Array2::from_shape_vec((4, 2), vec![
        0.5, 0.5,   // Normal point
        0.7, 0.7,   // Normal point
        10.0, 10.0, // Outlier
        -5.0, -5.0  // Outlier
    ]).unwrap();

    // Predict
    let predictions = forest.predict(&test_data).unwrap();
    assert_eq!(predictions.len(), 4);

    // The outliers' scores in the predictions array should be higher
    assert!(predictions[2] > predictions[0]); // 10.0, 10.0 > 0.5, 0.5
    assert!(predictions[3] > predictions[1]); // -5.0, -5.0 > 0.7, 0.7
}

#[test]
fn test_error_handling() {
    let forest = IsolationForest::default();

    // Test calling anomaly detection functions before fitting
    let sample = vec![0.5, 0.5];
    match forest.anomaly_score(&sample) {
        Err(_) => assert!(true), // Expected failure
        Ok(_) => panic!("Should return error when model is not fitted"),
    }

    // Test calling predict function before fitting
    let test_data = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.7, 0.7]).unwrap();
    match forest.predict(&test_data) {
        Err(_) => assert!(true), // Expected failure
        Ok(_) => panic!("Should return error when model is not fitted"),
    }
}

#[test]
fn test_dimension_validation() {
    // Train on 2D data
    let train_data = Array2::from_shape_vec((5, 2), vec![
        0.0, 0.0,
        0.1, 0.1,
        0.2, 0.2,
        0.3, 0.3,
        0.4, 0.4,
    ]).unwrap();

    let mut forest = IsolationForest::new(10, 4, Some(5), Some(42));
    forest.fit(&train_data);

    // Test with 2D point (should work)
    let correct_sample = vec![0.2, 0.2];
    assert!(forest.anomaly_score(&correct_sample).is_ok());

    // Test with 1D point (should fail)
    let too_small_sample = vec![0.2];
    assert!(forest.anomaly_score(&too_small_sample).is_err());

    // Test with 3D point (should fail)
    let too_large_sample = vec![0.2, 0.2, 0.2];
    assert!(forest.anomaly_score(&too_large_sample).is_err());
}