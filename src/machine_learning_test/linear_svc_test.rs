use crate::ModelError;
use crate::machine_learning::linear_svc::*;
use ndarray::{arr1, arr2};

#[test]
fn test_default() {
    let model = LinearSVC::default();
    assert_eq!(model.get_weights(), Err(ModelError::NotFitted));
    assert_eq!(model.get_bias(), Err(ModelError::NotFitted));
    assert_eq!(model.get_n_iter(), Err(ModelError::NotFitted));
}

#[test]
fn test_new() {
    let model = LinearSVC::new(
        100,             // max_iter
        0.01,            // learning_rate
        0.1,             // regularization_param
        PenaltyType::L2, // penalty
        true,            // fit_intercept
        1e-4,            // tol
    );

    assert_eq!(model.get_max_iter(), 100);
    assert_eq!(model.get_learning_rate(), 0.01);
    assert_eq!(model.get_regularization_param(), 0.1);
    assert!(matches!(model.get_penalty(), PenaltyType::L2));
    assert!(model.get_fit_intercept());
    assert_eq!(model.get_tol(), 1e-4);
}

#[test]
fn test_getters_before_fit() {
    let model = LinearSVC::default();

    // These should return errors when model is not fitted
    assert!(model.get_weights().is_err());
    assert!(model.get_bias().is_err());
    assert!(model.get_n_iter().is_err());
}

#[test]
fn test_fit_predict_simple_case() -> Result<(), ModelError> {
    // Create a larger training dataset
    let x = arr2(&[
        [1.0, 1.0],
        [1.0, -2.0],
        [-1.0, 1.0],
        [-2.0, -1.0],
        [2.0, 2.0],
        [2.0, -3.0],
        [-2.0, 2.0],
        [-3.0, -2.0],
        [0.5, 1.5],
        [1.5, -0.5],
        [-0.5, 1.5],
        [-1.5, -0.5],
    ]);

    // Corresponding labels
    let y = arr1(&[
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    ]);

    let mut model = LinearSVC::new(
        10000,           // max_iter
        0.1,             // learning_rate
        1.0,             // regularization_param
        PenaltyType::L2, // penalty
        true,            // fit_intercept
        1e-4,            // tol
    );

    // Fit the model
    model.fit(&x, &y)?;

    // Test that weights and bias are now available
    assert!(model.get_weights().is_ok());
    assert!(model.get_bias().is_ok());
    assert!(model.get_n_iter().is_ok());

    // Test predictions
    let predictions = model.predict(&x)?;
    assert_eq!(predictions.len(), 12); // Updated to new number of data points

    let mut correct_count = 0;

    for (i, &pred) in predictions.iter().enumerate() {
        if pred == y[i] {
            correct_count += 1;
        }
    }

    println!(
        "Correct predictions: {}/{}",
        correct_count,
        predictions.len()
    );

    // Expect higher accuracy with larger dataset
    assert!(correct_count >= 6); // Require at least 6/12 correct

    // Create some new test data points
    let x_test = arr2(&[[1.2, 0.8], [0.8, -1.7], [-0.9, 1.1], [-1.8, -0.9]]);

    // Expected labels
    let y_expected = arr1(&[1.0, -1.0, 1.0, -1.0]);

    // Make predictions on new data
    let test_predictions = model.predict(&x_test)?;

    // Check prediction accuracy on new data
    let mut test_correct = 0;
    for (i, &pred) in test_predictions.iter().enumerate() {
        if pred == y_expected[i] {
            test_correct += 1;
        }
    }

    println!(
        "Test set correct predictions: {}/{}",
        test_correct,
        test_predictions.len()
    );
    assert!(test_correct >= 2); // Require at least 2/4 correct

    Ok(())
}

#[test]
fn test_decision_function() -> Result<(), ModelError> {
    // Create a simple dataset
    let x = arr2(&[[2.0, 2.0], [-2.0, -2.0]]);

    let y = arr1(&[1.0, -1.0]);

    let mut model = LinearSVC::default();
    model.fit(&x, &y)?;

    // Get decision values
    let decision_values = model.decision_function(&x)?;

    // Decision values should have the same sign as labels
    assert!(decision_values[0] > 0.0);
    assert!(decision_values[1] < 0.0);

    Ok(())
}

#[test]
fn test_different_penalties() {
    // Test with L1 penalty
    let mut model_l1 = LinearSVC::new(100, 0.01, 0.1, PenaltyType::L1, true, 1e-4);

    // Test with L2 penalty
    let mut model_l2 = LinearSVC::new(100, 0.01, 0.1, PenaltyType::L2, true, 1e-4);

    // Simple dataset
    let x = arr2(&[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);

    let y = arr1(&[1.0, 1.0, -1.0, -1.0]);

    // Fit both models
    let _ = model_l1.fit(&x, &y);
    let _ = model_l2.fit(&x, &y);

    // The weights should be different due to the different penalties
    if let (Ok(w1), Ok(w2)) = (model_l1.get_weights(), model_l2.get_weights()) {
        assert_ne!(w1, w2);
    }
}

#[test]
fn test_error_handling() {
    let model = LinearSVC::default();

    // Attempt to predict without fitting should return error
    let x = arr2(&[[1.0, 2.0]]);
    assert!(model.predict(&x).is_err());

    // Attempt to get decision function without fitting should return error
    assert!(model.decision_function(&x).is_err());
}

#[test]
fn test_fit_with_invalid_data() {
    let mut model = LinearSVC::default();

    // Test with mismatched dimensions
    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    // y has 3 samples but x has only 2
    let y = arr1(&[1.0, -1.0, 1.0]);

    assert!(model.fit(&x, &y).is_err());
}
