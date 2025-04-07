use crate::ModelError;
use crate::utility::linear_discriminant_analysis::LDA;
use approx::assert_relative_eq;
use ndarray::{Array1, Array2, arr1, arr2, s};

#[test]
fn test_lda_new() {
    let lda = LDA::new();
    assert!(lda.get_classes().is_err());
    assert!(lda.get_priors().is_err());
    assert!(lda.get_means().is_err());
    assert!(lda.get_cov_inv().is_err());
    assert!(lda.get_projection().is_err());
}

#[test]
fn test_lda_default() {
    let lda = LDA::default();
    assert!(lda.get_classes().is_err());
    assert!(lda.get_priors().is_err());
    assert!(lda.get_means().is_err());
    assert!(lda.get_cov_inv().is_err());
    assert!(lda.get_projection().is_err());
}

#[test]
fn test_getters_on_unfitted_model() {
    let lda = LDA::new();
    assert!(matches!(lda.get_classes(), Err(ModelError::NotFitted)));
    assert!(matches!(lda.get_priors(), Err(ModelError::NotFitted)));
    assert!(matches!(lda.get_means(), Err(ModelError::NotFitted)));
    assert!(matches!(lda.get_cov_inv(), Err(ModelError::NotFitted)));
    assert!(matches!(lda.get_projection(), Err(ModelError::NotFitted)));
}

#[test]
fn test_fit_basic() {
    // Simple binary classification data with two features
    let x = arr2(&[
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [5.0, 5.0],
        [5.5, 4.5],
        [6.0, 5.0],
    ]);
    let y = arr1(&[0, 0, 0, 1, 1, 1]);

    let mut lda = LDA::new();
    let result = lda.fit(&x, &y);
    assert!(result.is_ok());

    // Verify model properties after fitting
    assert!(lda.get_classes().is_ok());
    assert!(lda.get_priors().is_ok());
    assert!(lda.get_means().is_ok());
    assert!(lda.get_cov_inv().is_ok());
    assert!(lda.get_projection().is_ok());

    // Verify specific values
    let classes = lda.get_classes().unwrap();
    assert_eq!(classes.len(), 2);
    assert_eq!(classes[0], 0);
    assert_eq!(classes[1], 1);

    let priors = lda.get_priors().unwrap();
    assert_relative_eq!(priors[0], 0.5, epsilon = 1e-10);
    assert_relative_eq!(priors[1], 0.5, epsilon = 1e-10);

    let means = lda.get_means().unwrap();
    assert_eq!(means.shape(), &[2, 2]);

    // Class 0 means should be around [1.5, 2.5]
    assert_relative_eq!(means[[0, 0]], 1.5, epsilon = 1e-10);
    assert_relative_eq!(means[[0, 1]], 2.5, epsilon = 1e-10);

    // Class 1 means should be around [5.5, 4.83]
    assert_relative_eq!(means[[1, 0]], 5.5, epsilon = 1e-10);
    assert_relative_eq!(means[[1, 1]], 4.833333333333333, epsilon = 1e-10);

    // Projection matrix should have one column (n_classes - 1)
    let projection = lda.get_projection().unwrap();
    assert_eq!(projection.shape(), &[2, 1]);
}

#[test]
fn test_fit_validation_errors() {
    let mut lda = LDA::new();

    // Test empty data
    let x_empty = Array2::<f64>::zeros((0, 2));
    let y_empty = Array1::<i32>::zeros(0);
    assert!(matches!(lda.fit(&x_empty, &y_empty), Err(_)));

    // Test mismatched sample count and label count
    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let y = arr1(&[0]);
    assert!(matches!(lda.fit(&x, &y), Err(_)));

    // Test case with only one class
    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let y = arr1(&[0, 0]);
    assert!(matches!(lda.fit(&x, &y), Err(_)));
}

#[test]
fn test_predict() {
    // Training data
    let x = arr2(&[
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [5.0, 5.0],
        [5.5, 4.5],
        [6.0, 5.0],
    ]);
    let y = arr1(&[0, 0, 0, 1, 1, 1]);

    // Test data
    let x_test = arr2(&[
        [1.2, 2.2], // Should be predicted as class 0
        [5.2, 4.8], // Should be predicted as class 1
    ]);

    let mut lda = LDA::new();
    lda.fit(&x, &y).unwrap();

    let predictions = lda.predict(&x_test).unwrap();
    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0], 0);
    assert_eq!(predictions[1], 1);

    // Test prediction on unfitted model
    let lda_unfitted = LDA::new();
    assert!(matches!(
        lda_unfitted.predict(&x_test),
        Err(ModelError::NotFitted)
    ));

    // Test empty input
    assert!(matches!(lda.predict(&Array2::<f64>::zeros((0, 2))), Err(_)));
}

#[test]
fn test_transform() {
    // Training data
    let x = arr2(&[
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [5.0, 5.0],
        [5.5, 4.5],
        [6.0, 5.0],
    ]);
    let y = arr1(&[0, 0, 0, 1, 1, 1]);

    let mut lda = LDA::new();
    lda.fit(&x, &y).unwrap();

    // Transform data
    let transformed = lda.transform(&x, 1).unwrap();
    assert_eq!(transformed.shape(), &[6, 1]);

    // Check if transformed data shows expected separation - same class should cluster together
    let class0_transformed = transformed.slice(s![0..3, 0]);
    let class1_transformed = transformed.slice(s![3..6, 0]);

    // All class 0 values should be on the same side, class 1 values on the other side
    // Fix: Use reference to the first element to get sign
    let class0_sign = class0_transformed.iter().next().unwrap().signum();
    for val in class0_transformed.iter() {
        assert_eq!(val.signum(), class0_sign);
    }

    let class1_sign = class1_transformed.iter().next().unwrap().signum();
    for val in class1_transformed.iter() {
        assert_eq!(val.signum(), class1_sign);
    }

    // Test invalid component count
    assert!(matches!(lda.transform(&x, 0), Err(_)));
    assert!(matches!(lda.transform(&x, 2), Err(_)));

    // Test unfitted model
    let lda_unfitted = LDA::new();
    assert!(matches!(
        lda_unfitted.transform(&x, 1),
        Err(ModelError::NotFitted)
    ));
}

#[test]
fn test_fit_transform() {
    // Training data
    let x = arr2(&[
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
        [5.0, 5.0],
        [5.5, 4.5],
        [6.0, 5.0],
    ]);
    let y = arr1(&[0, 0, 0, 1, 1, 1]);

    let mut lda = LDA::new();

    // Test fit_transform
    let transformed = lda.fit_transform(&x, &y, 1).unwrap();
    assert_eq!(transformed.shape(), &[6, 1]);

    // Verify the model has been fitted
    assert!(lda.get_classes().is_ok());
    assert!(lda.get_priors().is_ok());
    assert!(lda.get_means().is_ok());
    assert!(lda.get_cov_inv().is_ok());
    assert!(lda.get_projection().is_ok());

    // Transformed data should match separate transform call
    let transformed_separate = lda.transform(&x, 1).unwrap();
    for i in 0..transformed.len() {
        assert_relative_eq!(
            transformed[[i, 0]],
            transformed_separate[[i, 0]],
            epsilon = 1e-10
        );
    }
}
