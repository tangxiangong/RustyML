use crate::ModelError;
use crate::machine_learning::linear_regression::LinearRegression;

#[test]
fn test_default_constructor() {
    let model = LinearRegression::default();

    assert!(matches!(model.get_coefficients(), Err(ModelError::NotFitted)));
    assert!(matches!(model.get_intercept(), Err(ModelError::NotFitted)));
    assert!(matches!(model.get_n_iter(), Err(ModelError::NotFitted)));
    // Check if default values meet expectations
    assert_eq!(model.get_fit_intercept(), true); // Assuming default is true
    assert!(model.get_learning_rate() > 0.0);
    assert!(model.get_max_iter() > 0);
    assert!(model.get_tol() > 0.0);
}

#[test]
fn test_new_constructor() {
    let model = LinearRegression::new(false, 0.01, 1000, 1e-5);
    assert!(!model.get_fit_intercept());
    assert_eq!(model.get_learning_rate(), 0.01);
    assert_eq!(model.get_max_iter(), 1000);
    assert_eq!(model.get_tol(), 1e-5);
    assert!(matches!(model.get_n_iter(), Err(ModelError::NotFitted)));
}

#[test]
fn test_not_fitted_error() {
    let model = LinearRegression::default();

    // Try to get coefficients and intercept from an unfitted model
    let coef_result = model.get_coefficients();
    let intercept_result = model.get_intercept();

    assert!(matches!(coef_result, Err(ModelError::NotFitted)));
    assert!(matches!(intercept_result, Err(ModelError::NotFitted)));
}

#[test]
fn test_fit_and_predict() {
    // Create a simple dataset: y = 2*x + 1
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y = vec![3.0, 5.0, 7.0, 9.0];

    let mut model = LinearRegression::new(true, 0.01, 10000, 1e-8);
    model.fit(&x, &y);

    // Check if coefficients and intercept are close to expected values
    let coefficients = model.get_coefficients().unwrap();
    let intercept = model.get_intercept().unwrap();

    assert!((coefficients[0] - 2.0).abs() < 0.1);
    assert!((intercept - 1.0).abs() < 0.1);

    // Test predictions
    let test_x = vec![vec![5.0], vec![6.0]];
    let predictions = model.predict(&test_x).unwrap();

    assert!((predictions[0] - 11.0).abs() < 0.2);
    assert!((predictions[1] - 13.0).abs() < 0.2);
}

#[test]
fn test_multivariate_regression() {
    // Create multivariate dataset: y = 2*x1 + 3*x2 + 1
    // Using uncorrelated feature values to avoid multicollinearity issues
    let x = vec![
        vec![1.0, 2.0],  // Different values
        vec![2.0, 1.0],  // Inverse relationship
        vec![3.0, 4.0],
        vec![4.0, 3.0],
        vec![2.5, 3.5],  // Adding more samples
        vec![3.5, 2.5],
        vec![1.5, 3.0],
        vec![3.0, 1.5],
    ];

    // Calculate y values based on y = 2*x1 + 3*x2 + 1
    let mut y = Vec::new();
    for sample in &x {
        y.push(2.0 * sample[0] + 3.0 * sample[1] + 1.0);
    }

    // Create model and train
    // Using smaller learning rate and more iterations
    let mut model = LinearRegression::new(true, 0.005, 20000, 1e-10);
    model.fit(&x, &y);

    // Check if coefficients and intercept are close to expected values
    let coefficients = model.get_coefficients().unwrap();
    let intercept = model.get_intercept().unwrap();

    println!("Learned coefficients: {:?}", coefficients);
    println!("Learned intercept: {}", intercept);

    // Allow for some margin of error
    assert!((coefficients[0] - 2.0).abs() < 0.2,
            "First coefficient {} differs too much from expected 2.0", coefficients[0]);
    assert!((coefficients[1] - 3.0).abs() < 0.2,
            "Second coefficient {} differs too much from expected 3.0", coefficients[1]);
    assert!((intercept - 1.0).abs() < 0.2,
            "Intercept {} differs too much from expected 1.0", intercept);

    // Test predictions
    let test_x = vec![
        vec![5.0, 5.0],
        vec![2.0, 4.0],
    ];
    let predictions = model.predict(&test_x).unwrap();

    // Expected: y1 = 2*5 + 3*5 + 1 = 26, y2 = 2*2 + 3*4 + 1 = 17
    assert!((predictions[0] - 26.0).abs() < 0.5,
            "Prediction {} differs too much from expected 26.0", predictions[0]);
    assert!((predictions[1] - 17.0).abs() < 0.5,
            "Prediction {} differs too much from expected 17.0", predictions[1]);
}

#[test]
fn test_no_intercept() {
    // Test without intercept: y = 2*x
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y = vec![2.0, 4.0, 6.0, 8.0];

    let mut model = LinearRegression::new(false, 0.01, 10000, 1e-8);
    model.fit(&x, &y);

    // Check if coefficient is close to expected value (around 2.0)
    let coefficients = model.get_coefficients().unwrap();

    assert!((coefficients[0] - 2.0).abs() < 0.1);

    // Intercept should be close to 0 or not available
    if let Ok(intercept) = model.get_intercept() {
        assert!(intercept.abs() < 0.1);
    }
}