use ndarray::{Array1, Array2, ArrayView2};

/// Logistic Regression model implementation
///
/// This model uses gradient descent to train a binary classification logistic regression model.
///
/// # Features
///
/// * Supports adding an intercept term (bias)
/// * Uses gradient descent to optimize weights
/// * Provides probability predictions and class predictions
///
/// # Examples
///
/// ```
/// use rust_ai::machine_learning::logistic_regression::LogisticRegression;
/// use ndarray::{Array1, Array2};
///
/// // Create a logistic regression model
/// let mut model = LogisticRegression::default();
///
/// // Create some simple training data
/// // Two features: x1 and x2
/// // This data represents a simple logical AND function
/// let x_train = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0,  // [0,0] -> 0
///     0.0, 1.0,  // [0,1] -> 0
///     1.0, 0.0,  // [1,0] -> 0
///     1.0, 1.0,  // [1,1] -> 1
/// ]).unwrap();
///
/// let y_train = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
///
/// // Train the model
/// model.fit(&x_train, &y_train);
///
/// // Create test data
/// let x_test = Array2::from_shape_vec((2, 2), vec![
///     1.0, 0.0,  // Should predict 0
///     1.0, 1.0,  // Should predict 1
/// ]).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&x_test);
/// ```
pub struct LogisticRegression {
    /// Model weights, None before training
    weights: Option<Array1<f64>>,
    /// Whether to use intercept term (bias)
    fit_intercept: bool,
    /// Learning rate, controls gradient descent step size
    learning_rate: f64,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance, stops iteration when loss change is smaller than this value
    tolerance: f64,
}

impl LogisticRegression {
    /// Creates a new logistic regression model with specified parameters
    ///
    /// # Parameters
    ///
    /// * `fit_intercept` - Whether to add intercept term (bias)
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance, stops when loss change is below this value
    ///
    /// # Returns
    ///
    /// An untrained logistic regression model instance
    pub fn new(fit_intercept: bool,
               learning_rate: f64,
               max_iterations: usize,
               tolerance: f64,
    ) -> Self {
        LogisticRegression {
            weights: None,
            fit_intercept,
            learning_rate,
            max_iterations,
            tolerance,
        }
    }

    /// Creates a logistic regression model with default parameters
    ///
    /// Default parameters:
    /// * `fit_intercept`: true - Add intercept term
    /// * `learning_rate`: 0.01 - Learning rate
    /// * `max_iterations`: 100 - Maximum iterations
    /// * `tolerance`: 1e-4 - Convergence tolerance
    ///
    /// # Returns
    ///
    /// An untrained logistic regression model with default parameters
    pub fn default() -> Self {
        LogisticRegression {
            weights: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }

    /// Predicts probability scores for samples
    ///
    /// Uses the sigmoid function to convert linear predictions to probabilities between 0-1.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix view where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// A 1D array containing the probability of each sample belonging to the positive class
    ///
    /// # Panics
    ///
    /// If the model hasn't been trained yet (weights is None), it will panic
    fn predict_proba(&self, x: &ArrayView2<f64>) -> Array1<f64> {
        use crate::math::sigmoid;
        let weights = self.weights.as_ref().expect("Model not trained yet");

        let mut predictions = x.dot(weights);

        // Apply sigmoid activation function
        for val in predictions.iter_mut() {
            *val = sigmoid(*val);
        }

        predictions
    }

    /// Sets whether to fit the intercept term in the model
    ///
    /// If set to true, the model will include an intercept term.
    /// If set to false, the model will pass through the origin.
    ///
    /// # Parameters
    ///
    /// * `fit_intercept` - Boolean value indicating whether to fit the intercept
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to self for method chaining
    pub fn set_fit_intercept(&mut self, fit_intercept: bool) -> &mut Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Sets the learning rate for gradient descent
    ///
    /// The learning rate controls the step size in each iteration of gradient descent.
    /// A higher learning rate may lead to faster convergence or divergence,
    /// while a smaller learning rate may require more iterations but provide higher precision.
    ///
    /// # Parameters
    ///
    /// * `learning_rate` - A floating-point value, typically between 0 and 1
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to self for method chaining
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum number of iterations for gradient descent
    ///
    /// Limits the number of iterations in the gradient descent algorithm to prevent
    /// infinite loops in cases where convergence cannot be achieved.
    ///
    /// # Parameters
    ///
    /// * `max_iterations` - Maximum number of iterations to perform
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to self for method chaining
    pub fn set_max_iterations(&mut self, max_iterations: usize) -> &mut Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the convergence tolerance threshold
    ///
    /// The algorithm will stop iterating when the change in the loss function
    /// between consecutive iterations is less than this tolerance value.
    /// Smaller tolerance values typically lead to more accurate models but require more iterations.
    ///
    /// # Parameters
    ///
    /// * `tolerance` - A floating-point value representing the convergence threshold
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to self for method chaining
    pub fn set_tolerance(&mut self, tolerance: f64) -> &mut Self {
        self.tolerance = tolerance;
        self
    }

    /// Gets the current setting for fitting the intercept term
    ///
    /// # Returns
    ///
    /// Returns `true` if the model includes an intercept term, `false` otherwise
    pub fn fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Gets the current learning rate
    ///
    /// The learning rate controls the step size in each iteration of gradient descent.
    ///
    /// # Returns
    ///
    /// The current learning rate value
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Gets the maximum number of iterations
    ///
    /// # Returns
    ///
    /// The maximum number of iterations for the gradient descent algorithm
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Gets the convergence tolerance threshold
    ///
    /// The convergence tolerance is used to determine when to stop the training process.
    /// Training stops when the change in the loss function between consecutive iterations
    /// is less than this value.
    ///
    /// # Returns
    ///
    /// The current convergence tolerance value
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Trains the logistic regression model
    ///
    /// Uses gradient descent to minimize the logistic loss function.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix where each row is a sample and each column is a feature
    /// * `y` - Target variable containing 0 or 1 indicating sample class
    ///
    /// # Returns
    ///
    /// A mutable reference to the trained model, allowing for method chaining
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> &mut Self {
        use crate::math::sigmoid;
        let (n_samples, mut n_features) = x.dim();

        // If using intercept, add a column of ones
        let x_train = if self.fit_intercept {
            n_features += 1;
            let mut x_with_bias = Array2::ones((n_samples, n_features));
            for i in 0..n_samples {
                for j in 0..n_features - 1 {
                    x_with_bias[[i, j + 1]] = x[[i, j]];
                }
            }
            x_with_bias
        } else {
            x.clone()
        };

        // Initialize weights
        let mut weights = Array1::zeros(n_features);

        let mut prev_cost = f64::INFINITY;

        // Gradient descent optimization
        for _ in 0..self.max_iterations {
            let predictions = x_train.dot(&weights);
            let mut sigmoid_preds = Array1::zeros(n_samples);

            for i in 0..n_samples {
                sigmoid_preds[i] = sigmoid(predictions[i]);
            }

            // Calculate error
            let errors = &sigmoid_preds - y;

            // Calculate gradients
            let gradients = x_train.t().dot(&errors) / n_samples as f64;

            // Update weights
            weights = &weights - self.learning_rate * &gradients;

            // Calculate loss function using math module's logistic_loss
            let raw_preds: Vec<f64> = x_train.dot(&weights)
                .iter()
                .copied()
                .collect();

            let y_vec: Vec<f64> = y.iter().copied().collect();
            let cost = crate::math::logistic_loss(&raw_preds, &y_vec);

            // Check convergence
            if (prev_cost - cost).abs() < self.tolerance {
                break;
            }

            prev_cost = cost;
        }

        self.weights = Some(weights);
        self
    }

    /// Predicts class labels for samples
    ///
    /// Performs classification by applying a 0.5 threshold to probability values.
    ///
    /// # Parameters
    ///
    /// * `x` - Feature matrix where each row is a sample and each column is a feature
    ///
    /// # Returns
    ///
    /// A 1D array containing predicted class labels (0 or 1) for each sample
    ///
    /// # Panics
    ///
    /// If the model hasn't been trained, `predict_proba` will panic
    pub fn predict(&self, x: &Array2<f64>) -> Array1<i32> {
        let (n_samples, n_features) = x.dim();

        // Handle intercept term
        let x_test = if self.fit_intercept {
            let mut x_with_bias = Array2::ones((n_samples, n_features + 1));
            for i in 0..n_samples {
                for j in 0..n_features {
                    x_with_bias[[i, j + 1]] = x[[i, j]];
                }
            }
            x_with_bias
        } else {
            x.clone()
        };

        let probs = self.predict_proba(&x_test.view());

        // Apply threshold (0.5) for classification
        probs.mapv(|prob| if prob >= 0.5 { 1 } else { 0 })
    }

    /// Returns the model weights
    ///
    /// # Returns
    ///
    /// A reference to the weight array if the model has been trained, or None otherwise
    pub fn get_weights(&self) -> Option<&Array1<f64>> {
        self.weights.as_ref()
    }
}

/// Generates polynomial features from input features.
///
/// This function transforms the input feature matrix into a new feature matrix containing
/// polynomial combinations of the input features up to the specified degree.
///
/// # Examples
/// Following codes show how this function works with `LogisticRegression`:
/// ```
/// use ndarray::array;
/// use rust_ai::machine_learning::logistic_regression::{generate_polynomial_features, LogisticRegression};
///
/// // Example of using polynomial features with logistic regression
/// // Create a simple dataset for binary classification
/// let training_x = array![[0.5, 1.0], [1.0, 2.0], [1.5, 3.0], [2.0, 2.0], [2.5, 1.0]];
/// let training_y = array![0.0, 0.0, 1.0, 1.0, 0.0];
///
/// // Transform features to polynomial features
/// let poly_training_x = generate_polynomial_features(&training_x, 2);
///
/// // Create and train a logistic regression model with polynomial features
/// let mut model = LogisticRegression::default();
/// model.fit(&poly_training_x, &training_y);
/// ```
///
/// # Arguments
///
/// * `x` - Input feature matrix with shape (n_samples, n_features)
/// * `degree` - The maximum degree of polynomial features to generate
///
/// # Returns
///
/// A new feature matrix containing polynomial combinations of the input features
/// with shape (n_samples, n_output_features)
pub fn generate_polynomial_features(x: &Array2<f64>, degree: usize) -> Array2<f64> {
    let (n_samples, n_features) = x.dim();

    // Calculate the number of output features (including constant term)
    // Formula: C(n+d,d) = (n+d)!/(n!*d!) where n is feature count and d is degree
    let n_output_features = {
        let mut count = 1; // Constant term
        for d in 1..=degree {
            let mut term = 1;
            for i in 0..d {
                term = term * (n_features + i) / (i + 1);
            }
            count += term;
        }
        count
    };

    // Initialize result matrix with ones in the first column (constant term)
    let mut result = Array2::<f64>::ones((n_samples, n_output_features));

    // Add first-order features (original features)
    for i in 0..n_samples {
        for j in 0..n_features {
            result[[i, j+1]] = x[[i, j]];
        }
    }

    // If degree >= 2, add higher-order features
    if degree >= 2 {
        let mut col_idx = n_features + 1;

        // Define an inner recursive function to generate combinations
        fn add_combinations(
            x: &Array2<f64>,
            result: &mut Array2<f64>,
            col_idx: &mut usize,
            n_samples: usize,
            n_features: usize,
            degree: usize,
            current_degree: usize,
            start_feature: usize,
            combination: &mut Vec<usize>
        ) {
            // If we've reached the target degree, compute the feature value
            if current_degree == degree {
                for i in 0..n_samples {
                    let mut value = 1.0;
                    for &feat_idx in combination.iter() {
                        value *= x[[i, feat_idx]];
                    }
                    result[[i, *col_idx]] = value;
                }
                *col_idx += 1;
                return;
            }

            // Recursively build combinations
            for j in start_feature..n_features {
                combination.push(j);
                add_combinations(
                    x, result, col_idx, n_samples, n_features,
                    degree, current_degree + 1, j, combination
                );
                combination.pop();
            }
        }

        // Generate combinations for each degree
        for d in 2..=degree {
            add_combinations(x, &mut result, &mut col_idx, n_samples, n_features, d, 0, 0, &mut vec![]);
        }
    }

    result
}