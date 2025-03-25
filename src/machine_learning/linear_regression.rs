use crate::{math, ModelError};

/// Linear Regression model implementation
///
/// Trains a simple linear regression model using gradient descent algorithm. This implementation
/// supports multivariate regression, optional intercept term, and allows adjustment of learning rate,
/// maximum iterations, and convergence tolerance.
///
/// # Examples
///
/// ```
/// use rust_ai::machine_learning::linear_regression::LinearRegression;
///
/// // Create a linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6);
///
/// // Prepare training data
/// let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
/// let y = vec![6.0, 9.0, 12.0];
///
/// // Train the model
/// model.fit(&x, &y);
///
/// // Make predictions
/// let new_data = vec![vec![4.0, 5.0]];
/// let predictions = model.predict(&new_data);
///
/// // Since Clone is implemented, the model can be easily cloned
/// let model_copy = model.clone();
///
/// // Since Debug is implemented, detailed model information can be printed
/// println!("{:?}", model);
/// ```
///
/// # Parameters
///
/// * `coefficients` - Model coefficients (slopes), None before training
/// * `intercept` - Model intercept, None before training
/// * `fit_intercept` - Whether to include an intercept term in the model
/// * `learning_rate` - Learning rate for gradient descent
/// * `max_iterations` - Maximum number of iterations for gradient descent
/// * `tolerance` - Convergence tolerance
/// * `n_iter` - Number of iterations the algorithm ran for after fitting
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Coefficients (slopes)
    coefficients: Option<Vec<f64>>,
    /// Intercept
    intercept: Option<f64>,
    /// Whether to fit an intercept
    fit_intercept: bool,
    /// Learning rate
    learning_rate: f64,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    n_iter: Option<usize>,
}

impl Default for LinearRegression {
    /// implement `Default` trait, write `LinearRegression::default()` to use default arguments
    fn default() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iterations: 1000,
            tolerance: 1e-5,
            n_iter: None,
        }
    }
}

impl LinearRegression {
    /// Creates a new linear regression model with custom parameters
    pub fn new(
        fit_intercept: bool,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        LinearRegression {
            coefficients: None,
            intercept: None,
            fit_intercept,
            learning_rate,
            max_iterations,
            tolerance,
            n_iter: None,
        }
    }

    /// Gets the current setting for fitting the intercept term
    ///
    /// # Returns
    ///
    /// Returns `true` if the model includes an intercept term, `false` otherwise
    pub fn get_fit_intercept(&self) -> bool {
        self.fit_intercept
    }

    /// Gets the current learning rate
    ///
    /// The learning rate controls the step size in each iteration of gradient descent.
    ///
    /// # Returns
    ///
    /// The current learning rate value
    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Gets the maximum number of iterations
    ///
    /// # Returns
    ///
    /// The maximum number of iterations for the gradient descent algorithm
    pub fn get_max_iterations(&self) -> usize {
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
    pub fn get_tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Returns the model coefficients if the model has been fitted
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<f64>)` - A vector of model coefficients
    /// * `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_ai::machine_learning::linear_regression::LinearRegression;
    /// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-5);
    /// let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y = vec![6.0, 9.0, 12.0];
    ///
    /// // ... fit the model ...
    /// model.fit(&x, &y);
    /// let coefficients = model.get_coefficients().expect("Model should be fitted");
    /// ```
    pub fn get_coefficients(&self) -> Result<Vec<f64>, ModelError> {
        match &self.coefficients {
            Some(coeffs) => Ok(coeffs.clone()),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the intercept term if the model has been fitted
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - The intercept value
    /// * `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_ai::machine_learning::linear_regression::LinearRegression;
    /// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-5);
    /// let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
    /// let y = vec![6.0, 9.0, 12.0];
    ///
    /// // ... fit the model ...
    /// model.fit(&x, &y);
    /// let intercept = model.get_intercept().expect("Model should be fitted");
    /// ```
    pub fn get_intercept(&self) -> Result<f64, ModelError> {
        match self.intercept {
            Some(intercept) => Ok(intercept),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Returns the actual number of iterations performed during the last model fitting.
    ///
    /// # Returns
    ///
    /// * `Some(usize)` - The number of iterations if the model has been fitted
    /// * `None` - If the model has not been fitted yet
    pub fn get_n_iter(&self) -> Option<usize> {
        self.n_iter
    }

    /// Fits the linear regression model using gradient descent
    ///
    /// # Parameters
    /// * `x` - Feature matrix, each row is a sample, each column is a feature
    /// * `y` - Target variable vector
    ///
    /// # Return Value
    /// * `&mut self` - Returns mutable reference to self for method chaining
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> &mut Self {
        // Ensure x and y have the same number of samples
        assert!(!x.is_empty(), "x cannot be empty");
        assert!(!y.is_empty(), "y cannot be empty");
        assert_eq!(x.len(), y.len(), "x and y must have the same number of samples");

        let n_samples = x.len();
        let n_features = x[0].len();

        // Initialize parameters
        let mut weights = vec![0.0; n_features]; // Initialize weights to zero
        let mut intercept = 0.0;                 // Initialize intercept to zero

        let mut prev_cost = f64::INFINITY;
        
        let mut n_iter = 0;

        // Gradient descent iterations
        for iteration in 0..self.max_iterations {
            n_iter += 1;
            
            // Calculate predictions
            let mut predictions = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let mut pred = 0.0;
                for j in 0..n_features {
                    pred += x[i][j] * weights[j];
                }
                if self.fit_intercept {
                    pred += intercept;
                }
                predictions.push(pred);
            }

            // Calculate mean squared error
            let sse = math::sum_of_squared_errors(&predictions, y);
            let cost = sse / (2.0 * n_samples as f64); // Mean squared error divided by 2

            // Calculate gradients
            let mut gradients = vec![0.0; n_features];
            let mut intercept_gradient = 0.0;

            for i in 0..n_samples {
                let error = predictions[i] - y[i];

                // Update gradients for weights
                for j in 0..n_features {
                    gradients[j] += error * x[i][j];
                }

                // Update gradient for intercept
                if self.fit_intercept {
                    intercept_gradient += error;
                }
            }

            // Normalize gradients (divide by number of samples)
            for j in 0..n_features {
                gradients[j] /= n_samples as f64;
            }
            intercept_gradient /= n_samples as f64;

            // Update parameters
            for j in 0..n_features {
                weights[j] -= self.learning_rate * gradients[j];
            }
            if self.fit_intercept {
                intercept -= self.learning_rate * intercept_gradient;
            }

            // Check convergence
            if (prev_cost - cost).abs() < self.tolerance {
                println!("Model converged at iteration {}, cost: {}", iteration, cost);
                break;
            }

            prev_cost = cost;

            // Optional: Print progress at intervals
            if iteration % 100 == 0 {
                println!("Iteration {}: cost = {}", iteration, cost);
            }
        }

        // Save training results
        self.coefficients = Some(weights);
        self.intercept = Some(if self.fit_intercept { intercept } else { 0.0 });
        self.n_iter = Some(n_iter);

        self
    }

    /// Makes predictions using the trained model
    ///
    /// # Parameters
    /// * `x` - Prediction data, each row is a sample, each column is a feature
    ///
    /// # Return Value
    /// * Vector of predictions
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        assert!(self.coefficients.is_some(), "Model has not been trained yet");

        let coeffs = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        let mut predictions = Vec::with_capacity(x.len());
        for sample in x {
            assert_eq!(sample.len(), coeffs.len(), "Number of features does not match training data");

            let mut prediction = intercept;
            for (i, &feature_val) in sample.iter().enumerate() {
                prediction += feature_val * coeffs[i];
            }

            predictions.push(prediction);
        }

        predictions
    }
}