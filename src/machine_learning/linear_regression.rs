use crate::{math, ModelError};
use ndarray::{Array1, Array2};

/// # Linear Regression model implementation
///
/// Trains a simple linear regression model using gradient descent algorithm. This implementation
/// supports multivariate regression, optional intercept term, and allows adjustment of learning rate,
/// maximum iterations, and convergence tolerance.
///
/// ## Fields
///
/// * `coefficients` - Model coefficients (slopes), None before training
/// * `intercept` - Model intercept, None before training
/// * `fit_intercept` - Whether to include an intercept term in the model
/// * `learning_rate` - Learning rate for gradient descent
/// * `max_iter` - Maximum number of iterations for gradient descent
/// * `tol` - Convergence tolerance
/// * `n_iter` - Number of iterations the algorithm ran for after fitting
///
/// ## Examples
///
/// ```
/// use rust_ai::machine_learning::linear_regression::LinearRegression;
/// use ndarray::{Array1, Array2, array};
///
/// // Create a linear regression model
/// let mut model = LinearRegression::new(true, 0.01, 1000, 1e-6);
///
/// // Prepare training data
/// let raw_x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
/// let raw_y = vec![6.0, 9.0, 12.0];
///
/// // Convert Vec to ndarray types
/// let x = Array2::from_shape_vec((3, 2), raw_x.into_iter().flatten().collect()).unwrap();
/// let y = Array1::from_vec(raw_y);
///
/// // Train the model
/// model.fit(&x, &y);
///
/// // Make predictions
/// let new_data = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
/// let predictions = model.predict(&new_data);
///
/// // Since Clone is implemented, the model can be easily cloned
/// let model_copy = model.clone();
///
/// // Since Debug is implemented, detailed model information can be printed
/// println!("{:?}", model);
/// ```
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
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Number of iterations the algorithm ran for after fitting
    n_iter: Option<usize>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-5,
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
            max_iter: max_iterations,
            tol: tolerance,
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
    pub fn get_max_iter(&self) -> usize {
        self.max_iter
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
    pub fn get_tol(&self) -> f64 {
        self.tol
    }

    /// Returns the model coefficients if the model has been fitted
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<f64>)` - A vector of model coefficients
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
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
    /// - `Ok(f64)` - The intercept value
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
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
    /// - `Ok(usize)` - The number of iterations if the model has been fitted
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    pub fn get_n_iter(&self) -> Result<usize, ModelError> {
        match self.n_iter {
            Some(n_iter) => Ok(n_iter),
            None => Err(ModelError::NotFitted),
        }
    }

    /// Fits the linear regression model using gradient descent
    ///
    /// # Parameters
    /// * `x` - Feature matrix, each row is a sample, each column is a feature
    /// * `y` - Target variable vector
    ///
    /// # Return Value
    /// * `&mut self` - Returns mutable reference to self for method chaining
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> &mut Self {
        // Ensure x and y have the same number of samples
        assert!(x.nrows() > 0, "x cannot be empty");
        assert!(y.len() > 0, "y cannot be empty");
        assert_eq!(x.nrows(), y.len(), "x and y must have the same number of samples");

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize parameters
        let mut weights = Array1::<f64>::zeros(n_features); // Initialize weights to zero
        let mut intercept = 0.0;                 // Initialize intercept to zero

        let mut prev_cost = f64::INFINITY;
        let mut final_cost = prev_cost;

        let mut n_iter = 0;

        // Gradient descent iterations
        while n_iter < self.max_iter {
            n_iter += 1;

            // Calculate predictions
            let mut predictions = Array1::<f64>::zeros(n_samples);
            for i in 0..n_samples {
                let mut pred = 0.0;
                for j in 0..n_features {
                    pred += x[[i, j]] * weights[j];
                }
                if self.fit_intercept {
                    pred += intercept;
                }
                predictions[i] = pred;
            }

            // Calculate mean squared error
            let sse = math::sum_of_squared_errors(
                predictions.as_slice().expect("predictions should be contiguous"),
                y.as_slice().expect("y should be contiguous")
            );

            let cost = sse / (2.0 * n_samples as f64); // Mean squared error divided by 2
            final_cost = cost;

            // Calculate gradients
            let mut gradients = Array1::<f64>::zeros(n_features);
            let mut intercept_gradient = 0.0;

            for i in 0..n_samples {
                let error = predictions[i] - y[i];

                // Update gradients for weights
                for j in 0..n_features {
                    gradients[j] += error * x[[i, j]];
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
            if (prev_cost - cost).abs() < self.tol {
                break;
            }

            prev_cost = cost;
        }

        // Save training results
        self.coefficients = Some(weights.to_vec());
        self.intercept = Some(if self.fit_intercept { intercept } else { 0.0 });
        self.n_iter = Some(n_iter);

        // print training info
        println!("Linear regression model training finished at iteration {}, cost: {}",
                 n_iter, final_cost);

        self
    }

    /// Makes predictions using the trained model
    ///
    /// # Parameters
    /// * `x` - Prediction data, each row is a sample, each column is a feature
    ///
    /// # Return Value
    /// - `Ok(Vec<f64>)` - A vector of predictions
    /// - `Err(ModelError::NotFitted)` - If the model has not been fitted yet
    /// - `Err(ModelError::InputValidationError)` - If number of features does not match training data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Vec<f64>, ModelError> {
        if self.coefficients.is_none() {
            return Err(ModelError::NotFitted);
        }

        let coeffs = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        let mut predictions = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            if x.ncols() != coeffs.len() {
                return Err(ModelError::InputValidationError(
                    "Number of features does not match training data"
                ));
            }

            let mut prediction = intercept;
            for j in 0..x.ncols() {
                prediction += x[[i, j]] * coeffs[j];
            }

            predictions.push(prediction);
        }

        Ok(predictions)
    }

    /// Fits the model to the training data and then makes predictions on the same data.
    ///
    /// This is a convenience method that combines the `fit` and `predict` methods into one call.
    ///
    /// # Arguments
    ///
    /// * `x` - The input features matrix where each inner vector represents a training example
    /// * `y` - The target values corresponding to each training example
    ///
    /// # Returns
    ///
    /// A Result containing either:
    /// * `Vec<f64>` - The predicted values for the input data
    pub fn fit_predict(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Vec<f64> {
        self.fit(x, y);
        self.predict(x).unwrap()
    }
}