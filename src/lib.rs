pub mod math {
    /// Calculate the sum of squared errors
    ///
    /// # Parameters
    /// * `predicted` - Predicted values vector (y')
    /// * `actual` - Actual values vector (y)
    ///
    /// # Returns
    /// * Sum of squared errors sum((predicted_i - actual_i)^2)
    ///
    /// # Panic
    /// The function will panic if the vectors have different lengths
    /// # Example
    /// ```
    /// use rust_machine_learning::math::sum_of_squared_errors;
    ///
    /// let predicted = vec![1.0, 2.0, 3.0, 4.0];
    /// let actual = vec![1.2, 1.8, 3.3, 3.9];
    ///
    /// // Calculate the sum of squared errors
    /// let sse = sum_of_squared_errors(&predicted, &actual);
    /// ```

    pub fn sum_of_squared_errors(predicted: &[f64], actual: &[f64]) -> f64 {
        // Ensure both vectors have the same length
        assert_eq!(predicted.len(), actual.len(), "Vectors must have the same length");
        // Calculate the sum of squared errors
        predicted.iter()
            .zip(actual.iter())
            .map(|(p, a)| {
                let diff = p - a;
                diff * diff // Calculate square
            })
            .sum()
    }

    /// Calculates the Sum of Square Total
    ///
    /// SST measures the total variability in the data, computed as the sum of squared
    /// differences between each actual value and the mean of all values.
    ///
    /// # Parameters
    /// * `values` - A slice of observed values
    ///
    /// # Returns
    /// * The Sum of Square Total (SST)
    ///
    /// # Examples
    /// ```
    ///  use rust_machine_learning::math::sum_of_square_total;
    /// let values = vec![2.0, 4.0, 6.0, 8.0];
    /// let sst = sum_of_square_total(&values);
    /// ```
    pub fn sum_of_square_total(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        // Calculate the mean
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        // Calculate sum of squared differences from the mean
        values.iter()
            .map(|&value| (value - mean).powi(2))
            .sum()
    }

    /// Calculate the R-squared score
    ///
    /// R² = 1 - (SSE / SST)
    ///
    /// # Arguments
    /// * `predicted` - Array of predicted values
    /// * `actual` - Array of actual values
    ///
    /// # Returns
    /// * `f64` - R-squared value, typically ranges from 0 to 1
    ///
    /// # Notes
    /// - Returns 0 if SST is 0 (when all actual values are identical)
    /// - R-squared can theoretically be negative, indicating that the model performs worse than simply predicting the mean
    /// - A value close to 1 indicates a good fit
    /// # Example
    /// ```
    /// use rust_machine_learning::math::{r2_score, sum_of_squared_errors, sum_of_square_total};
    ///
    /// let actual = vec![3.0, 2.0, 5.0, 7.0, 9.0];
    /// let predicted = vec![2.8, 1.9, 5.2, 7.5, 8.9];
    ///
    /// // Calculate R-squared
    /// let r2 = r2_score(&predicted, &actual);
    /// ```

    pub fn r2_score(predicted: &[f64], actual: &[f64]) -> f64 {
        let sse = sum_of_squared_errors(predicted, actual);
        let sst = sum_of_square_total(actual);

        // Prevent division by zero (when all actual values are identical)
        if sst == 0.0 {
            return 0.0;
        }

        1.0 - (sse / sst)
    }
}

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
        }
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

        // Gradient descent iterations
        for iteration in 0..self.max_iterations {
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
