pub mod math {
    /// Calculate the sum of squared errors between two vectors: sum((predicted - actual)^2)
    ///
    /// # Parameters
    /// * `predicted` - Predicted values vector (y')
    /// * `actual` - Actual values vector (y)
    ///
    /// # Return Value
    /// * Sum of squared errors sum((predicted_i - actual_i)^2)
    ///
    /// # Panic
    /// The function will panic if the vectors have different lengths
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

/// Calculates the Sum of Square Total (SST)
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
///  use rust_machine_learning::sum_of_square_total;
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
/// * `f64` - R-squared value, typically ranges from 0 to 1, with values closer to 1 indicating better fit
///
/// # Notes
/// - Returns 0 if SST is 0 (when all actual values are identical)
/// - R-squared can theoretically be negative, indicating that the model performs worse than simply predicting the mean
pub fn r2_score(predicted: &[f64], actual: &[f64]) -> f64 {
    let sse = sum_of_squared_errors(predicted, actual);
    let sst = sum_of_square_total(actual);

    // Prevent division by zero (when all actual values are identical)
    if sst == 0.0 {
        return 0.0;
    }

    1.0 - (sse / sst)
}

pub struct LinearRegression {
    /// Coefficients (slopes)
    pub coefficients: Option<Vec<f64>>,
    /// Intercept
    pub intercept: Option<f64>,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Learning rate
    pub learning_rate: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
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

            // Calculate mean squared error using the provided function
            let sse = sum_of_squared_errors(&predictions, y);
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
