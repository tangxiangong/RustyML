use ndarray::{Array1, Array2};

pub mod principal_component_analysis;
pub mod train_test_split;


/// Standardizes data to have zero mean and unit variance
///
/// This function transforms input data by subtracting the mean and dividing
/// by the standard deviation for each feature, resulting in standardized data
/// where each feature has a mean of 0 and a standard deviation of 1.
///
/// # Parameters
///
/// * `x` - A 2D array where rows represent samples and columns represent features
///
/// # Returns
///
/// * `Array2<f64>` - A standardized 2D array with the same shape as the input
///
/// # Implementation Details
///
/// - Calculates mean and standard deviation for each feature column
/// - Handles cases where standard deviation is zero (or very small) by setting it to 1.0
/// - Applies the z-score transformation: (x - mean) / std_dev
pub fn standardize(x: &Array2<f64>) -> Array2<f64> {
    use crate::math::standard_deviation;

    let n_samples = x.nrows();
    let n_features = x.ncols();

    // Calculate mean for each column
    let mut means = Array1::<f64>::zeros(n_features);
    for i in 0..n_features {
        means[i] = x.column(i).mean().unwrap_or(0.0);
    }

    let mut stds = Array1::<f64>::zeros(n_features);
    for i in 0..n_features {
        let col_vec: Vec<f64> = x.column(i).iter().cloned().collect();
        stds[i] = standard_deviation(&col_vec);

        // Handle case where standard deviation is zero
        if stds[i] < 1e-10 {
            stds[i] = 1.0;
        }
    }

    // Standardize data
    let mut x_std = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x_std[[i, j]] = (x[[i, j]] - means[j]) / stds[j];
        }
    }

    x_std
}