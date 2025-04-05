use crate::utility::principal_component_analysis::*;
use crate::utility::standardize;
use ndarray::{Array2, arr2};
use std::error::Error;

// Helper function for approximate equality checks
fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

#[test]
fn test_pca_new() {
    // We can only test the public API behavior, not internal fields
    let pca = PCA::new(2);

    // All getters should return errors when PCA is not fitted
    assert!(pca.get_components().is_err());
    assert!(pca.get_explained_variance().is_err());
    assert!(pca.get_explained_variance_ratio().is_err());
    assert!(pca.get_singular_values().is_err());
}

#[test]
fn test_pca_default() {
    let pca = PCA::default();

    // All getters should return errors when PCA is not fitted
    assert!(pca.get_components().is_err());
    assert!(pca.get_explained_variance().is_err());
    assert!(pca.get_explained_variance_ratio().is_err());
    assert!(pca.get_singular_values().is_err());
}

#[test]
fn test_standardize() {
    let data = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let standardized = standardize(&data);

    // Verify mean is approximately 0
    for col in 0..standardized.shape()[1] {
        let mean = standardized.column(col).mean().unwrap();
        assert!(approx_eq(mean, 0.0, 1e-10));
    }

    // Verify standard deviation is approximately 1
    for col in 0..standardized.shape()[1] {
        let col_data = standardized.column(col);
        let std_dev = (col_data.map(|x| x.powi(2)).sum() / col_data.len() as f64).sqrt();
        assert!(approx_eq(std_dev, 1.0, 1e-10));
    }
}

#[test]
fn test_fit_and_transform() -> Result<(), Box<dyn Error>> {
    let mut pca = PCA::new(2);
    let data = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]);

    pca.fit(&data)?;

    // After fitting, getters should return valid components
    let components = pca.get_components()?;
    assert_eq!(components.shape(), &[2, 3]);

    let explained_variance = pca.get_explained_variance()?;
    assert_eq!(explained_variance.len(), 2);

    let variance_ratio = pca.get_explained_variance_ratio()?;
    assert_eq!(variance_ratio.len(), 2);
    assert!(approx_eq(variance_ratio.sum(), 1.0, 1e-10));

    let singular_values = pca.get_singular_values()?;
    assert_eq!(singular_values.len(), 2);

    // Test transform functionality
    let transformed = pca.transform(&data)?;
    assert_eq!(transformed.shape(), &[4, 2]);

    Ok(())
}

#[test]
fn test_fit_transform() -> Result<(), Box<dyn Error>> {
    let mut pca = PCA::new(2);
    let data = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]);

    let transformed = pca.fit_transform(&data)?;

    // After fit_transform, getters should return valid components
    assert!(pca.get_components().is_ok());
    assert_eq!(transformed.shape(), &[4, 2]);

    Ok(())
}

#[test]
fn test_inverse_transform() -> Result<(), Box<dyn Error>> {
    let mut pca = PCA::new(2);
    let data = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]);

    let transformed = pca.fit_transform(&data)?;
    let reconstructed = pca.inverse_transform(&transformed)?;

    // Verify shape of reconstructed data matches original data
    assert_eq!(reconstructed.shape(), data.shape());

    // Since dimensionality reduction loses some information,
    // reconstructed data should be close to but not identical to the original data
    for i in 0..data.shape()[0] {
        for j in 0..data.shape()[1] {
            assert!((data[[i, j]] - reconstructed[[i, j]]).abs() < 1.0);
        }
    }

    Ok(())
}

#[test]
fn test_errors_when_not_fitted() {
    let pca = PCA::new(2);

    // Attempting to get components before fitting should return an error
    assert!(pca.get_components().is_err());
    assert!(pca.get_explained_variance().is_err());
    assert!(pca.get_explained_variance_ratio().is_err());
    assert!(pca.get_singular_values().is_err());

    // Attempting to transform data before fitting should return an error
    let data = arr2(&[[1.0, 2.0, 3.0]]);
    assert!(pca.transform(&data).is_err());
    assert!(pca.inverse_transform(&data).is_err());
}

#[test]
fn test_with_different_n_components() -> Result<(), Box<dyn Error>> {
    // Test different numbers of components
    for n_components in 1..=3 {
        let mut pca = PCA::new(n_components);
        let data = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        pca.fit(&data)?;

        // Verify component dimensions
        let components = pca.get_components()?;
        assert_eq!(components.shape(), &[n_components, 4]);

        // Verify dimensions of transformed data
        let transformed = pca.transform(&data)?;
        assert_eq!(transformed.shape(), &[4, n_components]);
    }

    Ok(())
}

#[test]
fn test_variance_explained_properties() -> Result<(), Box<dyn Error>> {
    let mut pca = PCA::new(2);
    let data = arr2(&[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]);

    pca.fit(&data)?;

    // Check that variance explained is non-negative
    let explained_variance = pca.get_explained_variance()?;
    for &variance in explained_variance.iter() {
        assert!(variance >= 0.0);
    }

    // Check that variance ratio sums to approximately 1
    let variance_ratio = pca.get_explained_variance_ratio()?;
    assert!(approx_eq(variance_ratio.sum(), 1.0, 1e-10));

    // Check that variance ratios are sorted in descending order
    for i in 1..variance_ratio.len() {
        assert!(variance_ratio[i - 1] >= variance_ratio[i]);
    }

    Ok(())
}

#[test]
fn test_pca_with_random_data() -> Result<(), Box<dyn Error>> {
    use rand::SeedableRng;
    use rand::distr::{Distribution, Uniform};
    use rand::rngs::StdRng;

    // Create a deterministic random number generator
    let mut rng = StdRng::seed_from_u64(42);

    // Create uniform distribution and handle the Result
    let dist = Uniform::new(0.0, 10.0).expect("Failed to create uniform distribution");

    let mut data = Array2::<f64>::zeros((100, 10));
    for i in 0..data.shape()[0] {
        for j in 0..data.shape()[1] {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }

    // Test PCA with different numbers of components
    for n_components in [2, 5, 8] {
        let mut pca = PCA::new(n_components);
        pca.fit(&data)?;

        // Check dimensions
        assert_eq!(pca.get_components()?.shape(), &[n_components, 10]);

        // Transform and reconstruct
        let transformed = pca.transform(&data)?;
        assert_eq!(transformed.shape(), &[100, n_components]);

        let reconstructed = pca.inverse_transform(&transformed)?;
        assert_eq!(reconstructed.shape(), data.shape());
    }

    Ok(())
}
