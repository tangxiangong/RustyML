use crate::machine_learning::meanshift::*;
use ndarray::{Array2, arr2};
use rand::SeedableRng;
use rand::prelude::*;

fn create_test_data() -> Array2<f64> {
    // Create a simple test dataset with three clusters
    let mut rng = StdRng::seed_from_u64(42);

    let n_samples = 300;
    let mut data = Array2::zeros((n_samples, 2));

    // First cluster
    for i in 0..100 {
        data[[i, 0]] = rng.random_range(-20.0..=-10.0);
        data[[i, 1]] = rng.random_range(-20.0..=-10.0);
    }

    // Second cluster
    for i in 100..200 {
        data[[i, 0]] = rng.random_range(0.0..=10.0);
        data[[i, 1]] = rng.random_range(0.0..=10.0);
    }

    // Third cluster
    for i in 200..300 {
        data[[i, 0]] = rng.random_range(20.0..=30.0);
        data[[i, 1]] = rng.random_range(20.0..=30.0);
    }

    data
}

#[test]
fn test_meanshift_default() {
    let ms = MeanShift::default();
    assert_eq!(ms.get_bandwidth(), 1.0);
    assert_eq!(ms.get_max_iter(), 300);
    assert_eq!(ms.get_tol(), 1e-3);
    assert_eq!(ms.get_bin_seeding(), false);
    assert_eq!(ms.get_cluster_all(), true);
}

#[test]
fn test_meanshift_new() {
    let ms = MeanShift::new(2.0, Some(200), Some(1e-4), Some(true), Some(true));
    assert_eq!(ms.get_bandwidth(), 2.0);
    assert_eq!(ms.get_max_iter(), 200);
    assert_eq!(ms.get_tol(), 1e-4);
    assert_eq!(ms.get_bin_seeding(), true);
    assert_eq!(ms.get_cluster_all(), true);
}

#[test]
fn test_meanshift_getters_before_fit() {
    let ms = MeanShift::default();

    assert!(ms.get_cluster_centers().is_err());
    assert!(ms.get_labels().is_err());
    assert!(ms.get_n_iter().is_err());
    assert!(ms.get_n_samples_per_center().is_err());
}

#[test]
fn test_meanshift_fit() {
    let data = create_test_data();

    let mut ms = MeanShift::new(2.0, None, None, None, Some(true));
    ms.fit(&data).unwrap();

    // Check that all attributes are accessible after fitting
    assert!(ms.get_cluster_centers().is_ok());
    assert!(ms.get_labels().is_ok());
    assert!(ms.get_n_iter().is_ok());
    assert!(ms.get_n_samples_per_center().is_ok());

    // Check the shape of cluster centers
    let centers = ms.get_cluster_centers().unwrap();
    assert_eq!(centers.dim().1, 2); // Number of features should be 2

    // Check labels
    let labels = ms.get_labels().unwrap();
    assert_eq!(labels.len(), data.dim().0); // Number of labels should equal number of samples
}

#[test]
fn test_meanshift_predict() {
    let data = create_test_data();

    let mut ms = MeanShift::new(2.0, None, None, None, Some(true));
    ms.fit(&data).unwrap();

    // Create some new test points
    let test_points = arr2(&[
        [-15.0, -15.0], // Should belong to the first cluster
        [5.0, 5.0],     // Should belong to the second cluster
        [25.0, 25.0],   // Should belong to the third cluster
    ]);

    let predictions = ms.predict(&test_points).unwrap();
    assert_eq!(predictions.len(), 3);

    // Check that the predicted labels match the expected labels
    // Note: Since the specific label values are algorithm-determined,
    // we don't check the exact values, but rather that different points
    // are assigned to different clusters
    assert_ne!(predictions[0], predictions[1]);
    assert_ne!(predictions[1], predictions[2]);
    assert_ne!(predictions[0], predictions[2]);
}

#[test]
fn test_meanshift_fit_predict() {
    let data = create_test_data();

    let mut ms = MeanShift::new(2.0, None, None, None, Some(true));
    let labels = ms.fit_predict(&data).unwrap();

    assert_eq!(labels.len(), data.dim().0);

    // Check that there are multiple different labels (should have at least 3 clusters)
    let mut unique_labels = std::collections::HashSet::new();
    for label in labels.iter() {
        unique_labels.insert(*label);
    }

    assert!(unique_labels.len() >= 3);
}

#[test]
fn test_bin_seeding() {
    let data = create_test_data();

    // Compare with and without bin_seeding
    let mut ms1 = MeanShift::new(2.0, None, None, Some(false), None);
    let mut ms2 = MeanShift::new(2.0, None, None, Some(true), None);

    ms1.fit(&data).unwrap();
    ms2.fit(&data).unwrap();

    // Both methods should fit successfully
    assert!(ms1.get_cluster_centers().is_ok());
    assert!(ms2.get_cluster_centers().is_ok());

    // With bin_seeding we typically expect fewer initial points, but we can't make strong assertions about the final number of clusters
}

#[test]
fn test_estimate_bandwidth() {
    let data = create_test_data();

    // Default parameters
    let bw1 = estimate_bandwidth(&data, None, None, None);
    assert!(bw1 > 0.0);

    // Specified quantile
    let bw2 = estimate_bandwidth(&data, Some(0.3), None, None);
    assert!(bw2 > 0.0);

    // Specified n_samples
    let bw3 = estimate_bandwidth(&data, None, Some(50), None);
    assert!(bw3 > 0.0);

    // Specified random_state
    let bw4 = estimate_bandwidth(&data, None, None, Some(42));
    assert!(bw4 > 0.0);

    // Using the same random seed should yield the same result
    let bw5 = estimate_bandwidth(&data, None, None, Some(42));
    assert_eq!(bw4, bw5);
}

#[test]
fn test_cluster_all_parameter() {
    let data = create_test_data();

    // With cluster_all = false, some points may not be assigned to clusters
    let mut ms1 = MeanShift::new(1.0, None, None, None, Some(false));
    ms1.fit(&data).unwrap();
    let labels1 = ms1.get_labels().unwrap();

    // With cluster_all = true, all points should be assigned to clusters
    let mut ms2 = MeanShift::new(1.0, None, None, None, Some(true));
    ms2.fit(&data).unwrap();
    let labels2 = ms2.get_labels().unwrap();

    // Both should have the same number of labels
    assert_eq!(labels1.len(), labels2.len());
}

#[test]
fn test_fit_with_max_iterations() {
    let data = create_test_data();

    // Set a very low max_iter to force early stopping
    let mut ms = MeanShift::new(2.0, Some(1), None, None, None);
    ms.fit(&data).unwrap();

    // Should complete successfully and n_iter should be 1
    assert_eq!(ms.get_n_iter().unwrap(), 1);
}
