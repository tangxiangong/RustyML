use crate::machine_learning::decision_tree::*;
use ndarray::{Array1, Array2, arr1, arr2};

#[test]
fn test_decision_tree_new() {
    // Test creating a decision tree with default parameters
    let dt = DecisionTree::new(Algorithm::CART, true, None);
    assert!(matches!(dt.get_algorithm(), Algorithm::CART));
    assert!(dt.get_is_classifier());

    // Test creating a decision tree with custom parameters
    let params = DecisionTreeParams {
        max_depth: Some(5),
        min_samples_split: 10,
        min_samples_leaf: 5,
        min_impurity_decrease: 0.1,
        random_state: Some(42),
    };
    let dt = DecisionTree::new(Algorithm::CART, false, Some(params));
    assert!(matches!(dt.get_algorithm(), Algorithm::CART));
    assert!(!dt.get_is_classifier());
    assert_eq!(dt.get_params().max_depth, Some(5));
    assert_eq!(dt.get_params().min_samples_split, 10);
    assert_eq!(dt.get_params().min_samples_leaf, 5);
    assert_eq!(dt.get_params().min_impurity_decrease, 0.1);
    assert_eq!(dt.get_params().random_state, Some(42));
}

#[test]
fn test_decision_tree_params_default() {
    let params = DecisionTreeParams::default();
    assert_eq!(params.max_depth, None);
    assert_eq!(params.min_samples_split, 2);
    assert_eq!(params.min_samples_leaf, 1);
    assert!(params.min_impurity_decrease < 1e-6);
    assert_eq!(params.random_state, None);
}

#[test]
fn test_fit_predict_classifier() {
    // Create a simple binary classification dataset
    let x = arr2(&[
        [2.0, 2.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [3.0, 3.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ]);
    let y = arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

    // Create classifier
    let mut dt = DecisionTree::new(Algorithm::CART, true, None);

    // Train the model
    dt.fit(&x, &y).unwrap();

    // Ensure the model is trained correctly
    assert_eq!(dt.get_n_features(), 2);
    assert!(dt.get_root().is_ok());

    // Predict
    let predictions = dt.predict(&x).unwrap();
    assert_eq!(predictions.len(), 6);

    // Check prediction accuracy
    for i in 0..6 {
        assert_eq!(predictions[i], y[i]);
    }

    // Test single sample prediction
    let sample = &[2.0, 2.0];
    let pred = dt.predict_one(sample).unwrap();
    assert_eq!(pred, 0.0);

    let sample = &[1.0, 1.0];
    let pred = dt.predict_one(sample).unwrap();
    assert_eq!(pred, 1.0);

    // print tree
    println!("{}", dt.generate_tree_structure().unwrap());
}

#[test]
fn test_fit_predict_regressor() {
    // Create a simple regression dataset
    let x = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
    let y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Create regressor
    let mut dt = DecisionTree::new(Algorithm::CART, false, None);

    // Train the model
    dt.fit(&x, &y).unwrap();

    // Ensure the model is trained correctly
    assert_eq!(dt.get_n_features(), 1);
    assert!(dt.get_root().is_ok());

    // Predict
    let predictions = dt.predict(&x).unwrap();
    assert_eq!(predictions.len(), 6);

    // Check prediction accuracy (regression might not be exact)
    for i in 0..6 {
        assert!((predictions[i] - y[i]).abs() < 0.5);
    }

    // print tree
    println!("{}", dt.generate_tree_structure().unwrap());
}

#[test]
fn test_predict_proba() {
    // Create a simple multi-class dataset
    let x = arr2(&[
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]);
    let y = arr1(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

    // Create classifier
    let mut dt = DecisionTree::new(Algorithm::CART, true, None);

    // Train the model
    dt.fit(&x, &y).unwrap();

    // Test class probability prediction
    let prob = dt.predict_proba(&x).unwrap();

    // Check the shape of the probability matrix
    assert_eq!(prob.shape()[0], 6); // 6 samples
    assert_eq!(prob.shape()[1], 3); // 3 classes

    // Check that the sum of probabilities for each sample is 1
    for i in 0..6 {
        let row_sum: f64 = prob.row(i).sum();
        assert!((row_sum - 1.0).abs() < 1e-5);
    }

    // Test single sample probability prediction
    let sample = &[1.0, 1.0];
    let probs = dt.predict_proba_one(sample).unwrap();
    assert_eq!(probs.len(), 3);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-5);
}

#[test]
fn test_different_algorithms() {
    let x = arr2(&[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]);
    let y = arr1(&[0.0, 0.0, 1.0, 1.0]);

    // Test ID3 algorithm
    let mut dt_id3 = DecisionTree::new(Algorithm::ID3, true, None);
    dt_id3.fit(&x, &y).unwrap();
    let pred_id3 = dt_id3.predict(&x).unwrap();

    // Test C4.5 algorithm
    let mut dt_c45 = DecisionTree::new(Algorithm::C45, true, None);
    dt_c45.fit(&x, &y).unwrap();
    let pred_c45 = dt_c45.predict(&x).unwrap();

    // Test CART algorithm
    let mut dt_cart = DecisionTree::new(Algorithm::CART, true, None);
    dt_cart.fit(&x, &y).unwrap();
    let pred_cart = dt_cart.predict(&x).unwrap();

    // Verify that all algorithms correctly predict the training data
    for i in 0..4 {
        assert_eq!(pred_id3[i], y[i]);
        assert_eq!(pred_c45[i], y[i]);
        assert_eq!(pred_cart[i], y[i]);
    }
}

#[test]
fn test_max_depth_parameter() {
    let x = arr2(&[
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [4.0, 4.0],
    ]);
    let y = arr1(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

    let params_depth1 = DecisionTreeParams {
        max_depth: Some(1),
        ..DecisionTreeParams::default()
    };

    let mut dt_limited = DecisionTree::new(Algorithm::CART, true, Some(params_depth1));
    dt_limited.fit(&x, &y).unwrap();

    // Check if tree depth is limited
    // We can't directly check the depth, but we can infer that if the model
    // has only one level of depth, its prediction ability will be limited

    let _predictions = dt_limited.predict(&x).unwrap();

    // Create an unlimited depth tree for comparison
    let mut dt_unlimited = DecisionTree::new(Algorithm::CART, true, None);
    dt_unlimited.fit(&x, &y).unwrap();

    let predictions_unlimited = dt_unlimited.predict(&x).unwrap();

    // The unlimited depth tree should perfectly fit the training data
    for i in 0..8 {
        assert_eq!(predictions_unlimited[i], y[i]);
    }
}

#[test]
fn test_error_handling() {
    // Create an untrained tree
    let dt = DecisionTree::new(Algorithm::CART, true, None);

    // Trying to get the root node should fail
    assert!(dt.get_root().is_err());

    // Trying to get the number of classes should also fail
    assert!(dt.get_n_classes().is_err());

    // Trying to predict should fail
    let x = arr2(&[[1.0, 2.0]]);
    assert!(dt.predict(&x).is_err());

    // Trying to get a single prediction should fail
    let sample = &[1.0, 2.0];
    assert!(dt.predict_one(sample).is_err());

    // Trying to get class probabilities should fail
    assert!(dt.predict_proba(&x).is_err());
    assert!(dt.predict_proba_one(sample).is_err());
}

#[test]
fn test_node_creation() {
    // Test leaf node creation
    let leaf = Node::new_leaf(1.5, Some(0), Some(vec![0.8, 0.2]));
    match leaf.node_type {
        NodeType::Leaf {
            value,
            class,
            probabilities,
        } => {
            assert_eq!(value, 1.5);
            assert_eq!(class, Some(0));
            assert_eq!(probabilities, Some(vec![0.8, 0.2]));
        }
        _ => panic!("Expected a leaf node"),
    }

    // Test internal node creation
    let internal = Node::new_internal(1, 0.5);
    match internal.node_type {
        NodeType::Internal {
            feature_index,
            threshold,
            categories,
        } => {
            assert_eq!(feature_index, 1);
            assert_eq!(threshold, 0.5);
            assert_eq!(categories, None);
        }
        _ => panic!("Expected an internal node"),
    }

    // Test categorical node creation
    let categorical = Node::new_categorical(2, vec!["A".to_string(), "B".to_string()]);
    match categorical.node_type {
        NodeType::Internal {
            feature_index,
            threshold: _,
            categories,
        } => {
            assert_eq!(feature_index, 2);
            assert_eq!(categories, Some(vec!["A".to_string(), "B".to_string()]));
        }
        _ => panic!("Expected an internal categorical node"),
    }
}

#[test]
fn test_decision_tree_fit_predict() {
    // Create a simple training dataset
    // Features: two columns, first for "height", second for "weight"
    let x_train = Array2::from(vec![
        [170.0, 65.0], // tall and light
        [180.0, 85.0], // tall and heavy
        [155.0, 55.0], // short and light
        [160.0, 80.0], // short and heavy
    ]);

    // Labels: 0 means "unhealthy", 1 means "healthy"
    let y_train = Array1::from(vec![1.0, 0.0, 1.0, 0.0]);

    // Create test dataset
    let x_test = Array2::from(vec![
        [175.0, 60.0], // should predict healthy (1)
        [165.0, 90.0], // should predict unhealthy (0)
    ]);

    // Create a decision tree classifier (using CART algorithm)
    let mut tree = DecisionTree::new(Algorithm::CART, true, None);

    // Use fit_predict method to train and predict
    let predictions = tree.fit_predict(&x_train, &y_train, &x_test).unwrap();

    // Verify prediction results
    assert_eq!(predictions.len(), 2);
    assert_eq!(predictions[0], 1.0); // First test sample should be predicted as healthy (1)
    assert_eq!(predictions[1], 0.0); // Second test sample should be predicted as unhealthy (0)

    // Test other properties of the model
    assert_eq!(tree.get_n_features(), 2);
    assert!(tree.get_is_classifier());
    assert_eq!(*tree.get_algorithm(), Algorithm::CART);
    assert!(tree.get_root().is_ok()); // Tree should be built
}

#[test]
fn test_decision_tree_with_custom_params() {
    // Create custom parameters
    let params = DecisionTreeParams {
        max_depth: Some(2),
        min_samples_split: 2,
        min_samples_leaf: 1,
        min_impurity_decrease: 0.0,
        random_state: Some(42),
    };

    // Create simple dataset
    let x_train = Array2::from(vec![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
    ]);
    let y_train = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    let x_test = Array2::from(vec![[2.5, 3.5], [4.5, 5.5]]);

    // Create decision tree with custom parameters
    let mut tree = DecisionTree::new(Algorithm::ID3, true, Some(params));

    // Train and predict
    let predictions = tree.fit_predict(&x_train, &y_train, &x_test).unwrap();

    // Verify prediction results length
    assert_eq!(predictions.len(), 2);

    // Verify parameters were correctly applied
    assert_eq!(tree.get_params().max_depth, Some(2));
    assert_eq!(tree.get_params().random_state, Some(42));
}
