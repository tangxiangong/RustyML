# Changelog

All notable changes to this project will be documented in this file.
This change log records updates after 2025-3-24.

Please view [SomeB1oody/RustyML](https://github.com/SomeB1oody/RustyML) for more info.

## [v0.2.1] - 2025-4-4
### Added
- Add variance calculation in math module
- Add MSE calculation in metric module
- Print info after training completes for `fit` functions of struct `LDA`, `SVC`, `LinearSVC` and `PCA`
- Integrate Rayon for parallel computation across modules (**7s faster** in `cargo test`!!!)

### Changed
- Refactor functions in math module and metric module to use `ArrayView1` for improved efficiency

### Removed
- Remove MSE calculation(named mean_squared_error, but actually calculate variance) in math module

## [v0.2.1] - 2025-4-3
### Added
- Add t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation

### Changed
- Remove gaussian kernel calculation function, put gaussian kernel calculation directly in function `fit` of struct `MeanShift`

## [v0.2.1] - 2025-4-2
### Added
- Add LinearSVC support
- Add KernelPCA support
- Add `ProcessingError(String)` to `crate::ModelError` 
- Add LDA(Linear Discriminant Analysis) support

### Changed
- Change the location of function `standardize` from `crate::utility::principal_component_analysis` to `crate::utility`

## [v0.2.1] - 2025-4-1
### Added
- Add SVC(Support Vector Classification) support

## [v0.2.0] - 2025-4-1
### Added
- Add `train_test_split` function in utility module to split dataset for training and dataset for test
- Add function `normalized_mutual_info` and `adjusted_mutual_info` to metric module to calculate NMI and AMI info
- Add AUC-ROC value calculation in metric module

## [v0.2.0] - 2025-3-31
### Changed
- Change `principal_component_analysis` module to `utility` module, change `principal_component_analysis_test` module to `utility_test` module
- Keep the algorithm functions in the math module, and move the functions that evaluate the model's performance (such as R-square values) and structures (confusion matrices) to the metric module. Some of them are used in both ways, then keep them in both modules.
- Change the output of some of the functions in math module and metric module from `T` to `Result<T, crate::ModelError>`

## [v0.1.1] - 2025-3-31
### Added
- Add function `preliminary_check` in machine_learning module to performs validation checks on the input data matrices
- Add confusion matrix in math module

### Changed
- Change type of field `coefficients` of struct `LinearRegression` from `Option<Vec<f64>>` to `Option<Array1<f64>>`
- Change the output of some methods of struct `LinearRegression` from `Vec<f64>` to `Array1<f64>`
- Change variant `InputValidationError` of enum type `ModelError` from `InputValidationError(&str)` to `InputValidationError(String)`

## [v0.1.0] - 2025-3-30
### Added
- Add function `fit_predict` for some models
- Add examples for functions in math.rs
- Add input validation
- Add doc comments for machine learning modules
- Add prelude module(all re-exports are there)

### Changed
- Change input types of function `fit`, `predict` and `fit_predict` to `Array1` and `Array2`
- Rename the crate from `rust_ai` to `rustyml`
- Change the output of function `fit` from `&mut Self` to `Result<&mut Self, ModelError>` or `Result<&mut Self, Box<dyn std::error::Error>>`

## [v0.1.0] - 2025-3-29
### Added
- Add function `generate_tree_structure` for `DecisionTree` to generate tree structure as string
- Add isolation forest implementation
- Add PCA(Principal Component Analysis) implementation
- Add function `standard_deviation` in math module to calculates the standard deviation of a set of values

## [v0.1.0] - 2025-3-28
### Added
- Add Decision Tree model
- Add following functions to math.rs:
    - `entropy`: Calculates the entropy of a label set
    - `gini`: Calculates the Gini impurity of a label set
    - `information_gain`: Calculates the information gain when splitting a dataset
    - `gain_ratio`: Calculates the gain ratio for a dataset split
    - `mean_squared_error`: Calculates the Mean Squared Error (MSE) of a set of values

### Changed
- Replaced string-based distance calculation method options with an enum `crate::machine_learning::DistanceCalculation`
- For KNN model: replaced string-based weight function options with an enum `crate::machine_learning::knn::WeightingStrategy`
- For decision tree: replaced string-based algorithm options with an enum `crate::machine_learning::decision_tree::Algorithm`

## [v0.1.0] - 2025-3-27
### Added
- Add changelog.md to record updates
- Add DBSCAN model
- Add function `fit_predict` to fit and predict in one step
- Add doc comments to tell user `p` value of function `minkowski_distance` in model is always 3

## [v0.1.0] - 2025-3-26
### Added
- Add "xx model converged at iteration x, cost: x" when finishing `fit`
- Add description for `n_iter` field
- Add getter functions for `KMeans`
- implement `Default` trait for `KMeans`

### Changed
- Rename `max_iteration` and `tolerance` to `max_iter` and `tol`
- Change doc comments to enhanced consistency

### Removed
- Remove examples in math.rs(add them back later)

## [v0.1.0] - 2025-3-25
### Added
- Add MeanShift model
- Add `InputValidationError` in `ModelError`, indicating the input data provided  does not meet the expected format, type, or validation rules
- Add `gaussian_kernel` in math module, calculate the Gaussian kernel (RBF kernel)

### Changed
- Change the output of all `predict` functions(except KNN) from `T` to `Result<T, crate::ModelError>`
- Correct doc comments