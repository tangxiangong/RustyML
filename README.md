# RustyML
A comprehensive machine learning and deep learning library written in pure Rust.
一个用纯Rust编写的全面机器学习和深度学习库。
## Overview | 概述
Rust AI aims to be a feature-rich machine learning and deep learning framework that leverages Rust's performance, memory safety, and concurrency features. While currently in early development stages with foundational components, the project's long-term vision is to provide a complete ecosystem for machine learning, deep learning, and transformer-based models.
Rust AI 旨在成为一个功能丰富的机器学习和深度学习框架，充分利用Rust的性能、内存安全性和并发特性。虽然目前处于早期开发阶段，只实现了基础组件，但项目的长期愿景是提供一个完整的机器学习、深度学习和基于transformer架构的模型生态系统。
## Current Features | 当前功能
- **Mathematical Utilities | 数学工具**: Core mathematical functions for statistical operations and model evaluation:
    - Sum of square total (SST) for measuring data variability | 总平方和(SST)，用于衡量数据变异性
    - Sum of squared errors (SSE) for evaluating prediction errors | 误差平方和(SSE)，用于评估预测误差
    - R-squared (R²) score for assessing model fit quality | R平方(R²)分数，用于评估模型拟合质量
    - Sigmoid function for logistic regression and neural networks | Sigmoid函数，用于逻辑回归和神经网络
    - Logistic loss (log loss) for binary classification models | 逻辑损失函数(对数损失)，用于二元分类模型
    - Accuracy score for classification model evaluation | 准确率分数，用于分类模型评估

- **Machine Learning Models | 机器学习模型**: Initial implementation of basic machine learning algorithms:
    - Linear Regression (initial implementation) | 线性回归(初步实现)
    - Logistic Regression (initial implementation) | 逻辑回归(初步实现)

## Vision | 愿景
While the library is in its early stages, Rust AI aims to evolve into a comprehensive crate that includes:
虽然该库处于早期阶段，但Rust AI旨在发展成为一个包含以下内容的综合性crate：
- **Classical Machine Learning Algorithms | 经典机器学习算法**:
    - Linear and Logistic Regression | 线性和逻辑回归
    - Support Vector Machines | 支持向量机
    - Decision Trees and Random Forests | 决策树和随机森林
    - Clustering algorithms (K-means, DBSCAN) | 聚类算法(K均值，DBSCAN)
    - Dimensionality reduction techniques (PCA, t-SNE) | 降维技术(PCA, t-SNE)

- **Deep Learning | 深度学习**:
    - Neural network building blocks | 神经网络构建模块
    - Convolutional neural networks | 卷积神经网络
    - Recurrent neural networks | 循环神经网络
    - Optimization algorithms | 优化算法

- **Transformer Architecture | Transformer架构**:
    - Self-attention mechanisms | 自注意力机制
    - Multi-head attention | 多头注意力
    - Encoder-decoder architectures | 编码器-解码器架构
    - Pre-training and fine-tuning capabilities | 预训练和微调能力

- **Utilities | 实用工具**:
    - Data preprocessing | 数据预处理
    - Cross-validation | 交叉验证
    - Performance metrics | 性能指标
    - Visualization helpers | 可视化辅助工具

## Dependencies | 依赖
- [ndarray](https://crates.io/crates/ndarray) (0.16.1): N-dimensional array library for Rust | Rust的N维数组库

## Getting Started | 开始使用(Not available right now)
Add the library to your `Cargo.toml`:
将库添加到您的`Cargo.toml`文件中：
``` toml
[dependencies]
rustyml = "0.1.0"
```
Example usage | 使用示例:
``` rust
use rustyml::math::{sum_of_squared_errors, r2_score};

// Example data | 示例数据
let predicted = vec![2.1, 3.8, 5.2, 7.1];
let actual = vec![2.0, 4.0, 5.0, 7.0];

// Calculate error metrics | 计算误差指标
let sse = sum_of_squared_errors(&predicted, &actual);
let r2 = r2_score(&predicted, &actual);

println!("Sum of Squared Errors: {}", sse);
println!("R² Score: {}", r2);
```
## Project Status | 项目状态
This project is in the **early development stage**. Currently, only a small subset of the planned functionality has been implemented. The API is unstable and subject to significant changes as the project evolves.
该项目处于**早期开发阶段**。目前，仅实现了计划功能的一小部分。API不稳定，随着项目的发展可能会有重大变化。
## Contribution | 贡献
Contributions are welcome! If you're interested in helping build a robust machine learning ecosystem in Rust, please feel free to:
欢迎贡献！如果您有兴趣帮助构建Rust中的强大机器学习生态系统，请随时：
1. Submit issues for bugs or feature requests | 提交bug或功能请求
2. Create pull requests for improvements | 创建改进的拉取请求
3. Provide feedback on the API design | 提供API设计的反馈意见
4. Help with documentation and examples | 帮助完善文档和示例

## License | 许可证
MIT - License
## Authors | 作者
- SomeB1oody (stanyin64@gmail.com)

_RustyML - Bringing the power and safety of Rust to machine learning and AI._
_RustyML - 将Rust的强大性能和安全特性引入机器学习和人工智能领域。_
