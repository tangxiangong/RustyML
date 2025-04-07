pub mod optimizer;
pub mod loss_function;
pub mod layer;
pub mod sequential;

pub use optimizer::*;
pub use loss_function::*;
pub use layer::*;
pub use sequential::*;

use ndarray::ArrayD;

pub type Tensor = ArrayD<f32>;

// 定义层（Layer）的接口
pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    // 返回该层的类型名称，例如 "Dense"
    fn layer_type(&self) -> &str {
        "Unknown"
    }
    // 返回该层前向传播输出的形状描述（这里只给出一个简单字符串）
    fn output_shape(&self) -> String {
        "Unknown".to_string()
    }
    // 返回该层参数数量
    fn param_count(&self) -> usize {
        0
    }
    // 根据给定学习率更新该层的参数（如果有）
    fn update_parameters(&mut self, _lr: f32) {
        // 默认什么都不做
    }
}

pub trait LossFunction {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32;
    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor;
}

// 定义优化器（Optimizer）的接口
pub trait Optimizer {
    fn update(&mut self, layer: &mut dyn Layer);
}