use super::*;

pub struct MeanSquaredError;

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for MeanSquaredError {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // 计算预测值和真实值之间的差异
        let diff = y_pred - y_true;

        // 计算差异的平方
        let squared_diff = &diff.mapv(|x| x * x);

        // 计算平均值（总和除以元素数量）
        let n = squared_diff.len() as f32;
        squared_diff.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // 计算预测值和真实值之间的差异
        let diff = y_pred - y_true;

        // 梯度是差异的2倍除以样本数量
        let n = diff.len() as f32;
        diff.mapv(|x| 2.0 * x / n)
    }
}

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunction for BinaryCrossEntropy {
    fn compute_loss(&self, y_true: &Tensor, y_pred: &Tensor) -> f32 {
        // 确保预测值在 (0,1) 范围内，避免数值问题
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 二元交叉熵公式: -1/n * Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        let losses = y_true.mapv(|y_t| y_t).to_owned() * &y_pred_clipped.mapv(|y_p| y_p.ln())
            + (1.0 - y_true).mapv(|y_t| y_t) * &(1.0 - &y_pred_clipped).mapv(|y_p| y_p.ln());

        // 计算平均损失（负号在这里加上）
        let n = losses.len() as f32;
        -losses.sum() / n
    }

    fn compute_grad(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
        // 确保预测值在 (0,1) 范围内，避免数值问题
        let y_pred_clipped = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // 二元交叉熵的梯度: -y_true/y_pred + (1-y_true)/(1-y_pred)
        let grad = -y_true / &y_pred_clipped + (1.0 - y_true) / (1.0 - &y_pred_clipped);

        // 除以样本数量以获得平均梯度
        let n = grad.len() as f32;
        grad / n
    }
}