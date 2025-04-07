use super::*;

pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layer: &mut dyn Layer) {
        // 直接调用层的参数更新方法即可
        layer.update_parameters(self.learning_rate);
    }
}