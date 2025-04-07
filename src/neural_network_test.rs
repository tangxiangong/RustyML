use ndarray::Array;
use crate::neural_network::*;

#[test]
fn main_test() {
    // 构造输入和目标张量，假设输入维度为 4，输出维度为 3，batch_size = 2
    let x = Array::ones((2, 4)).into_dyn();
    let y = Array::ones((2, 1)).into_dyn();

    // 构建模型，注意第二个 Dense 层必须用 Dense::new(3, 1)
    let mut model = Sequential::new();
    model.add(Dense::new(4, 3))
        .add(Dense::new(3, 1));
    model.compile(SGD::new(0.01), MeanSquaredError::new());

    // 打印模型结构（summary）
    model.summary();

    // 训练模型
    model.fit(&x, &y, 3);

    // 使用 predict 进行前向传播预测
    let prediction = model.predict(&x);
    println!("预测结果: {:?}", prediction);
}
