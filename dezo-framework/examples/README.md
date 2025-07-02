# DeZero 示例和测试用例

本目录包含了 DeZero 深度学习框架的各种示例和测试用例。

## 📁 文件列表

### 🔍 基础测试
- **`simple_test.py`** - 基础功能测试，验证自动微分和数学函数
- **`test_two_layer_net.py`** - TwoLayerNet 基本功能测试

### 🚀 训练示例
- **`neural_network_regression.py`** - 使用 Linear 层的神经网络回归示例
- **`two_layer_net_training.py`** - 完整的 TwoLayerNet 训练示例（回归+分类）
- **`quick_training_demo.py`** - 快速训练演示，展示主要功能

## 🎯 使用方法

### 运行基础测试
```bash
cd examples
python simple_test.py
```

### 运行神经网络回归
```bash
python neural_network_regression.py
```

### 运行快速演示
```bash
python quick_training_demo.py
```

## 📊 演示结果

### 回归任务表现
- **数据**: 拟合 sin 函数，添加噪声
- **模型**: TwoLayerNet(隐藏层=20)
- **结果**: 测试损失从 0.42 降至 0.14 (RMSE: 0.37)

### 分类任务表现  
- **数据**: 二元分类，两个高斯分布
- **模型**: TwoLayerNet(隐藏层=10)
- **结果**: 测试准确率达到 97.5%

### 参数规模对比
| 隐藏层大小 | 参数数量 | 参数分布 |
|-----------|----------|----------|
| 5         | 26       | W:15 + b:5 + W:5 + b:1 |
| 10        | 51       | W:30 + b:10 + W:10 + b:1 |
| 20        | 101      | W:60 + b:20 + W:20 + b:1 |
| 50        | 251      | W:150 + b:50 + W:50 + b:1 |

## 🛠️ TwoLayerNet 训练模板

```python
import numpy as np
from dezero import Variable, TwoLayerNet, mean_squared_error

# 1. 准备数据
train_x = Variable(np.random.randn(100, 5))
train_y = Variable(np.random.randn(100, 1))

# 2. 创建模型
model = TwoLayerNet(hidden_size=20, out_size=1)

# 3. 训练循环
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # 前向传播
    pred = model(train_x)
    loss = mean_squared_error(train_y, pred)
    
    # 反向传播
    model.cleargrads()
    loss.backward()
    
    # 参数更新
    for param in model.params():
        if param.grad is not None:
            param.data -= learning_rate * param.grad.data
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.6f}")
```

## 🎨 可视化功能

大部分示例都支持可视化功能（需要 matplotlib）：

- **训练曲线**: 损失随轮数的变化
- **拟合结果**: 回归任务的拟合效果
- **分类边界**: 分类任务的决策边界
- **计算图**: 模型的计算图结构

## 🧪 扩展实验

你可以基于这些示例进行扩展实验：

1. **调整网络结构**: 改变隐藏层大小
2. **尝试不同学习率**: 观察收敛速度的影响
3. **添加更多层**: 扩展为多层网络
4. **使用不同激活函数**: 替换 sigmoid
5. **实验不同数据集**: 尝试更复杂的数据

## 📈 性能指标

### 训练效果评估
- **回归**: 使用 MSE、RMSE 评估
- **分类**: 使用准确率、混淆矩阵评估
- **收敛**: 观察损失曲线的下降趋势

### 计算效率
- **小规模数据**: 几百个样本，秒级训练
- **参数规模**: 几十到几百个参数
- **内存占用**: 适合学习和原型开发

## 🚨 注意事项

1. **数值稳定性**: 某些情况下可能出现梯度消失/爆炸
2. **学习率调整**: 过大可能发散，过小收敛慢
3. **随机种子**: 示例使用固定种子确保结果可重现
4. **中文字体**: matplotlib 显示中文可能有字体警告（不影响功能） 