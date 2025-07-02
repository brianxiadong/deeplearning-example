# DeZero 深度学习框架

DeZero 是一个用于学习的轻量级深度学习框架，基于自动微分实现。本项目经过重构，具有清晰的模块化结构。

## 项目结构

```
dezo-framework/
├── dezero/                    # 核心框架代码
│   ├── __init__.py           # 主要接口导入
│   ├── core.py              # 核心类（Variable, Function, Config）
│   ├── functions.py         # 数学函数（sin, cos, exp等）
│   ├── tensor_ops.py        # 张量操作（reshape, transpose等）
│   ├── layers.py            # 神经网络层（Layer, Linear）
│   ├── losses.py            # 损失函数
│   └── utils.py             # 工具函数和可视化
├── examples/                 # 示例代码
│   ├── simple_test.py       # 基础功能测试
│   └── neural_network_regression.py  # 神经网络回归示例
├── images/                   # 生成的图片文件
├── tests/                    # 测试代码（待添加）
└── README.md                # 项目说明
```

## 核心功能

### 1. 自动微分
```python
from dezero import Variable

x = Variable(2.0)
y = x ** 3 + 2 * x ** 2 + x + 1
y.backward()
print(x.grad)  # 自动计算梯度
```

### 2. 数学函数
```python
from dezero import sin, cos, exp, square

x = Variable(0.5)
y = sin(x) + cos(x) + exp(x)
```

### 3. 神经网络层
```python
from dezero import Linear, Variable

# 创建线性层
layer = Linear(10)  # 输出维度为10
x = Variable([[1, 2, 3]])
y = layer(x)
```

### 4. 损失函数
```python
from dezero import mean_squared_error

pred = Variable([[1, 2, 3]])
target = Variable([[1.1, 1.9, 3.1]])
loss = mean_squared_error(pred, target)
```

## 使用示例

### 简单测试
```bash
cd examples
python simple_test.py
```

### 神经网络回归
```bash
cd examples  
python neural_network_regression.py
```

## 主要改进

1. **模块化设计**: 将原来的单个大文件拆分为功能明确的模块
2. **清晰的API**: 统一的导入接口，便于使用
3. **示例分离**: 将测试和示例代码从核心代码中分离
4. **文档完善**: 添加详细的注释和说明
5. **目录整理**: 图片文件统一存放，结构更清晰

## 核心模块说明

- **core.py**: Variable 和 Function 基类，自动微分的核心实现
- **functions.py**: 各种数学函数的实现（加减乘除、三角函数等）
- **tensor_ops.py**: 张量操作（reshape、转置、求和等）
- **layers.py**: 神经网络层的实现
- **losses.py**: 常用损失函数
- **utils.py**: 可视化工具和辅助函数

## 扩展性

框架设计具有良好的扩展性：
- 可以轻松添加新的数学函数
- 可以扩展新的神经网络层
- 支持自定义损失函数
- 便于添加优化器等组件

## 依赖项

- NumPy: 核心数值计算
- matplotlib (可选): 用于绘图和可视化
- graphviz (可选): 用于计算图可视化 