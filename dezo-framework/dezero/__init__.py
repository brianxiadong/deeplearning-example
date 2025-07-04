# 核心组件
from .core import Variable, Function, Config, using_config, no_grad, as_variable, as_array

# 数学函数
from .functions import (
    square, exp, sin, cos, tanh, sigmoid_simple, my_sin,
    add, mul, sub, div, pow, neg, rsub, rdiv,
    numerical_diff
)

# 张量操作
from .tensor_ops import (
    reshape, transpose, sum, sum_to, broadcast_to, matmul
)

# 神经网络层
from .layers import Parameter, Layer, Linear, linear_simple

# 模型和网络
from .models import Model, MLP
from .nets import TwoLayerNet

# 损失函数
from .losses import mean_squared_error, mean_absolute_error

# 优化器
from . import optimizers

# 工具函数
from .utils import plot_dot_graph, get_dot_graph

from .datasets import *


def setup_variable():
    """设置Variable类的运算符重载"""
    Variable.__pow__ = pow
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__rsub__ = rsub
    Variable.__sub__ = sub
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul


# 自动设置运算符重载
setup_variable()