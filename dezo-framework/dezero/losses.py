from .tensor_ops import sum


def mean_squared_error(x0, x1):
    """均方误差损失函数"""
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)


def mean_absolute_error(x0, x1):
    """平均绝对误差损失函数"""
    diff = x0 - x1
    return sum(abs(diff)) / len(diff)


def binary_cross_entropy(x, target):
    """二元交叉熵损失函数"""
    # 为了数值稳定性，对x进行裁剪
    eps = 1e-15
    x = x.clip(eps, 1 - eps)
    return -sum(target * log(x) + (1 - target) * log(1 - x)) / len(x)


def categorical_cross_entropy(x, target):
    """分类交叉熵损失函数"""
    eps = 1e-15
    x = x.clip(eps, 1.0)
    return -sum(target * log(x)) / len(x) 