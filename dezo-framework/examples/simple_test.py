"""
简单测试示例
验证重构后的DeZero框架基本功能
"""
import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import Variable, sin, cos, exp, square
from dezero.utils import goldstein


def test_basic_operations():
    """测试基本数学运算"""
    print("=== 测试基本数学运算 ===")
    
    # 创建变量
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))
    
    # 基本运算
    z1 = x + y
    z2 = x * y
    z3 = x ** 2
    
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"x + y = {z1.data}")
    print(f"x * y = {z2.data}")
    print(f"x^2 = {z3.data}")


def test_math_functions():
    """测试数学函数"""
    print("\n=== 测试数学函数 ===")
    
    x = Variable(np.array(0.5))
    
    y1 = sin(x)
    y2 = cos(x)
    y3 = exp(x)
    y4 = square(x)
    
    print(f"x = {x.data}")
    print(f"sin(x) = {y1.data:.6f}")
    print(f"cos(x) = {y2.data:.6f}")
    print(f"exp(x) = {y3.data:.6f}")
    print(f"square(x) = {y4.data:.6f}")


def test_backward():
    """测试反向传播"""
    print("\n=== 测试反向传播 ===")
    
    x = Variable(np.array(2.0))
    y = x ** 3 + 2 * x ** 2 + x + 1  # y = x^3 + 2x^2 + x + 1
    
    print(f"x = {x.data}")
    print(f"y = x^3 + 2x^2 + x + 1 = {y.data}")
    
    y.backward()
    
    # 理论梯度: dy/dx = 3x^2 + 4x + 1 = 3*4 + 4*2 + 1 = 21
    print(f"dy/dx = {x.grad.data} (理论值: 21)")


def test_goldstein_function():
    """测试Goldstein-Price函数"""
    print("\n=== 测试Goldstein-Price函数 ===")
    
    x = Variable(np.array(0.0))
    y = Variable(np.array(-1.0))
    
    z = goldstein(x, y)
    z.backward()
    
    print(f"Goldstein({x.data}, {y.data}) = {z.data}")
    print(f"梯度: dx = {x.grad.data:.6f}, dy = {y.grad.data:.6f}")


def main():
    """运行所有测试"""
    print("DeZero 框架重构后测试")
    print("=" * 50)
    
    test_basic_operations()
    test_math_functions()
    test_backward()
    test_goldstein_function()
    
    print("\n所有测试完成!")


if __name__ == '__main__':
    main() 