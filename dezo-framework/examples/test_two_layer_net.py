"""
测试两层神经网络
演示如何使用TwoLayerNet模型
"""
import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import Variable, TwoLayerNet, mean_squared_error


def test_two_layer_net():
    """测试两层神经网络的基本功能"""
    print("=== 两层神经网络测试 ===\n")
    
    # 创建测试数据
    batch_size, input_dim = 5, 10
    hidden_size, output_dim = 20, 3
    
    x = Variable(np.random.randn(batch_size, input_dim), name='input')
    target = Variable(np.random.randn(batch_size, output_dim), name='target')
    
    # 创建模型
    model = TwoLayerNet(hidden_size, output_dim)
    
    print(f"输入维度: {input_dim}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"输出维度: {output_dim}")
    print(f"批次大小: {batch_size}")
    print()
    
    # 前向传播
    y = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出值范围: [{y.data.min():.3f}, {y.data.max():.3f}]")
    print()
    
    # 计算损失
    loss = mean_squared_error(y, target)
    print(f"初始损失: {loss.data:.6f}")
    
    # 反向传播
    model.cleargrads()
    loss.backward()
    
    # 检查梯度
    param_count = 0
    grad_count = 0
    for param in model.params():
        param_count += 1
        if param.grad is not None:
            grad_count += 1
    
    print(f"参数数量: {param_count}")
    print(f"有梯度的参数: {grad_count}")
    
    # 简单训练几步
    print("\n=== 简单训练 ===")
    lr = 0.01
    
    for i in range(5):
        y = model(x)
        loss = mean_squared_error(y, target)
        
        model.cleargrads()
        loss.backward()
        
        # 更新参数
        for param in model.params():
            if param.grad is not None:
                param.data -= lr * param.grad.data
        
        print(f"迭代 {i+1}: 损失 = {loss.data:.6f}")
    
    print("\n训练完成！")


def test_model_plot():
    """测试模型可视化功能"""
    print("\n=== 模型可视化测试 ===")
    
    try:
        # 创建简单模型
        model = TwoLayerNet(5, 2)
        x = Variable(np.random.randn(1, 3), name='x')
        
        # 生成计算图
        model.plot(x, to_file="../images/two_layer_net_demo.png")
        print("计算图已保存到 images/two_layer_net_demo.png")
        
    except Exception as e:
        print(f"可视化功能需要graphviz: {e}")


if __name__ == '__main__':
    test_two_layer_net()
    test_model_plot() 