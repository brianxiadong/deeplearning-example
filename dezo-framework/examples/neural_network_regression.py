"""
神经网络回归示例
使用两层神经网络拟合 sin 函数
"""
import numpy as np
import sys
import os

# 添加父目录到Python路径，以便导入dezero包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import Variable, Linear, mean_squared_error, sigmoid_simple


def main():
    """主函数：训练神经网络拟合sin函数"""
    
    # 生成训练数据
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    
    # 转换为Variable
    x, y = Variable(x), Variable(y)
    
    # 创建两层神经网络
    l1 = Linear(10)  # 隐藏层：1 -> 10
    l2 = Linear(1)   # 输出层：10 -> 1
    
    def predict(x):
        """前向传播"""
        y = l1(x)
        y = sigmoid_simple(y)
        y = l2(y)
        return y
    
    # 训练参数
    lr = 0.2      # 学习率
    iters = 10000 # 迭代次数
    
    print("开始训练...")
    print("迭代次数  |  损失值")
    print("-" * 25)
    
    # 训练循环
    for i in range(iters):
        # 前向传播
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)
        
        # 清除梯度
        l1.cleargrads()
        l2.cleargrads()
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        for layer in [l1, l2]:
            for param in layer.params():
                param.data -= lr * param.grad.data
        
        # 打印训练进度
        if i % 1000 == 0:
            print(f"{i:8d}  |  {loss.data:.6f}")
    
    print("\n训练完成!")
    print(f"最终损失: {loss.data:.6f}")
    
    # 可选：保存结果图片（需要matplotlib）
    try:
        import matplotlib.pyplot as plt
        
        # 生成测试数据
        t = np.arange(0, 1, 0.01)[:, np.newaxis]
        t_var = Variable(t)
        y_pred_test = predict(t_var)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.scatter(x.data, y.data, s=10, alpha=0.6, label='训练数据')
        plt.plot(t, y_pred_test.data, color='red', linewidth=2, label='神经网络预测')
        plt.plot(t, np.sin(2 * np.pi * t), color='green', linestyle='--', label='真实函数')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('神经网络拟合 sin 函数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.savefig('../images/neural_network_regression.png', dpi=150, bbox_inches='tight')
        print("结果图片已保存到 images/neural_network_regression.png")
        
    except ImportError:
        print("matplotlib 未安装，跳过绘图")


if __name__ == '__main__':
    main() 