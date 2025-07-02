"""
TwoLayerNet 完整训练示例
包含回归任务和分类任务的训练测试用例
"""
import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import Variable, TwoLayerNet, mean_squared_error, sigmoid_simple


def generate_regression_data(n_samples=1000, noise=0.1):
    """生成回归任务的数据 - 拟合sin函数"""
    np.random.seed(42)
    x = np.random.uniform(-2*np.pi, 2*np.pi, (n_samples, 1))
    y = np.sin(x) + noise * np.random.randn(n_samples, 1)
    return x, y


def generate_classification_data(n_samples=1000, n_features=2):
    """生成分类任务的数据 - 二元分类"""
    np.random.seed(42)
    
    # 生成两个类别的数据
    class_0 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], n_samples//2)
    class_1 = np.random.multivariate_normal([-2, -2], [[1, 0], [0, 1]], n_samples//2)
    
    x = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    y = y.reshape(-1, 1)
    
    return x, y


def train_regression():
    """回归任务训练示例"""
    print("=" * 60)
    print("回归任务：使用TwoLayerNet拟合sin函数")
    print("=" * 60)
    
    # 生成数据
    train_x, train_y = generate_regression_data(800, noise=0.2)
    test_x, test_y = generate_regression_data(200, noise=0.2)
    
    print(f"训练数据: {train_x.shape}, 测试数据: {test_x.shape}")
    
    # 转换为Variable
    train_x = Variable(train_x)
    train_y = Variable(train_y)
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    
    # 创建模型
    model = TwoLayerNet(hidden_size=50, out_size=1)
    
    # 训练参数
    learning_rate = 0.01
    epochs = 2000
    print_interval = 200
    
    print(f"模型参数: 隐藏层={50}, 输出层={1}")
    print(f"训练参数: 学习率={learning_rate}, 轮数={epochs}")
    print()
    
    # 训练循环
    train_losses = []
    test_losses = []
    
    print("开始训练...")
    print("轮数      训练损失      测试损失")
    print("-" * 40)
    
    for epoch in range(epochs):
        # 训练
        train_pred = model(train_x)
        train_loss = mean_squared_error(train_y, train_pred)
        
        # 清除梯度并反向传播
        model.cleargrads()
        train_loss.backward()
        
        # 更新参数
        for param in model.params():
            if param.grad is not None:
                param.data -= learning_rate * param.grad.data
        
        # 记录损失
        train_losses.append(train_loss.data)
        
        # 测试
        if epoch % print_interval == 0 or epoch == epochs - 1:
            test_pred = model(test_x)
            test_loss = mean_squared_error(test_y, test_pred)
            test_losses.append(test_loss.data)
            
            print(f"{epoch:4d}      {train_loss.data:.6f}      {test_loss.data:.6f}")
    
    print("\n回归训练完成!")
    
    # 最终评估
    final_train_pred = model(train_x)
    final_test_pred = model(test_x)
    final_train_loss = mean_squared_error(train_y, final_train_pred)
    final_test_loss = mean_squared_error(test_y, final_test_pred)
    
    print(f"最终训练损失: {final_train_loss.data:.6f}")
    print(f"最终测试损失: {final_test_loss.data:.6f}")
    
    # 保存结果图片
    try:
        import matplotlib.pyplot as plt
        
        # 绘制拟合结果
        plt.figure(figsize=(12, 4))
        
        # 子图1：训练结果
        plt.subplot(1, 2, 1)
        x_sorted = np.argsort(test_x.data.flatten())
        plt.scatter(test_x.data[x_sorted], test_y.data[x_sorted], alpha=0.6, s=20, label='真实值')
        plt.plot(test_x.data[x_sorted], final_test_pred.data[x_sorted], 'r-', linewidth=2, label='预测值')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Sin函数拟合结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(0, epochs, print_interval), test_losses[:-1], 'b-', label='测试损失')
        plt.plot(range(epochs), train_losses, 'r-', alpha=0.7, label='训练损失')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../images/regression_training.png', dpi=150, bbox_inches='tight')
        print("拟合结果图片已保存到 images/regression_training.png")
        
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
    
    return model


def binary_cross_entropy_simple(pred, target):
    """简单的二元交叉熵损失"""
    # 为数值稳定性添加小的epsilon
    eps = 1e-15
    pred_clipped = pred  # 注意：这里应该添加clipping，但Variable不支持
    
    # 简化版本：使用均方误差作为替代
    return mean_squared_error(pred, target)


def train_classification():
    """分类任务训练示例"""
    print("\n" + "=" * 60)
    print("分类任务：使用TwoLayerNet进行二元分类")
    print("=" * 60)
    
    # 生成数据
    train_x, train_y = generate_classification_data(800)
    test_x, test_y = generate_classification_data(200)
    
    print(f"训练数据: {train_x.shape}, 测试数据: {test_x.shape}")
    print(f"类别分布 - 训练集: {np.bincount(train_y.flatten().astype(int))}")
    print(f"类别分布 - 测试集: {np.bincount(test_y.flatten().astype(int))}")
    
    # 转换为Variable
    train_x = Variable(train_x)
    train_y = Variable(train_y)
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    
    # 创建模型
    model = TwoLayerNet(hidden_size=20, out_size=1)
    
    # 训练参数
    learning_rate = 0.1
    epochs = 1000
    print_interval = 100
    
    print(f"模型参数: 隐藏层={20}, 输出层={1}")
    print(f"训练参数: 学习率={learning_rate}, 轮数={epochs}")
    print()
    
    # 训练循环
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("开始训练...")
    print("轮数      训练损失      训练准确率    测试准确率")
    print("-" * 50)
    
    for epoch in range(epochs):
        # 训练
        train_logits = model(train_x)
        train_pred = sigmoid_simple(train_logits)
        
        # 计算损失（使用MSE作为替代）
        train_loss = binary_cross_entropy_simple(train_pred, train_y)
        
        # 清除梯度并反向传播
        model.cleargrads()
        train_loss.backward()
        
        # 更新参数
        for param in model.params():
            if param.grad is not None:
                param.data -= learning_rate * param.grad.data
        
        # 记录损失
        train_losses.append(train_loss.data)
        
        # 计算准确率
        if epoch % print_interval == 0 or epoch == epochs - 1:
            # 训练准确率
            train_pred_binary = (train_pred.data > 0.5).astype(float)
            train_acc = np.mean(train_pred_binary == train_y.data)
            train_accuracies.append(train_acc)
            
            # 测试准确率
            test_logits = model(test_x)
            test_pred = sigmoid_simple(test_logits)
            test_pred_binary = (test_pred.data > 0.5).astype(float)
            test_acc = np.mean(test_pred_binary == test_y.data)
            test_accuracies.append(test_acc)
            
            print(f"{epoch:4d}      {train_loss.data:.6f}      {train_acc:.4f}        {test_acc:.4f}")
    
    print("\n分类训练完成!")
    
    # 最终评估
    final_test_logits = model(test_x)
    final_test_pred = sigmoid_simple(final_test_logits)
    final_test_pred_binary = (final_test_pred.data > 0.5).astype(float)
    final_test_acc = np.mean(final_test_pred_binary == test_y.data)
    
    print(f"最终测试准确率: {final_test_acc:.4f}")
    
    # 保存结果图片
    try:
        import matplotlib.pyplot as plt
        
        # 绘制分类结果
        plt.figure(figsize=(12, 4))
        
        # 子图1：分类结果
        plt.subplot(1, 2, 1)
        
        # 绘制数据点
        class_0_idx = test_y.data.flatten() == 0
        class_1_idx = test_y.data.flatten() == 1
        
        plt.scatter(test_x.data[class_0_idx, 0], test_x.data[class_0_idx, 1], 
                   c='blue', alpha=0.6, label='类别 0', s=30)
        plt.scatter(test_x.data[class_1_idx, 0], test_x.data[class_1_idx, 1], 
                   c='red', alpha=0.6, label='类别 1', s=30)
        
        # 绘制错误分类的点
        wrong_pred = final_test_pred_binary.flatten() != test_y.data.flatten()
        if np.any(wrong_pred):
            plt.scatter(test_x.data[wrong_pred, 0], test_x.data[wrong_pred, 1], 
                       c='black', marker='x', s=50, label='错误分类')
        
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title(f'二元分类结果 (准确率: {final_test_acc:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：准确率曲线
        plt.subplot(1, 2, 2)
        epochs_plot = list(range(0, epochs, print_interval)) + [epochs-1]
        epochs_plot = epochs_plot[:len(train_accuracies)]
        
        plt.plot(epochs_plot, train_accuracies, 'b-', label='训练准确率')
        plt.plot(epochs_plot, test_accuracies, 'r-', label='测试准确率')
        plt.xlabel('轮数')
        plt.ylabel('准确率')
        plt.title('训练准确率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../images/classification_training.png', dpi=150, bbox_inches='tight')
        print("分类结果图片已保存到 images/classification_training.png")
        
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
    
    return model


def compare_models():
    """比较不同隐藏层大小的模型性能"""
    print("\n" + "=" * 60)
    print("模型比较：不同隐藏层大小对性能的影响")
    print("=" * 60)
    
    # 生成测试数据
    train_x, train_y = generate_regression_data(500, noise=0.1)
    test_x, test_y = generate_regression_data(100, noise=0.1)
    
    train_x = Variable(train_x)
    train_y = Variable(train_y)
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    
    hidden_sizes = [5, 10, 20, 50, 100]
    results = []
    
    print("隐藏层大小    最终训练损失    最终测试损失")
    print("-" * 45)
    
    for hidden_size in hidden_sizes:
        # 创建并训练模型
        model = TwoLayerNet(hidden_size=hidden_size, out_size=1)
        
        # 简化训练
        learning_rate = 0.01
        epochs = 500
        
        for epoch in range(epochs):
            pred = model(train_x)
            loss = mean_squared_error(train_y, pred)
            
            model.cleargrads()
            loss.backward()
            
            for param in model.params():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad.data
        
        # 最终评估
        final_train_pred = model(train_x)
        final_test_pred = model(test_x)
        final_train_loss = mean_squared_error(train_y, final_train_pred)
        final_test_loss = mean_squared_error(test_y, final_test_pred)
        
        results.append((hidden_size, final_train_loss.data, final_test_loss.data))
        
        print(f"{hidden_size:8d}        {final_train_loss.data:.6f}        {final_test_loss.data:.6f}")
    
    # 找出最佳模型
    best_result = min(results, key=lambda x: x[2])  # 按测试损失排序
    print(f"\n最佳隐藏层大小: {best_result[0]} (测试损失: {best_result[2]:.6f})")


def main():
    """主函数：运行所有训练示例"""
    print("TwoLayerNet 训练测试用例")
    print("包含回归、分类和模型比较")
    
    # 回归任务
    regression_model = train_regression()
    
    # 分类任务
    classification_model = train_classification()
    
    # 模型比较
    compare_models()
    
    print("\n" + "=" * 60)
    print("所有训练测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    main() 