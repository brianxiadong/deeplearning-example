"""
TwoLayerNet 快速训练演示
展示主要功能，训练轮数较少以便快速测试
"""
import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import Variable, TwoLayerNet, mean_squared_error, sigmoid_simple


def quick_regression_demo():
    """快速回归演示"""
    print("🔥 快速回归演示：拟合sin函数")
    print("=" * 50)
    
    # 生成数据
    np.random.seed(42)
    x = np.random.uniform(-np.pi, np.pi, (200, 1))
    y = np.sin(x) + 0.1 * np.random.randn(200, 1)
    
    # 划分训练集和测试集
    train_x, test_x = Variable(x[:160]), Variable(x[160:])
    train_y, test_y = Variable(y[:160]), Variable(y[160:])
    
    print(f"📊 数据: 训练集 {train_x.shape}, 测试集 {test_x.shape}")
    
    # 创建模型
    model = TwoLayerNet(hidden_size=20, out_size=1)
    
    # 训练参数
    learning_rate = 0.02
    epochs = 300
    
    print(f"🧠 模型: 隐藏层=20, 学习率={learning_rate}, 训练轮数={epochs}")
    print()
    
    # 训练循环
    print("🚀 开始训练...")
    print("轮数     训练损失     测试损失")
    print("-" * 35)
    
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
        
        # 打印进度
        if epoch % 50 == 0 or epoch == epochs - 1:
            test_pred = model(test_x)
            test_loss = mean_squared_error(test_y, test_pred)
            print(f"{epoch:4d}     {loss.data:.6f}     {test_loss.data:.6f}")
    
    # 最终评估
    final_pred = model(test_x)
    final_loss = mean_squared_error(test_y, final_pred)
    
    print(f"\n✅ 训练完成! 最终测试损失: {final_loss.data:.6f}")
    
    # 计算一些简单的评估指标
    mse = final_loss.data
    rmse = np.sqrt(mse)
    print(f"📈 RMSE: {rmse:.6f}")
    
    return model


def quick_classification_demo():
    """快速分类演示"""
    print("\n" + "🎯 快速分类演示：二元分类")
    print("=" * 50)
    
    # 生成简单的分类数据
    np.random.seed(42)
    
    # 类别0: 中心在(-1, -1)
    class_0 = np.random.multivariate_normal([-1, -1], [[0.5, 0], [0, 0.5]], 100)
    # 类别1: 中心在(1, 1)  
    class_1 = np.random.multivariate_normal([1, 1], [[0.5, 0], [0, 0.5]], 100)
    
    x = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(100), np.ones(100)]).reshape(-1, 1)
    
    # 打乱数据
    indices = np.random.permutation(200)
    x, y = x[indices], y[indices]
    
    # 划分数据集
    train_x, test_x = Variable(x[:160]), Variable(x[160:])
    train_y, test_y = Variable(y[:160]), Variable(y[160:])
    
    print(f"📊 数据: 训练集 {train_x.shape}, 测试集 {test_x.shape}")
    print(f"📊 类别分布: {np.bincount(train_y.data.flatten().astype(int))}")
    
    # 创建模型
    model = TwoLayerNet(hidden_size=10, out_size=1)
    
    # 训练参数
    learning_rate = 0.1
    epochs = 200
    
    print(f"🧠 模型: 隐藏层=10, 学习率={learning_rate}, 训练轮数={epochs}")
    print()
    
    # 训练循环
    print("🚀 开始训练...")
    print("轮数     训练损失     训练准确率   测试准确率")
    print("-" * 45)
    
    for epoch in range(epochs):
        # 前向传播
        logits = model(train_x)
        pred = sigmoid_simple(logits)
        loss = mean_squared_error(pred, train_y)  # 使用MSE作为损失
        
        # 反向传播
        model.cleargrads()
        loss.backward()
        
        # 参数更新
        for param in model.params():
            if param.grad is not None:
                param.data -= learning_rate * param.grad.data
        
        # 打印进度
        if epoch % 40 == 0 or epoch == epochs - 1:
            # 训练准确率
            train_pred_binary = (pred.data > 0.5).astype(float)
            train_acc = np.mean(train_pred_binary == train_y.data)
            
            # 测试准确率
            test_logits = model(test_x)
            test_pred = sigmoid_simple(test_logits)
            test_pred_binary = (test_pred.data > 0.5).astype(float)
            test_acc = np.mean(test_pred_binary == test_y.data)
            
            print(f"{epoch:4d}     {loss.data:.6f}     {train_acc:.4f}       {test_acc:.4f}")
    
    print(f"\n✅ 分类训练完成! 最终测试准确率: {test_acc:.4f}")
    
    return model


def parameter_comparison():
    """参数数量对比"""
    print("\n" + "⚙️  参数数量对比")
    print("=" * 50)
    
    hidden_sizes = [5, 10, 20, 50]
    input_size = 3
    output_size = 1
    
    print("隐藏层大小   参数数量   参数分布")
    print("-" * 35)
    
    for hidden_size in hidden_sizes:
        model = TwoLayerNet(hidden_size, output_size)
        
        # 初始化权重以计算参数数量
        dummy_input = Variable(np.random.randn(1, input_size))
        _ = model(dummy_input)
        
        # 统计参数
        total_params = 0
        param_info = []
        
        for i, param in enumerate(model.params()):
            param_count = param.data.size
            total_params += param_count
            param_info.append(f"{param.name}:{param_count}")
        
        param_str = " + ".join(param_info)
        print(f"{hidden_size:8d}     {total_params:6d}     {param_str}")


def model_architecture_demo():
    """模型结构演示"""
    print("\n" + "🏗️  模型结构演示")
    print("=" * 50)
    
    # 创建模型
    model = TwoLayerNet(hidden_size=15, out_size=3)
    
    # 创建输入
    x = Variable(np.random.randn(2, 5), name='input')
    
    print(f"输入: {x.shape}")
    print("模型结构:")
    print("  输入层  -> 线性层1 -> Sigmoid -> 线性层2 -> 输出")
    print(f"    {x.shape[1]}     ->    {15}    ->   激活   ->    {3}    -> {model(x).shape}")
    
    # 显示参数信息
    print(f"\n参数详情:")
    for i, param in enumerate(model.params()):
        print(f"  {param.name}: {param.shape} (元素数: {param.size})")
    
    # 测试前向传播
    y = model(x)
    print(f"\n前向传播结果: {y.shape}")
    print(f"输出值范围: [{y.data.min():.3f}, {y.data.max():.3f}]")


def main():
    """主演示函数"""
    print("🎉 TwoLayerNet 快速训练演示")
    print("包含回归、分类、参数对比和结构演示")
    print("=" * 60)
    
    # 回归演示
    regression_model = quick_regression_demo()
    
    # 分类演示
    classification_model = quick_classification_demo()
    
    # 参数对比
    parameter_comparison()
    
    # 模型结构演示
    model_architecture_demo()
    
    print("\n" + "🎊 所有演示完成!")
    print("TwoLayerNet 可以成功处理回归和分类任务!")


if __name__ == '__main__':
    main() 