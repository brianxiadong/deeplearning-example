#!/usr/bin/env python3
"""
测试优化后的 SimpleViT 模型
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import os
import random

# 添加上级目录到路径，以便导入vit_pytorch模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from vit_pytorch import SimpleViT

# CIFAR-10 类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def get_device():
    """智能设备选择"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 使用 CUDA 设备")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🍎 使用 Apple Silicon MPS 加速")
    else:
        device = torch.device('cpu')
        print(f"💻 使用 CPU 设备")
    return device

def create_model():
    """创建与训练时相同的模型架构"""
    model = SimpleViT(
        image_size=128,      # 与训练时相同
        patch_size=16,
        num_classes=10,
        dim=384,             # 与训练时相同
        depth=4,             # 与训练时相同
        heads=6,             # 与训练时相同
        mlp_dim=768,         # 与训练时相同
        channels=3,
        dim_head=64
    )
    return model

def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"📂 加载模型: {model_path}")
    
    # 创建模型
    model = create_model()
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    if 'best_acc' in checkpoint:
        print(f"📊 模型最佳准确率: {checkpoint['best_acc']:.2f}%")
    
    return model

def test_with_cifar10_samples(model, device, num_samples=10):
    """使用CIFAR-10测试样本进行测试"""
    print(f"\n=== 使用CIFAR-10测试样本 ===")
    
    # 加载CIFAR-10测试集
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 与训练时相同
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 测试随机样本
    correct = 0
    total = 0
    
    for i in range(num_samples):
        idx = random.randint(0, len(test_dataset) - 1)
        image, true_label = test_dataset[idx]
        true_class = CIFAR10_CLASSES[true_label]
        
        # 预测
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_batch)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            
            # 获取top-3预测
            top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        predicted_class = CIFAR10_CLASSES[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()
        
        # 检查预测是否正确
        is_correct = predicted_class == true_class
        if is_correct:
            correct += 1
        total += 1
        
        print(f"\n样本 {i+1} (索引: {idx}):")
        print(f"真实标签: {true_class}")
        print(f"预测结果: {predicted_class} (置信度: {confidence:.3f})")
        print(f"状态: {'✅ 正确' if is_correct else '❌ 错误'}")
        
        # 显示top-3预测
        print("Top-3 预测:")
        for j in range(3):
            class_name = CIFAR10_CLASSES[top3_indices[0][j]]
            prob = top3_prob[0][j].item()
            marker = "👑" if j == 0 else f"{j+1}."
            print(f"  {marker} {class_name}: {prob:.3f}")
    
    accuracy = correct / total * 100
    print(f"\n📊 测试结果:")
    print(f"总样本数: {total}")
    print(f"正确预测: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    
    return accuracy

def test_random_data(model, device, num_samples=5):
    """使用随机数据测试模型"""
    print(f"\n=== 使用随机数据测试 ===")
    
    for i in range(num_samples):
        # 生成随机数据
        random_image = torch.randn(1, 3, 128, 128).to(device)
        
        with torch.no_grad():
            outputs = model(random_image)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
        
        predicted_class = CIFAR10_CLASSES[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()
        
        print(f"随机样本 {i+1}: {predicted_class} (置信度: {confidence:.3f})")

def main():
    """主函数"""
    print("SimpleViT 优化模型测试")
    print("=" * 50)
    
    # 设备选择
    device = get_device()
    
    # 检查模型文件
    model_path = 'best_simple_vit_optimized.pth'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 测试模型
    try:
        # 使用CIFAR-10测试样本
        accuracy = test_with_cifar10_samples(model, device, num_samples=10)
        
        # 使用随机数据测试
        test_random_data(model, device, num_samples=5)
        
        print(f"\n🎉 测试完成！")
        print(f"💡 模型在随机样本上的准确率: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    main()
