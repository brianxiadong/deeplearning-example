#!/usr/bin/env python3
"""
测试单张图片的分类效果
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import sys
import os

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

def preprocess_image(image_path):
    """预处理图像"""
    print(f"🖼️  处理图像: {image_path}")
    
    # 与训练时相同的预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 与训练时相同
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载并预处理图像
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"📏 原始图像尺寸: {image.size}")
        
        image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
        print(f"📦 预处理后张量形状: {image_tensor.shape}")
        
        return image_tensor, image
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        return None, None

def predict_image(model, image_tensor, device, top_k=5):
    """对图像进行预测"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 前向传播
        outputs = model(image_tensor)
        
        # 计算概率
        probabilities = F.softmax(outputs, dim=1)
        
        # 获取top-k预测
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            class_name = CIFAR10_CLASSES[class_idx]
            results.append((class_name, prob))
    
    return results

def display_results(results, image_path):
    """显示预测结果"""
    print(f"\n🎯 预测结果 - {os.path.basename(image_path)}")
    print("=" * 50)
    
    # 显示最高预测
    top_class, top_prob = results[0]
    print(f"🏆 最可能的类别: {top_class}")
    print(f"🎲 置信度: {top_prob:.3f} ({top_prob*100:.1f}%)")
    
    # 显示所有top-k结果
    print(f"\n📊 Top-{len(results)} 预测:")
    for i, (class_name, prob) in enumerate(results):
        if i == 0:
            marker = "👑"
        elif i == 1:
            marker = "🥈"
        elif i == 2:
            marker = "🥉"
        else:
            marker = f"{i+1}."
        
        bar_length = int(prob * 30)  # 进度条长度
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        print(f"  {marker} {class_name:12} {prob:.3f} |{bar}| {prob*100:.1f}%")

def analyze_prediction(results):
    """分析预测结果"""
    print(f"\n🔍 预测分析:")
    
    top_class, top_prob = results[0]
    
    # 置信度分析
    if top_prob > 0.8:
        confidence_level = "非常高"
        emoji = "🎯"
    elif top_prob > 0.6:
        confidence_level = "高"
        emoji = "✅"
    elif top_prob > 0.4:
        confidence_level = "中等"
        emoji = "⚠️"
    else:
        confidence_level = "低"
        emoji = "❓"
    
    print(f"  {emoji} 置信度水平: {confidence_level}")
    
    # 竞争分析
    if len(results) > 1:
        second_prob = results[1][1]
        gap = top_prob - second_prob
        
        if gap > 0.3:
            print(f"  🎪 预测很明确，与第二名差距: {gap:.3f}")
        elif gap > 0.1:
            print(f"  🤔 预测较明确，与第二名差距: {gap:.3f}")
        else:
            print(f"  😕 预测不够明确，与第二名差距仅: {gap:.3f}")

def main():
    """主函数"""
    print("SimpleViT 单张图片分类测试")
    print("=" * 50)
    
    # 设备选择
    device = get_device()
    
    # 检查模型文件
    model_path = 'best_simple_vit_optimized.pth'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 图片路径
    image_path = '../../images/188451751860259_.pic.jpg'
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    try:
        # 加载模型
        model = load_model(model_path, device)
        
        # 预处理图像
        image_tensor, original_image = preprocess_image(image_path)
        if image_tensor is None:
            return
        
        # 进行预测
        print(f"\n🔮 开始预测...")
        results = predict_image(model, image_tensor, device, top_k=5)
        
        # 显示结果
        display_results(results, image_path)
        
        # 分析预测
        analyze_prediction(results)
        
        print(f"\n💡 注意: 此模型是在CIFAR-10数据集上训练的，只能识别以下10个类别:")
        print(f"   {', '.join(CIFAR10_CLASSES)}")
        print(f"   如果您的图片不属于这些类别，预测结果可能不准确。")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
