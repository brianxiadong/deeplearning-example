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

# 全局变量，用于缓存模型
_cached_model = None
_cached_device = None

def predict_single_image(image_path, model_path='best_simple_vit_optimized.pth', top_k=5, verbose=True):
    """
    封装的预测方法 - 只需提供图片路径即可获得预测结果

    Args:
        image_path (str): 图片文件路径
        model_path (str): 模型文件路径，默认为 'best_simple_vit_optimized.pth'
        top_k (int): 返回前k个预测结果，默认为5
        verbose (bool): 是否显示详细信息，默认为True

    Returns:
        dict: 包含预测结果的字典
        {
            'success': bool,           # 是否预测成功
            'top_prediction': str,     # 最高预测类别
            'confidence': float,       # 最高预测的置信度
            'all_predictions': list,   # 所有top_k预测结果 [(class_name, probability), ...]
            'image_size': tuple,       # 原始图像尺寸
            'error': str              # 错误信息（如果有）
        }
    """
    global _cached_model, _cached_device

    result = {
        'success': False,
        'top_prediction': None,
        'confidence': 0.0,
        'all_predictions': [],
        'image_size': None,
        'error': None
    }

    try:
        # 检查图片文件
        if not os.path.exists(image_path):
            result['error'] = f"图片文件不存在: {image_path}"
            if verbose:
                print(f"❌ {result['error']}")
            return result

        # 检查模型文件
        if not os.path.exists(model_path):
            result['error'] = f"模型文件不存在: {model_path}"
            if verbose:
                print(f"❌ {result['error']}")
            return result

        # 初始化设备和模型（使用缓存避免重复加载）
        if _cached_model is None or _cached_device is None:
            if verbose:
                print("🔧 初始化模型...")
            _cached_device = get_device()
            _cached_model = load_model(model_path, _cached_device)

        # 预处理图像
        if verbose:
            print(f"🖼️  处理图像: {os.path.basename(image_path)}")

        image_tensor, original_image = preprocess_image(image_path)
        if image_tensor is None:
            result['error'] = "图像预处理失败"
            return result

        result['image_size'] = original_image.size

        # 进行预测
        if verbose:
            print("🔮 开始预测...")

        predictions = predict_image(_cached_model, image_tensor, _cached_device, top_k)

        # 填充结果
        result['success'] = True
        result['top_prediction'] = predictions[0][0]
        result['confidence'] = predictions[0][1]
        result['all_predictions'] = predictions

        # 显示结果（如果需要）
        if verbose:
            display_results(predictions, image_path)
            analyze_prediction(predictions)

        return result

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"❌ 预测过程中出现错误: {e}")
        return result

def simple_predict(image_path):
    """
    最简单的预测接口 - 只返回最可能的类别和置信度

    Args:
        image_path (str): 图片文件路径

    Returns:
        tuple: (predicted_class, confidence) 或 (None, 0.0) 如果失败
    """
    result = predict_single_image(image_path, verbose=False)
    if result['success']:
        return result['top_prediction'], result['confidence']
    else:
        return None, 0.0

def batch_predict(image_paths, verbose=True):
    """
    批量预测多张图片

    Args:
        image_paths (list): 图片路径列表
        verbose (bool): 是否显示详细信息

    Returns:
        list: 每张图片的预测结果字典列表
    """
    results = []

    if verbose:
        print(f"📦 开始批量预测 {len(image_paths)} 张图片...")

    for i, image_path in enumerate(image_paths):
        if verbose:
            print(f"\n--- 图片 {i+1}/{len(image_paths)} ---")

        result = predict_single_image(image_path, verbose=verbose)
        results.append(result)

    if verbose:
        # 显示批量预测汇总
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 批量预测完成:")
        print(f"   成功: {successful}/{len(image_paths)}")
        print(f"   失败: {len(image_paths) - successful}/{len(image_paths)}")

    return results

def main():
    """主函数 - 演示如何使用封装的方法"""
    print("SimpleViT 单张图片分类测试")
    print("=" * 50)

    # 示例1: 使用详细预测方法
    image_path = '../../images/188451751860259_.pic.jpg'
    print("🔥 示例1: 详细预测")
    result = predict_single_image(image_path)

    if result['success']:
        print(f"\n✅ 预测成功!")
        print(f"   最可能类别: {result['top_prediction']}")
        print(f"   置信度: {result['confidence']:.3f}")
    else:
        print(f"\n❌ 预测失败: {result['error']}")

    # 示例2: 使用简单预测方法
    print(f"\n🔥 示例2: 简单预测")
    predicted_class, confidence = simple_predict(image_path)
    if predicted_class:
        print(f"   结果: {predicted_class} (置信度: {confidence:.3f})")
    else:
        print(f"   预测失败")

    print(f"\n💡 使用说明:")
    print(f"   1. predict_single_image(path) - 完整预测，返回详细结果")
    print(f"   2. simple_predict(path) - 简单预测，只返回类别和置信度")
    print(f"   3. batch_predict([paths]) - 批量预测多张图片")

if __name__ == "__main__":
    main()
