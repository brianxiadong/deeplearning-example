#!/usr/bin/env python3
"""
SimpleViT 推理示例 - 内存优化版本
演示如何使用训练好的 SimpleViT 模型进行预测
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
import numpy as np

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

def load_model(model_path, device):
    """
    加载训练好的模型 - 内存优化版本
    """
    print(f"=== 加载模型 ===")
    print(f"模型路径: {model_path}")

    # 创建模型架构 - 与训练时的优化配置相同
    model = SimpleViT(
        image_size=128,      # 优化后的尺寸
        patch_size=16,
        num_classes=10,
        dim=384,             # 优化后的维度
        depth=4,             # 优化后的层数
        heads=6,             # 优化后的头数
        mlp_dim=768,         # 优化后的MLP维度
        channels=3,
        dim_head=64
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功!")
    if 'best_acc' in checkpoint:
        print(f"模型最佳准确率: {checkpoint['best_acc']:.2f}%")
    
    return model

def preprocess_image(image_path):
    """
    图像预处理
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 优化后的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    return input_tensor, image

def predict_single_image(model, image_path, device):
    """
    对单张图像进行预测
    """
    print(f"\n=== 预测图像: {image_path} ===")
    
    # 预处理图像
    input_tensor, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # 获取预测结果
    predicted_class = CIFAR10_CLASSES[predicted.item()]
    confidence_score = confidence.item()
    
    print(f"预测类别: {predicted_class}")
    print(f"置信度: {confidence_score:.4f}")
    
    # 显示前5个预测结果
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    print(f"\n前5个预测结果:")
    for i in range(5):
        class_name = CIFAR10_CLASSES[top5_indices[0][i]]
        prob = top5_prob[0][i].item()
        print(f"  {i+1}. {class_name}: {prob:.4f}")
    
    return predicted_class, confidence_score

def predict_batch_images(model, image_folder, device, max_images=10):
    """
    批量预测图像
    """
    print(f"\n=== 批量预测图像 ===")
    print(f"图像文件夹: {image_folder}")
    
    # 获取图像文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    image_files = image_files[:max_images]  # 限制图像数量
    
    if not image_files:
        print("未找到图像文件!")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 批量预测
    results = []
    for image_path in image_files:
        try:
            predicted_class, confidence = predict_single_image(
                model, image_path, device
            )
            results.append({
                'image': os.path.basename(image_path),
                'prediction': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
    
    # 显示结果汇总
    print(f"\n=== 批量预测结果汇总 ===")
    for result in results:
        print(f"{result['image']}: {result['prediction']} ({result['confidence']:.4f})")
    
    return results

def demo_with_sample_data(model, device):
    """
    使用示例数据进行演示
    """
    print(f"\n=== 示例数据演示 ===")
    
    # 创建随机图像数据
    random_image = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(random_image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CIFAR10_CLASSES[predicted.item()]
    confidence_score = confidence.item()
    
    print(f"随机图像预测:")
    print(f"  预测类别: {predicted_class}")
    print(f"  置信度: {confidence_score:.4f}")
    
    # 显示所有类别的概率
    print(f"\n所有类别概率:")
    for i, class_name in enumerate(CIFAR10_CLASSES):
        prob = probabilities[0][i].item()
        print(f"  {class_name}: {prob:.4f}")

def get_device():
    """
    智能设备选择：优先级 CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 使用 CUDA 设备: {device_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🍎 使用 Apple Silicon MPS 加速")
    else:
        device = torch.device('cpu')
        print(f"💻 使用 CPU 设备")

    return device

def main():
    """
    主函数
    """
    print("SimpleViT 推理示例")
    print("=" * 50)

    # 智能设备选择
    device = get_device()
    
    # 模型路径 - 优先使用优化版本
    model_paths = ['best_simple_vit_optimized.pth', 'best_simple_vit.pth']
    model_path = None

    # 查找可用的模型文件
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    # 检查模型文件是否存在
    if model_path is None:
        print(f"⚠️  模型文件不存在!")
        print("💡 请先运行以下命令训练模型:")
        print("   python train_simple_vit.py")
        print("\n🎲 现在使用未训练的模型进行演示...")

        # 创建一个未训练的模型进行演示
        model = SimpleViT(
            image_size=128,      # 使用优化配置
            patch_size=16,
            num_classes=10,
            dim=384,
            depth=4,
            heads=6,
            mlp_dim=768,
            channels=3,
            dim_head=64
        ).to(device)
        model.eval()

        demo_with_sample_data(model, device)
        return

    print(f"📂 找到模型文件: {model_path}")
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 演示选项
    print(f"\n=== 选择演示模式 ===")
    print("1. 单张图像预测")
    print("2. 批量图像预测")
    print("3. 随机数据演示")
    
    try:
        choice = input("请选择模式 (1/2/3): ").strip()
        
        if choice == '1':
            image_path = input("请输入图像路径: ").strip()
            if os.path.exists(image_path):
                predict_single_image(model, image_path, device)
            else:
                print(f"图像文件 {image_path} 不存在!")
        
        elif choice == '2':
            folder_path = input("请输入图像文件夹路径: ").strip()
            if os.path.exists(folder_path):
                predict_batch_images(model, folder_path, device)
            else:
                print(f"文件夹 {folder_path} 不存在!")
        
        elif choice == '3':
            demo_with_sample_data(model, device)
        
        else:
            print("无效选择，使用随机数据演示")
            demo_with_sample_data(model, device)
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
