#!/usr/bin/env python3
"""
Simple ViT 测试用例
测试 Simple Vision Transformer 的基本功能
"""

import torch
import torch.nn as nn
import sys
import os

# 添加上级目录到路径，以便导入vit_pytorch模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vit_pytorch import SimpleViT

def test_simple_vit_basic():
    """测试 Simple ViT 的基本功能"""
    print("=== Simple ViT 基本功能测试 ===")
    
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    
    # 创建 Simple ViT 模型
    model = SimpleViT(
        image_size=224,      # 输入图像尺寸 224x224
        patch_size=16,       # 每个patch的尺寸 16x16
        num_classes=1000,    # 分类数量（如ImageNet的1000类）
        dim=512,             # 模型维度
        depth=6,             # Transformer层数
        heads=8,             # 多头注意力的头数
        mlp_dim=1024,        # MLP隐藏层维度
        channels=3,          # 输入图像通道数（RGB）
        dim_head=64          # 每个注意力头的维度
    )
    
    # 创建随机输入数据：批次大小为2，3通道，224x224图像
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"输出张量形状: {output.shape}")
    print(f"期望输出形状: ({batch_size}, {1000})")
    
    # 验证输出形状是否正确
    assert output.shape == (batch_size, 1000), f"输出形状错误: {output.shape}"
    
    # 验证输出是否包含有效数值
    assert not torch.isnan(output).any(), "输出包含NaN值"
    assert not torch.isinf(output).any(), "输出包含无穷值"
    
    print("✅ 基本功能测试通过")
    return True


def test_simple_vit_different_sizes():
    """测试不同输入尺寸的 Simple ViT"""
    print("\n=== Simple ViT 不同输入尺寸测试 ===")
    
    # 测试配置列表
    test_configs = [
        {"image_size": 32, "patch_size": 4, "dim": 128, "depth": 3},
        {"image_size": 64, "patch_size": 8, "dim": 256, "depth": 4},
        {"image_size": 128, "patch_size": 16, "dim": 384, "depth": 5},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n测试配置 {i+1}: {config}")
        
        # 创建模型
        model = SimpleViT(
            image_size=config["image_size"],
            patch_size=config["patch_size"],
            num_classes=10,  # 简化为10类
            dim=config["dim"],
            depth=config["depth"],
            heads=4,
            mlp_dim=config["dim"] * 2,
            channels=3,
            dim_head=64
        )
        
        # 创建对应尺寸的输入
        input_tensor = torch.randn(1, 3, config["image_size"], config["image_size"])
        
        # 前向传播
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"  输入形状: {input_tensor.shape}")
        print(f"  输出形状: {output.shape}")
        
        # 验证输出
        assert output.shape == (1, 10), f"配置{i+1}输出形状错误"
        assert not torch.isnan(output).any(), f"配置{i+1}输出包含NaN"
        
        print(f"  ✅ 配置{i+1}测试通过")
    
    print("\n✅ 不同尺寸测试全部通过")
    return True


def test_simple_vit_gradients():
    """测试 Simple ViT 的梯度计算"""
    print("\n=== Simple ViT 梯度计算测试 ===")
    
    # 创建小规模模型进行梯度测试
    model = SimpleViT(
        image_size=32,
        patch_size=8,
        num_classes=5,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128,
        channels=3,
        dim_head=32
    )
    
    # 创建输入和目标
    input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)
    target = torch.tensor([2])  # 目标类别
    
    # 前向传播
    output = model(input_tensor)
    
    # 计算损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"参数 {name}: 梯度形状 {param.grad.shape}, 梯度范数 {param.grad.norm().item():.6f}")
            
            # 验证梯度不为零且不包含NaN
            assert not torch.isnan(param.grad).any(), f"参数{name}梯度包含NaN"
            assert not torch.isinf(param.grad).any(), f"参数{name}梯度包含无穷值"
    
    assert has_grad, "模型没有计算梯度"
    print("✅ 梯度计算测试通过")
    return True


def test_model_components():
    """测试模型各个组件"""
    print("\n=== 模型组件测试 ===")
    
    # 创建模型
    model = SimpleViT(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128
    )
    
    # 测试patch embedding
    input_tensor = torch.randn(1, 3, 32, 32)
    
    # 手动进行patch embedding
    patches = model.to_patch_embedding(input_tensor)
    print(f"Patch embedding输出形状: {patches.shape}")
    
    # 计算期望的patch数量
    expected_patches = (32 // 8) * (32 // 8)  # 4 * 4 = 16
    assert patches.shape == (1, expected_patches, 64), f"Patch embedding形状错误: {patches.shape}"
    
    # 测试位置编码
    pos_embed = model.pos_embedding
    print(f"位置编码形状: {pos_embed.shape}")
    assert pos_embed.shape == (expected_patches, 64), f"位置编码形状错误: {pos_embed.shape}"
    
    # 测试完整前向传播
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"最终输出形状: {output.shape}")
    assert output.shape == (1, 10), f"最终输出形状错误: {output.shape}"
    
    print("✅ 模型组件测试通过")
    return True


def main():
    """主测试函数"""
    print("开始 Simple ViT 测试...")
    
    try:
        # 运行所有测试
        test_simple_vit_basic()
        test_simple_vit_different_sizes()
        test_simple_vit_gradients()
        test_model_components()
        
        print("\n🎉 所有测试都通过了！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 