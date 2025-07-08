#!/usr/bin/env python3
"""
SimpleViT 训练示例 - 内存优化版本
演示如何使用 SimpleViT 进行图像分类训练，包含内存优化技术
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import sys
import os
import time
import gc
import psutil
from tqdm import tqdm

# 添加上级目录到路径，以便导入vit_pytorch模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from vit_pytorch import SimpleViT

def get_memory_info():
    """获取内存使用信息"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def clear_memory(device):
    """清理内存"""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

def create_data_loaders(batch_size=16, num_workers=2):
    """
    创建数据加载器 - 内存优化版本
    使用 CIFAR-10 数据集作为示例
    """
    print("=== 准备数据集 ===")

    # 检查数据集是否已存在
    data_dir = './data'
    cifar10_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    cifar10_tar = os.path.join(data_dir, 'cifar-10-python.tar.gz')

    # 检查解压后的数据或压缩包是否存在
    if os.path.exists(cifar10_dir):
        print("✅ 检测到已解压的CIFAR-10数据集，跳过下载")
        download_flag = False
    elif os.path.exists(cifar10_tar):
        print("✅ 检测到CIFAR-10压缩包，跳过下载，将自动解压")
        download_flag = False
    else:
        print("📥 CIFAR-10数据集不存在，开始下载...")
        download_flag = True

    # 数据预处理 - 使用更小的图像尺寸以节省内存
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),  # 减小到128x128以节省内存
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),  # 减小到128x128以节省内存
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集（只在需要时下载）
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download_flag, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download_flag, transform=transform_test
    )

    # 创建数据加载器 - 内存优化配置
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False  # 禁用pin_memory以节省内存
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False  # 禁用pin_memory以节省内存
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: 128x128 (内存优化)")

    return train_loader, test_loader

def create_model():
    """
    创建 SimpleViT 模型 - 内存优化版本
    """
    print("=== 创建模型 ===")

    model = SimpleViT(
        image_size=128,      # 减小输入图像尺寸以节省内存
        patch_size=16,       # patch 尺寸
        num_classes=10,      # CIFAR-10 有 10 个类别
        dim=384,             # 减小模型维度以节省内存
        depth=4,             # 减少Transformer层数
        heads=6,             # 减少多头注意力头数
        mlp_dim=768,         # 减小MLP隐藏层维度
        channels=3,          # RGB 图像
        dim_head=64          # 每个注意力头的维度
    )

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print("📊 内存优化配置:")
    print(f"  - 图像尺寸: 128x128 (原224x224)")
    print(f"  - 模型维度: 384 (原512)")
    print(f"  - 层数: 4 (原6)")
    print(f"  - 注意力头数: 6 (原8)")

    return model

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, accumulation_steps=2):
    """
    训练一个 epoch - 内存优化版本，支持梯度累积
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    # 梯度累积计数器
    accumulation_count = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 梯度累积：将损失除以累积步数
        loss = loss / accumulation_steps

        # 反向传播
        loss.backward()

        accumulation_count += 1

        # 当达到累积步数时，更新参数
        if accumulation_count % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # 统计
        running_loss += loss.item() * accumulation_steps  # 恢复原始损失值
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 定期清理内存
        if batch_idx % 100 == 0:
            clear_memory(device)

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'Mem': f'{get_memory_info():.0f}MB'
        })

    # 处理剩余的梯度
    if accumulation_count % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, device):
    """
    验证模型 - 内存优化版本
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Validating')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 定期清理内存
            if batch_idx % 50 == 0:
                clear_memory(device)

            # 更新进度条
            pbar.set_postfix({
                'Acc': f'{100.*correct/total:.2f}%',
                'Mem': f'{get_memory_info():.0f}MB'
            })

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc

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
    主训练函数 - 内存优化版本
    """
    print("SimpleViT 训练示例 - 内存优化版本")
    print("=" * 60)

    # 智能设备选择
    device = get_device()

    # 训练参数 - 内存优化配置
    batch_size = 16          # 减小批次大小
    accumulation_steps = 2   # 梯度累积步数，有效批次大小 = 16 * 2 = 32
    num_epochs = 8           # 减少训练轮数
    learning_rate = 2e-4     # 稍微降低学习率
    weight_decay = 1e-4
    patience = 3             # 早停耐心值

    print(f"💾 内存优化配置:")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 梯度累积步数: {accumulation_steps}")
    print(f"  - 有效批次大小: {batch_size * accumulation_steps}")
    print(f"  - 初始内存使用: {get_memory_info():.0f}MB")

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(batch_size)

    # 创建模型
    model = create_model()
    model = model.to(device)

    # 清理初始内存
    clear_memory(device)
    print(f"  - 模型加载后内存: {get_memory_info():.0f}MB")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n=== 开始训练 ===")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"权重衰减: {weight_decay}")
    print(f"早停耐心值: {patience}")

    # 训练循环
    best_acc = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, accumulation_steps
        )

        # 验证
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        epoch_time = time.time() - start_time
        current_memory = get_memory_info()

        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
        print(f'  耗时: {epoch_time:.2f}s, 内存: {current_memory:.0f}MB')

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0  # 重置早停计数器
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_simple_vit_optimized.pth')
            print(f'  ✅ 保存最佳模型 (准确率: {best_acc:.2f}%)')
        else:
            patience_counter += 1
            print(f'  ⏳ 早停计数器: {patience_counter}/{patience}')

        # 早停检查
        if patience_counter >= patience:
            print(f"\n🛑 早停触发！连续{patience}个epoch无改善")
            break

        # 定期清理内存
        clear_memory(device)

    print(f"\n=== 训练完成 ===")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print(f"实际训练轮数: {epoch}")
    print(f"最终内存使用: {get_memory_info():.0f}MB")

    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_acc': best_acc,
    }, 'final_simple_vit_optimized.pth')

    print("\n📁 模型已保存:")
    print("  - best_simple_vit_optimized.pth (最佳模型)")
    print("  - final_simple_vit_optimized.pth (最终模型)")

    # 最终内存清理
    clear_memory(device)

if __name__ == "__main__":
    main()
