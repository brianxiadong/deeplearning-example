#!/usr/bin/env python3
"""
SimpleViT 训练示例
演示如何使用 SimpleViT 进行图像分类训练
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
from tqdm import tqdm

# 添加上级目录到路径，以便导入vit_pytorch模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vit_pytorch import SimpleViT

def create_data_loaders(batch_size=32, num_workers=4):
    """
    创建数据加载器
    使用 CIFAR-10 数据集作为示例
    """
    print("=== 准备数据集 ===")
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到 ViT 期望的尺寸
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 下载并加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"批次大小: {batch_size}")
    
    return train_loader, test_loader

def create_model():
    """
    创建 SimpleViT 模型
    """
    print("=== 创建模型 ===")
    
    model = SimpleViT(
        image_size=224,      # 输入图像尺寸
        patch_size=16,       # patch 尺寸
        num_classes=10,      # CIFAR-10 有 10 个类别
        dim=512,             # 模型维度
        depth=6,             # Transformer 层数
        heads=8,             # 多头注意力头数
        mlp_dim=1024,        # MLP 隐藏层维度
        channels=3,          # RGB 图像
        dim_head=64          # 每个注意力头的维度
    )
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个 epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, device):
    """
    验证模型
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Validating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def main():
    """
    主训练函数
    """
    print("SimpleViT 训练示例")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练参数
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    weight_decay = 1e-4
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(batch_size)
    
    # 创建模型
    model = create_model()
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"\n=== 开始训练 ===")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"权重衰减: {weight_decay}")
    
    # 训练循环
    best_acc = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
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
        
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
        print(f'  学习率: {scheduler.get_last_lr()[0]:.6f}')
        print(f'  耗时: {epoch_time:.2f}s')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'best_simple_vit.pth')
            print(f'  ✅ 保存最佳模型 (准确率: {best_acc:.2f}%)')
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
    }, 'final_simple_vit.pth')
    
    print("模型已保存:")
    print("  - best_simple_vit.pth (最佳模型)")
    print("  - final_simple_vit.pth (最终模型)")

if __name__ == "__main__":
    main()
