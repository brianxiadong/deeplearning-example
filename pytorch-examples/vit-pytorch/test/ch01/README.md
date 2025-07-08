# Chapter 01: Simple Vision Transformer 基础教程

本章节包含 SimpleViT 的完整学习材料，从基础测试到实际训练和推理。

## 📁 文件说明

### 1. `01-simple_vit.py` - 基础功能测试
**功能**：验证 SimpleViT 模型的基本功能和正确性

**测试内容**：
- ✅ 基本功能测试：模型创建、前向传播、输出形状验证
- ✅ 不同输入尺寸测试：测试模型对不同图像尺寸的适应性
- ✅ 梯度计算测试：验证反向传播和梯度计算的正确性
- ✅ 模型组件测试：测试各个组件的输出形状和数据流
- ✅ 智能设备检测：自动选择最优计算设备

**运行方式**：
```bash
cd pytorch-examples/vit-pytorch
python test/ch01/01-simple_vit.py
```

**期望输出**：
```
开始 Simple ViT 测试...
=== Simple ViT 基本功能测试 ===
模型参数总数: 21,123,456
✅ 基本功能测试通过
✅ 不同尺寸测试全部通过
✅ 梯度计算测试通过
✅ 模型组件测试通过
🎉 所有测试都通过了！
```

### 2. `train_simple_vit.py` - 完整训练脚本
**功能**：在 CIFAR-10 数据集上训练 SimpleViT 模型

**主要特性**：
- 🔄 智能数据管理：自动检测已存在数据集，避免重复下载
- 📊 实时训练进度显示
- 💾 自动保存最佳模型
- 📈 训练历史记录
- ⚡ 智能设备选择：CUDA > Apple MPS > CPU

**训练配置**：
- 数据集：CIFAR-10 (10类，50K训练+10K测试)
- 图像尺寸：224×224 (从32×32调整)
- 批次大小：32
- 训练轮数：10
- 优化器：AdamW (lr=3e-4, weight_decay=1e-4)
- 学习率调度：CosineAnnealingLR

**运行方式**：
```bash
cd pytorch-examples/vit-pytorch
python test/ch01/train_simple_vit.py
```

**输出文件**：
- `best_simple_vit.pth` - 最佳模型权重
- `final_simple_vit.pth` - 最终模型权重和训练历史

### 3. `inference_simple_vit.py` - 模型推理脚本
**功能**：使用训练好的模型进行图像分类预测

**推理模式**：
- 🖼️ 单张图像预测：对指定图像进行分类
- 📁 批量图像预测：处理整个文件夹的图像
- 🎲 随机数据演示：使用随机数据测试模型

**支持格式**：jpg, jpeg, png, bmp, tiff

**运行方式**：
```bash
cd pytorch-examples/vit-pytorch
python test/ch01/inference_simple_vit.py
```

**交互示例**：
```
=== 选择演示模式 ===
1. 单张图像预测
2. 批量图像预测
3. 随机数据演示
请选择模式 (1/2/3): 1

请输入图像路径: /path/to/image.jpg

=== 预测图像: image.jpg ===
预测类别: cat
置信度: 0.8234

前5个预测结果:
  1. cat: 0.8234
  2. dog: 0.1123
  3. horse: 0.0234
  4. deer: 0.0198
  5. bird: 0.0156
```

### 4. `check_device.py` - 设备检测工具
**功能**：检测当前系统的 PyTorch 设备支持情况

**检测内容**：
- 🔍 PyTorch 版本和系统信息
- 🚀 CUDA GPU 支持检测
- 🍎 Apple MPS 支持检测
- 💻 CPU 性能测试
- 🎯 设备推荐建议

**运行方式**：
```bash
python test/ch01/check_device.py
```

**输出示例**：
```
🍎 Apple MPS 支持检查
========================================
✅ MPS 可用
✅ Apple Silicon 加速已启用
✅ MPS 张量操作测试通过
🚀 MPS 相对 CPU 加速比: 3.2x

🎯 推荐设备配置
========================================
🍎 推荐使用 Apple MPS 进行训练
```

## 🚀 快速开始

### 步骤0：检测设备支持（推荐）
```bash
python test/ch01/check_device.py
```
检测当前系统的设备支持情况，获取最优配置建议。

### 步骤1：运行基础测试
```bash
python test/ch01/01-simple_vit.py
```
确保模型实现正确，所有测试通过。

### 步骤2：训练模型
```bash
python test/ch01/train_simple_vit.py
```
在 CIFAR-10 上训练模型，时间取决于设备：
- CUDA GPU: ~20-30分钟
- Apple MPS: ~40-60分钟
- CPU: ~2-4小时

### 步骤3：测试推理
```bash
python test/ch01/inference_simple_vit.py
```
使用训练好的模型进行预测。

## 📊 预期性能

在 CIFAR-10 数据集上的预期性能：

| 指标 | CUDA GPU | Apple MPS | CPU |
|------|----------|-----------|-----|
| 训练准确率 | ~85-90% | ~85-90% | ~85-90% |
| 测试准确率 | ~75-80% | ~75-80% | ~75-80% |
| 训练时间/epoch | ~2-3分钟 | ~4-6分钟 | ~15-25分钟 |
| 模型大小 | ~85MB | ~85MB | ~85MB |
| 参数数量 | ~21M | ~21M | ~21M |

### 🍎 Apple Silicon 优化

在 Apple M1/M2/M3 芯片上，脚本会自动启用 MPS (Metal Performance Shaders) 加速：

```
🍎 使用 Apple Silicon MPS 加速
```

**MPS 优势**：
- 🚀 比 CPU 快 3-5 倍
- 💡 统一内存架构，内存使用更高效
- 🔋 功耗更低，发热更少
- 🎯 专为 Apple Silicon 优化

## 💾 数据集管理

### 数据集存储位置
- 数据集保存在：`./data/` 目录下
- CIFAR-10 解压后路径：`./data/cifar-10-batches-py/`

### 智能下载机制
训练脚本会自动检测数据集是否存在：
- ✅ **已存在**：跳过下载，直接使用
- 📥 **不存在**：自动下载并解压

### 手动管理数据集
```bash
# 查看数据集状态
ls -la ./data/

# 删除数据集（如需重新下载）
rm -rf ./data/cifar-10-batches-py/

# 数据集大小约 170MB
du -sh ./data/
```

## 🔧 自定义配置

### 修改模型参数
在 `train_simple_vit.py` 中修改模型配置：

```python
model = SimpleViT(
    image_size=224,      # 图像尺寸
    patch_size=16,       # patch大小 (8, 16, 32)
    num_classes=10,      # 类别数量
    dim=512,             # 模型维度 (256, 512, 1024)
    depth=6,             # 层数 (3, 6, 12)
    heads=8,             # 注意力头数 (4, 8, 16)
    mlp_dim=1024,        # MLP维度
)
```

### 修改训练参数
```python
batch_size = 32          # 批次大小
num_epochs = 10          # 训练轮数
learning_rate = 3e-4     # 学习率
weight_decay = 1e-4      # 权重衰减
```

## 🐛 常见问题

### Q1: CUDA out of memory
**解决方案**：
- 减小 `batch_size` (如改为16或8)
- 减小模型 `dim` (如改为256)
- 使用 CPU 训练 (较慢)

### Q2: 训练准确率不提升
**解决方案**：
- 检查学习率是否合适
- 增加训练轮数
- 调整数据增强策略

### Q3: 推理时找不到模型文件
**解决方案**：
- 确保先运行训练脚本
- 检查 `best_simple_vit.pth` 文件是否存在
- 使用随机数据演示模式测试

### Q4: Apple Silicon MPS 相关问题
**Q**: 如何确认 MPS 是否可用？
**A**: 运行脚本时查看输出：
```bash
🍎 使用 Apple Silicon MPS 加速  # MPS 可用
💻 使用 CPU 设备              # MPS 不可用，回退到 CPU
```

**Q**: MPS 训练时出现错误怎么办？
**A**: 可能的解决方案：
- 更新 PyTorch 到最新版本：`pip install --upgrade torch torchvision`
- 如果仍有问题，可以强制使用 CPU：
```python
# 在脚本开头添加
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

**Q**: 如何检查 PyTorch MPS 支持？
**A**: 运行以下命令：
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"PyTorch version: {torch.__version__}")
```

## 📁 完整文件列表

```
test/ch01/
├── README.md                  # 📖 章节说明文档
├── check_device.py           # 🔧 设备检测工具
├── 01-simple_vit.py          # ✅ 基础功能测试
├── train_simple_vit.py       # 🚀 完整训练脚本
└── inference_simple_vit.py   # 🔍 推理和部署脚本
```

## 📚 相关资源

- [SimpleViT 完整教程](../tutorial/01_SimpleViT_Tutorial.md)
- [Vision Transformer 原论文](https://arxiv.org/abs/2010.11929)
- [PyTorch 官方文档](https://pytorch.org/docs/)

## 🎯 下一步

完成本章节后，建议：

1. 尝试不同的模型配置
2. 在其他数据集上训练
3. 学习标准 ViT 实现
4. 探索 ViT 的变种模型

祝学习愉快！🚀
