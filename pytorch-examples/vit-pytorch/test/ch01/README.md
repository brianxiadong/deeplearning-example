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
- 🔄 自动数据下载和预处理
- 📊 实时训练进度显示
- 💾 自动保存最佳模型
- 📈 训练历史记录
- ⚡ GPU/CPU 自动检测

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

## 🚀 快速开始

### 步骤1：运行基础测试
```bash
python test/ch01/01-simple_vit.py
```
确保模型实现正确，所有测试通过。

### 步骤2：训练模型
```bash
python test/ch01/train_simple_vit.py
```
在 CIFAR-10 上训练模型，大约需要 20-30 分钟（GPU）。

### 步骤3：测试推理
```bash
python test/ch01/inference_simple_vit.py
```
使用训练好的模型进行预测。

## 📊 预期性能

在 CIFAR-10 数据集上的预期性能：

| 指标 | 数值 |
|------|------|
| 训练准确率 | ~85-90% |
| 测试准确率 | ~75-80% |
| 训练时间 | ~20-30分钟 (GPU) |
| 模型大小 | ~85MB |
| 参数数量 | ~21M |

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
