# Word2Vec模型详解：CBOW与Skip-gram

## 目录
1. [模型概述](#模型概述)
2. [理论基础](#理论基础)
3. [CBOW模型详解](#cbow模型详解)
4. [Skip-gram模型详解](#skip-gram模型详解)
5. [模型对比](#模型对比)
6. [代码实现详解](#代码实现详解)
7. [训练过程分析](#训练过程分析)
8. [应用场景](#应用场景)
9. [实验结果](#实验结果)
10. [总结](#总结)

---

## 模型概述

Word2Vec是一种用于生成词向量的神经网络模型，由Google在2013年提出。它包含两种主要架构：

- **CBOW (Continuous Bag of Words)**: 根据上下文词汇预测目标词
- **Skip-gram**: 根据目标词预测上下文词汇

这两种模型都能将词汇转换为高维密集向量，使得语义相近的词在向量空间中距离较近。

## 理论基础

### 基本假设
Word2Vec基于**分布式假设**：在相似上下文中出现的词汇具有相似的语义。

### 核心思想
通过神经网络学习词汇的分布式表示，使得：
1. 语义相似的词汇在向量空间中距离较近
2. 词汇间的语义关系通过向量运算体现
3. 支持词汇类比推理（如：国王-男人+女人=皇后）

---

## CBOW模型详解

### 模型架构

```
上下文词汇 → 嵌入层 → 平均池化 → 线性层 → 目标词概率
```

CBOW模型通过上下文词汇的平均嵌入来预测中心词汇。

### 代码实现

#### 1. 模型定义

```python
# 定义CBOW模型类，继承自PyTorch的神经网络模块
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        # 调用父类构造函数
        super(CBOWModel, self).__init__()
        # 创建词嵌入层，将词汇索引映射到密集向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 创建线性层，将嵌入向量映射到词汇表大小的输出
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, context_words):
        # 前向传播函数，context_words形状: [batch_size, context_size]
        # 获取上下文词汇的嵌入向量
        embeds = self.embedding(context_words)  # 形状: [batch_size, context_size, embedding_dim]
        # 计算上下文词汇嵌入的平均值
        mean_embed = torch.mean(embeds, dim=1)  # 形状: [batch_size, embedding_dim]
        # 通过线性层得到最终输出
        out = self.linear(mean_embed)  # 形状: [batch_size, vocab_size]
        return out
```

**关键特点**：
- 单一嵌入层：所有词汇共享同一个嵌入空间
- 平均池化：对上下文词汇嵌入求平均
- 分类任务：输出目标词汇的概率分布

#### 2. 数据集创建

```python
def create_cbow_dataset(corpus, window_size=2):
    """创建CBOW训练数据集"""
    # 初始化数据列表
    data = []
    # 遍历语料库，跳过边界位置
    for i in range(window_size, len(corpus) - window_size):
        # 初始化上下文词汇列表
        context = []
        # 当前位置的词汇作为目标词
        target = corpus[i]
        
        # 收集目标词周围的上下文词汇
        for j in range(i - window_size, i + window_size + 1):
            # 跳过目标词本身
            if j != i:
                # 添加上下文词汇到列表
                context.append(corpus[j])
        
        # 将上下文和目标词作为训练样本
        data.append((context, target))
    
    # 返回训练数据集
    return data
```

**数据格式**：每个样本为 `([上下文词汇列表], 目标词)`

#### 3. 完形填空功能

```python
def cloze_test(model, sentence, mask_position, word_to_id, id_to_word, window_size=2, top_k=5):
    """完形填空函数"""
    # 将句子分割成词汇列表
    words = sentence.lower().replace('.', ' .').split()
    
    # 检查mask位置是否有效
    if mask_position >= len(words) or mask_position < 0:
        print(f"无效的mask位置: {mask_position}")
        return None
    
    # 保存被遮掩的真实词汇
    true_word = words[mask_position]
    print(f"被遮掩的词汇: '{true_word}'")
    
    # 提取上下文词汇（排除mask位置）
    context_words = []
    for i in range(max(0, mask_position - window_size), 
                   min(len(words), mask_position + window_size + 1)):
        if i != mask_position and words[i] in word_to_id:
            context_words.append(word_to_id[words[i]])
    
    # 进行预测
    with torch.no_grad():
        context_tensor = torch.tensor(context_words, dtype=torch.long).unsqueeze(0)
        output = model(context_tensor)
        probabilities = F.softmax(output, dim=1)
        
        # 获取top-k预测结果
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # 显示预测结果
        predictions = []
        for i in range(top_k):
            word_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            predicted_word = id_to_word[word_idx]
            predictions.append((predicted_word, prob))
            print(f"{predicted_word}\t\t{prob:.4f}")
        
        return predictions
```

**完形填空原理**：CBOW天然适合完形填空，因为它就是根据上下文预测目标词。

---

## Skip-gram模型详解

### 模型架构

```
目标词 → 目标嵌入层 ↘
                    → 相似度计算 → 上下文词概率
上下文词 → 上下文嵌入层 ↗
```

Skip-gram使用两个独立的嵌入层，通过负采样进行训练。

### 代码实现

#### 1. 模型定义

```python
# 定义Skip-gram模型类，继承自PyTorch的神经网络模块
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        # 调用父类构造函数
        super(SkipGramModel, self).__init__()
        # 创建输入词嵌入层，将目标词索引映射到密集向量
        self.target_embedding = nn.Embedding(vocab_size, embedding_dim)
        # 创建输出词嵌入层，用于预测上下文词
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, target_word, context_words):
        # 前向传播函数
        # target_word: [batch_size] - 目标词索引
        # context_words: [batch_size] - 上下文词索引
        
        # 获取目标词的嵌入向量
        target_embed = self.target_embedding(target_word)  # [batch_size, embedding_dim]
        # 获取上下文词的嵌入向量
        context_embed = self.context_embedding(context_words)  # [batch_size, embedding_dim]
        
        # 计算目标词和上下文词的相似度得分
        score = torch.sum(target_embed * context_embed, dim=1)  # [batch_size]
        
        return score
```

**关键特点**：
- 双嵌入层：目标词和上下文词有不同的嵌入空间
- 点积相似度：计算目标词和上下文词嵌入的内积
- 回归任务：输出相似度得分

#### 2. 负采样训练

```python
def negative_sampling(vocab_size, num_negatives=5):
    """负采样：随机选择负样本词汇"""
    # 随机生成负样本词汇索引
    negative_samples = torch.randint(0, vocab_size, (num_negatives,))
    return negative_samples

def train_skipgram(model, train_data, vocab_size, epochs=100, learning_rate=0.01, num_negatives=5):
    """训练Skip-gram模型（使用负采样）"""
    # 定义二元交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss()
    # 定义Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for target, context in train_data:
            # 正样本训练
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)
            positive_label = torch.tensor([1.0], dtype=torch.float)
            
            positive_score = model(target_tensor, context_tensor)
            positive_loss = criterion(positive_score, positive_label)
            
            # 负样本训练
            negative_samples = negative_sampling(vocab_size, num_negatives)
            negative_labels = torch.zeros(num_negatives, dtype=torch.float)
            
            target_repeated = target_tensor.repeat(num_negatives)
            negative_scores = model(target_repeated, negative_samples)
            negative_loss = criterion(negative_scores, negative_labels)
            
            # 总损失
            total_sample_loss = positive_loss + negative_loss
            
            # 反向传播和参数更新
            optimizer.zero_grad()
            total_sample_loss.backward()
            optimizer.step()
            
            total_loss += total_sample_loss.item()
    
    return losses
```

**负采样原理**：
- 正样本：真实的(目标词, 上下文词)对，标签为1
- 负样本：随机的(目标词, 非上下文词)对，标签为0
- 避免计算整个词汇表的softmax，提高训练效率

#### 3. 上下文预测功能

```python
def predict_context_words(model, target_word, word_to_id, id_to_word, top_k=5):
    """预测给定目标词的上下文词汇"""
    target_idx = word_to_id[target_word]
    target_tensor = torch.tensor([target_idx], dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        # 计算目标词与所有词汇的相似度得分
        scores = []
        vocab_size = len(word_to_id)
        
        for word_idx in range(vocab_size):
            context_tensor = torch.tensor([word_idx], dtype=torch.long)
            score = model(target_tensor, context_tensor)
            scores.append((word_idx, score.item()))
        
        # 按得分降序排序并返回top-k结果
        scores.sort(key=lambda x: x[1], reverse=True)
        
        predictions = []
        for i in range(min(top_k, len(scores))):
            word_idx, score = scores[i]
            predicted_word = id_to_word[word_idx]
            if predicted_word != target_word:
                predictions.append((predicted_word, score))
        
        return predictions
```

**上下文预测原理**：Skip-gram擅长预测给定词汇的可能上下文，适用于词汇扩展和同义词发现。

---

## 模型对比

| 特性 | CBOW | Skip-gram |
|------|------|-----------|
| **输入** | 上下文词汇 | 目标词 |
| **输出** | 目标词概率 | 上下文词得分 |
| **嵌入层** | 单一共享嵌入 | 双独立嵌入 |
| **训练数据量** | 较少 | 较多 |
| **计算复杂度** | 较低 | 较高 |
| **适用场景** | 完形填空、语言模型 | 词汇扩展、相似词发现 |
| **对频率的敏感性** | 对高频词效果好 | 对低频词效果好 |

### 数学表示

**CBOW目标函数**：
```
maximize Σ log P(w_t | context(w_t))
```

**Skip-gram目标函数**：
```
maximize Σ Σ log P(w_c | w_t)
```

其中：
- `w_t`: 目标词
- `w_c`: 上下文词
- `context(w_t)`: 目标词的上下文窗口

---

## 代码实现详解

### 共享组件

#### 1. 数据预处理

```python
def preprocess(text):
    """文本预处理函数"""
    # 转换为小写并分割单词
    word = text.lower().replace('.', ' .').split(' ')

    # 创建词汇到ID的映射
    word_to_id = {}
    id_to_word = {}
    for w in word:
        if w not in word_to_id:
            tmp = len(word_to_id)
            word_to_id[w] = tmp
            id_to_word[tmp] = w

    # 获得ID列表
    corpus = [word_to_id[w] for w in word]

    return corpus, word_to_id, id_to_word
```

#### 2. 相似度计算

```python
def cos_similarity(x, y, eps=1e-8):
    """计算余弦相似度"""
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)
```

#### 3. 词向量可视化

```python
def visualize_embeddings(word_vectors, word_to_id, max_words=50):
    """可视化词向量（使用PCA降维到2D）"""
    from sklearn.decomposition import PCA
    
    words = list(word_vectors.keys())[:max_words]
    vectors = [word_vectors[word] for word in words]
    
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
    
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title('Word Embeddings Visualization (PCA)')
    plt.show()
```

### 训练流程对比

#### CBOW训练流程
1. 提取上下文词汇窗口
2. 计算上下文词汇嵌入的平均值
3. 通过线性层预测目标词
4. 使用交叉熵损失进行优化

#### Skip-gram训练流程
1. 选择目标词和一个上下文词
2. 计算正样本得分和损失
3. 进行负采样，计算负样本损失
4. 使用二元交叉熵损失进行优化

---

## 训练过程分析

### 超参数设置

```python
# 共同超参数
window_size = 2        # 上下文窗口大小
embedding_dim = 100    # 词向量维度
epochs = 100          # 训练轮数
learning_rate = 0.01  # 学习率

# Skip-gram特有参数
num_negatives = 5     # 负采样数量
```

### 训练数据量对比

以示例文本为例：
- **CBOW**: 15个训练样本（每个位置一个样本）
- **Skip-gram**: 60个训练样本（每个(目标词,上下文词)对一个样本）

### 损失函数分析

**CBOW损失曲线特点**：
- 初始损失较高（约3.0）
- 快速收敛到接近0
- 曲线较为平滑

**Skip-gram损失曲线特点**：
- 初始损失很高（约7.6）
- 收敛较慢，存在波动
- 最终损失相对较高（约1.3）

---

## 应用场景

### CBOW适用场景

1. **完形填空任务**
   ```python
   # 例：根据"you say ___ and I"预测"goodbye"
   cloze_test(model, "you say goodbye and I say hello", 2, word_to_id, id_to_word)
   ```

2. **语言模型构建**
   - 文本生成
   - 语法检查
   - 自动补全

3. **高频词处理**
   - 对常见词汇效果更好
   - 适合处理语法功能词

### Skip-gram适用场景

1. **词汇扩展和发现**
   ```python
   # 例：根据"love"找到相关词汇
   predict_context_words(model, "love", word_to_id, id_to_word)
   ```

2. **同义词挖掘**
   - 发现语义相似词汇
   - 构建同义词词典

3. **低频词处理**
   - 对罕见词汇效果更好
   - 适合处理专业术语

4. **词汇类比推理**
   ```python
   # 概念上的：king - man + woman ≈ queen
   ```

---

## 实验结果

### 训练效果对比

**测试文本**: "you say goodbye and I say hello. what are you talking about? I love natural language processing."

#### CBOW结果
```
相似词查找:
- "you" 的相似词: are, language, goodbye, i, about?
- "say" 的相似词: goodbye, what, natural, love, talking
- "hello" 的相似词: and, love, are, goodbye, about?

完形填空准确性: 中等
收敛速度: 快
```

#### Skip-gram结果
```
相似词查找:
- "you" 的相似词: say, language, natural, love, hello
- "say" 的相似词: talking, hello, you, language, i
- "hello" 的相似词: goodbye, ., say, language, about?

上下文预测能力: 强
词汇发现能力: 强
```

### 性能分析

| 指标 | CBOW | Skip-gram |
|------|------|-----------|
| 训练速度 | 快 | 慢 |
| 内存消耗 | 低 | 高 |
| 收敛稳定性 | 高 | 中等 |
| 词汇发现能力 | 中等 | 强 |
| 语法理解能力 | 强 | 中等 |

---

## 代码使用指南

### 运行CBOW模型

```bash
cd nlp
python cbow_pytorch.py
```

**功能包括**：
- 模型训练和损失可视化
- 相似词查找
- 批量完形填空测试
- 交互式完形填空
- 词向量可视化

### 运行Skip-gram模型

```bash
cd nlp
python skipgram_pytorch.py
```

**功能包括**：
- 负采样训练
- 相似词查找
- 上下文词预测
- 词向量可视化

### 自定义使用

```python
# 导入模型
from cbow_pytorch import CBOWModel, train_cbow
from skipgram_pytorch import SkipGramModel, train_skipgram

# 准备数据
text = "your custom text here"
corpus, word_to_id, id_to_word = preprocess(text)

# 训练CBOW
cbow_model = CBOWModel(len(word_to_id), 100)
cbow_data = create_cbow_dataset(corpus)
train_cbow(cbow_model, cbow_data)

# 训练Skip-gram
skipgram_model = SkipGramModel(len(word_to_id), 100)
skipgram_data = create_skipgram_dataset(corpus)
train_skipgram(skipgram_model, skipgram_data, len(word_to_id))
```

---

## 优化建议

### 模型改进方向

1. **层次化Softmax**
   - 替代负采样，提高大词汇表训练效率
   - 基于词频构建二叉树结构

2. **子词信息**
   - 引入字符级或子词级信息
   - 处理未登录词和形态变化

3. **动态窗口**
   - 根据距离调整上下文权重
   - 更好捕捉长距离依赖

### 工程优化

1. **批处理训练**
   ```python
   # 当前实现是单样本训练，可以改为批处理
   batch_size = 32
   ```

2. **数据并行**
   ```python
   # 使用PyTorch的DataParallel进行多GPU训练
   model = nn.DataParallel(model)
   ```

3. **内存优化**
   - 使用gradient checkpointing
   - 动态词汇表

---

## 总结

### 核心要点

1. **CBOW与Skip-gram是互补的**：
   - CBOW：上下文→目标词，适合完形填空
   - Skip-gram：目标词→上下文，适合词汇发现

2. **训练策略不同**：
   - CBOW：使用交叉熵损失，直接优化
   - Skip-gram：使用负采样，避免昂贵的softmax

3. **应用场景各异**：
   - CBOW：语言建模、语法任务
   - Skip-gram：语义分析、词汇扩展

### 选择建议

- **小数据集**：选择CBOW，训练更稳定
- **大数据集**：选择Skip-gram，效果更好
- **完形填空**：使用CBOW
- **词汇发现**：使用Skip-gram
- **平衡考虑**：可以同时训练两个模型并集成

### 扩展方向

1. **FastText**: 加入子词信息
2. **GloVe**: 结合全局统计信息
3. **ELMo**: 上下文相关的词向量
4. **BERT**: 双向Transformer编码器

这两个模型为现代NLP奠定了重要基础，理解它们的原理和实现对深入学习词嵌入技术具有重要意义。 