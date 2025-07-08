# 导入PyTorch核心模块
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入优化器模块
import torch.optim as optim
# 导入函数式接口模块
import torch.nn.functional as F
# 导入数值计算库
import numpy as np
# 导入默认字典
from collections import defaultdict
# 导入绘图库
import matplotlib.pyplot as plt
# 从工具模块导入预处理函数和余弦相似度函数
from common.util import preprocess, cos_similarity

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
    
    def get_target_embedding(self, word_idx):
        # 获取目标词的嵌入向量
        return self.target_embedding(word_idx)

def create_skipgram_dataset(corpus, window_size=2):
    """创建Skip-gram训练数据集"""
    # 初始化数据列表
    data = []
    # 遍历语料库，跳过边界位置
    for i in range(window_size, len(corpus) - window_size):
        # 当前位置的词汇作为目标词
        target = corpus[i]
        
        # 收集目标词周围的上下文词汇
        for j in range(i - window_size, i + window_size + 1):
            # 跳过目标词本身
            if j != i:
                # 每个(目标词, 上下文词)对作为一个训练样本
                context = corpus[j]
                data.append((target, context))
    
    # 返回训练数据集
    return data

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
    
    # 设置模型为训练模式
    model.train()
    # 初始化损失记录列表
    losses = []
    
    # 开始训练循环
    for epoch in range(epochs):
        # 初始化当前epoch的总损失
        total_loss = 0
        
        # 遍历训练数据
        for target, context in train_data:
            # 将目标词和上下文词转换为张量
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)
            
            # 正样本标签（1表示是真实的上下文词）
            positive_label = torch.tensor([1.0], dtype=torch.float)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 计算正样本得分
            positive_score = model(target_tensor, context_tensor)
            # 计算正样本损失
            positive_loss = criterion(positive_score, positive_label)
            
            # 负采样
            negative_samples = negative_sampling(vocab_size, num_negatives)
            # 负样本标签（0表示不是真实的上下文词）
            negative_labels = torch.zeros(num_negatives, dtype=torch.float)
            
            # 计算负样本得分
            target_repeated = target_tensor.repeat(num_negatives)
            negative_scores = model(target_repeated, negative_samples)
            # 计算负样本损失
            negative_loss = criterion(negative_scores, negative_labels)
            
            # 总损失
            total_sample_loss = positive_loss + negative_loss
            
            # 反向传播
            total_sample_loss.backward()
            # 更新参数
            optimizer.step()
            
            # 累加损失值
            total_loss += total_sample_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_data)
        # 记录损失
        losses.append(avg_loss)
        
        # 每10个epoch打印一次损失
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    # 返回损失历史
    return losses

def get_word_vectors_skipgram(model, word_to_id):
    """获取Skip-gram模型的词向量"""
    # 设置模型为评估模式
    model.eval()
    # 初始化词向量字典
    word_vectors = {}
    
    # 关闭梯度计算以节省内存
    with torch.no_grad():
        # 遍历词汇表中的每个词汇
        for word, idx in word_to_id.items():
            # 获取目标词的嵌入向量并转换为numpy数组
            vector = model.target_embedding(torch.tensor([idx])).numpy().flatten()
            # 存储词向量
            word_vectors[word] = vector
    
    # 返回词向量字典
    return word_vectors

def find_similar_words_skipgram(query, word_vectors, word_to_id, id_to_word, top=5):
    """查找最相似的词汇"""
    # 检查查询词是否在词汇表中
    if query not in word_to_id:
        print(f'{query} not found in vocabulary')
        return
    
    # 获取查询词的向量
    query_vector = word_vectors[query]
    # 初始化相似度列表
    similarities = []
    
    # 计算与所有其他词汇的相似度
    for word, vector in word_vectors.items():
        # 跳过查询词本身
        if word != query:
            # 计算余弦相似度
            similarity = cos_similarity(query_vector, vector)
            # 添加到相似度列表
            similarities.append((word, similarity))
    
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 打印最相似的词汇
    print(f'\nSkip-gram: Words similar to "{query}":')
    for word, similarity in similarities[:top]:
        print(f'{word}: {similarity:.4f}')

def predict_context_words(model, target_word, word_to_id, id_to_word, top_k=5):
    """预测给定目标词的上下文词汇"""
    # 检查目标词是否在词汇表中
    if target_word not in word_to_id:
        print(f'{target_word} not found in vocabulary')
        return None
    
    # 获取目标词索引
    target_idx = word_to_id[target_word]
    target_tensor = torch.tensor([target_idx], dtype=torch.long)
    
    # 设置模型为评估模式
    model.eval()
    
    # 关闭梯度计算
    with torch.no_grad():
        # 计算目标词与所有词汇的相似度得分
        scores = []
        vocab_size = len(word_to_id)
        
        for word_idx in range(vocab_size):
            context_tensor = torch.tensor([word_idx], dtype=torch.long)
            score = model(target_tensor, context_tensor)
            scores.append((word_idx, score.item()))
        
        # 按得分降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nSkip-gram预测: '{target_word}' 的上下文词汇 (Top {top_k}):")
        print("预测词汇\t\t得分")
        print("-" * 30)
        
        # 记录预测结果
        predictions = []
        for i in range(min(top_k, len(scores))):
            word_idx, score = scores[i]
            predicted_word = id_to_word[word_idx]
            # 跳过目标词本身
            if predicted_word != target_word:
                predictions.append((predicted_word, score))
                print(f"{predicted_word}\t\t{score:.4f}")
            else:
                top_k += 1  # 如果遇到目标词，增加显示数量
        
        return predictions[:top_k-1]  # 排除目标词后返回

def visualize_embeddings_skipgram(word_vectors, word_to_id, max_words=50):
    """可视化Skip-gram词向量（使用PCA降维到2D）"""
    # 导入PCA降维算法
    from sklearn.decomposition import PCA
    
    # 取前max_words个词汇进行可视化
    words = list(word_vectors.keys())[:max_words]
    # 获取对应的向量
    vectors = [word_vectors[word] for word in words]
    
    # 创建PCA对象，降维到2维
    pca = PCA(n_components=2)
    # 执行降维
    vectors_2d = pca.fit_transform(vectors)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    # 绘制散点图
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
    
    # 为每个点添加词汇标签
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
    
    # 设置图表标题和轴标签
    plt.title('Skip-gram Word Embeddings Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # 显示图表
    plt.show()

def compare_models(target_word, cbow_vectors, skipgram_vectors, word_to_id):
    """比较CBOW和Skip-gram模型的词向量相似度"""
    if target_word not in word_to_id:
        print(f'{target_word} not found in vocabulary')
        return
    
    print(f"\n=== 模型比较: '{target_word}' 的相似词 ===")
    
    # CBOW相似词
    print("\nCBOW模型:")
    cbow_similarities = []
    target_vector_cbow = cbow_vectors[target_word]
    
    for word, vector in cbow_vectors.items():
        if word != target_word:
            similarity = cos_similarity(target_vector_cbow, vector)
            cbow_similarities.append((word, similarity))
    
    cbow_similarities.sort(key=lambda x: x[1], reverse=True)
    for word, sim in cbow_similarities[:5]:
        print(f'{word}: {sim:.4f}')
    
    # Skip-gram相似词
    print("\nSkip-gram模型:")
    skipgram_similarities = []
    target_vector_skipgram = skipgram_vectors[target_word]
    
    for word, vector in skipgram_vectors.items():
        if word != target_word:
            similarity = cos_similarity(target_vector_skipgram, vector)
            skipgram_similarities.append((word, similarity))
    
    skipgram_similarities.sort(key=lambda x: x[1], reverse=True)
    for word, sim in skipgram_similarities[:5]:
        print(f'{word}: {sim:.4f}')

def main():
    # 设置窗口大小超参数
    window_size = 2
    # 设置嵌入维度
    embedding_dim = 100
    # 设置训练轮数
    epochs = 100
    # 设置学习率
    learning_rate = 0.01
    # 设置负采样数量
    num_negatives = 5
    
    # 准备训练文本数据
    text = 'you say goodbye and I say hello. what are you talking about? I love natural language processing.'
    # 预处理文本，获取语料库和词汇映射
    corpus, word_to_id, id_to_word = preprocess(text)
    # 获取词汇表大小
    vocab_size = len(word_to_id)
    
    # 打印数据集信息
    print(f'Vocabulary size: {vocab_size}')
    print(f'Corpus length: {len(corpus)}')
    
    # 创建Skip-gram训练数据集
    train_data = create_skipgram_dataset(corpus, window_size)
    print(f'Training samples: {len(train_data)}')
    print(f'Sample training data: {train_data[:5]}')
    
    # 创建Skip-gram模型实例
    model = SkipGramModel(vocab_size, embedding_dim)
    
    # 开始训练模型
    print('\nTraining Skip-gram model...')
    losses = train_skipgram(model, train_data, vocab_size, epochs, learning_rate, num_negatives)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Skip-gram Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # 从训练好的模型中提取词向量
    word_vectors = get_word_vectors_skipgram(model, word_to_id)
    
    # 测试相似词查找功能
    test_words = ['you', 'say', 'hello']
    for word in test_words:
        # 检查词汇是否在词汇表中
        if word in word_to_id:
            # 查找相似词汇
            find_similar_words_skipgram(word, word_vectors, word_to_id, id_to_word)
    
    # 测试上下文词预测
    print("\n=== 上下文词预测测试 ===")
    for word in ['say', 'love', 'you']:
        if word in word_to_id:
            predict_context_words(model, word, word_to_id, id_to_word)
    
    # 可视化词嵌入
    visualize_embeddings_skipgram(word_vectors, word_to_id)
    
    # 返回训练结果
    return model, word_vectors, word_to_id, id_to_word

# 主程序入口
if __name__ == '__main__':
    # 执行主函数
    model, word_vectors, word_to_id, id_to_word = main() 