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

def train_cbow(model, train_data, epochs=100, learning_rate=0.01):
    """训练CBOW模型"""
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
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
        for context, target in train_data:
            # 将上下文词汇转换为张量并增加batch维度
            context_tensor = torch.tensor(context, dtype=torch.long).unsqueeze(0)
            # 将目标词转换为张量
            target_tensor = torch.tensor([target], dtype=torch.long)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = model(context_tensor)
            # 计算损失
            loss = criterion(output, target_tensor)
            
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            # 累加损失值
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_data)
        # 记录损失
        losses.append(avg_loss)
        
        # 每10个epoch打印一次损失
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    # 返回损失历史
    return losses

def get_word_vectors(model, word_to_id):
    """获取词向量"""
    # 设置模型为评估模式
    model.eval()
    # 初始化词向量字典
    word_vectors = {}
    
    # 关闭梯度计算以节省内存
    with torch.no_grad():
        # 遍历词汇表中的每个词汇
        for word, idx in word_to_id.items():
            # 获取词汇的嵌入向量并转换为numpy数组
            vector = model.embedding(torch.tensor([idx])).numpy().flatten()
            # 存储词向量
            word_vectors[word] = vector
    
    # 返回词向量字典
    return word_vectors

def find_similar_words(query, word_vectors, word_to_id, id_to_word, top=5):
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
    print(f'\nWords similar to "{query}":')
    for word, similarity in similarities[:top]:
        print(f'{word}: {similarity:.4f}')

def visualize_embeddings(word_vectors, word_to_id, max_words=50):
    """可视化词向量（使用PCA降维到2D）"""
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
    plt.title('Word Embeddings Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # 显示图表
    plt.show()

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
    
    # 检查是否有足够的上下文
    if len(context_words) == 0:
        print("没有找到有效的上下文词汇")
        return None
    
    print(f"上下文词汇: {[id_to_word[idx] for idx in context_words]}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 关闭梯度计算
    with torch.no_grad():
        # 转换为张量
        context_tensor = torch.tensor(context_words, dtype=torch.long).unsqueeze(0)
        
        # 进行预测
        output = model(context_tensor)
        # 计算softmax概率
        probabilities = F.softmax(output, dim=1)
        
        # 获取top-k预测结果
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        print(f"\n完形填空预测结果 (Top {top_k}):")
        print("预测词汇\t\t概率")
        print("-" * 30)
        
        # 记录预测结果
        predictions = []
        for i in range(top_k):
            word_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            predicted_word = id_to_word[word_idx]
            predictions.append((predicted_word, prob))
            print(f"{predicted_word}\t\t{prob:.4f}")
        
        # 检查真实答案是否在预测中
        true_word_in_predictions = any(pred[0] == true_word for pred in predictions)
        if true_word_in_predictions:
            true_word_rank = next(i for i, (word, _) in enumerate(predictions) if word == true_word) + 1
            print(f"\n✓ 真实答案 '{true_word}' 排在第 {true_word_rank} 位")
        else:
            print(f"\n✗ 真实答案 '{true_word}' 不在Top {top_k}预测中")
        
        return predictions

def interactive_cloze_test(model, word_to_id, id_to_word, window_size=2):
    """交互式完形填空测试"""
    print("\n=== 交互式完形填空测试 ===")
    print("输入句子和要遮掩的词汇位置")
    print("例如: 'you say goodbye and I say hello' 位置2 (遮掩'goodbye')")
    print("输入 'quit' 退出")
    
    while True:
        # 获取用户输入
        user_input = input("\n请输入句子: ").strip()
        if user_input.lower() == 'quit':
            break
        
        try:
            # 获取遮掩位置
            position = int(input("请输入要遮掩的词汇位置 (从0开始): "))
            
            # 执行完形填空测试
            cloze_test(model, user_input, position, word_to_id, id_to_word, window_size)
            
        except ValueError:
            print("请输入有效的数字位置")
        except Exception as e:
            print(f"发生错误: {e}")

def batch_cloze_test(model, word_to_id, id_to_word, window_size=2):
    """批量完形填空测试"""
    # 定义测试句子和遮掩位置
    test_cases = [
        ("you say goodbye and I say hello", 2),  # 遮掩 "goodbye"
        ("you say goodbye and I say hello", 5),  # 遮掩 "say"
        ("what are you talking about", 3),       # 遮掩 "talking"
        ("I love natural language processing", 1), # 遮掩 "love"
    ]
    
    print("\n=== 批量完形填空测试 ===")
    
    for i, (sentence, mask_pos) in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {sentence}")
        print(f"遮掩位置: {mask_pos}")
        print("-" * 50)
        
        # 执行完形填空
        cloze_test(model, sentence, mask_pos, word_to_id, id_to_word, window_size)

def main():
    # 设置窗口大小超参数
    window_size = 2
    # 设置嵌入维度
    embedding_dim = 100
    # 设置训练轮数
    epochs = 100
    # 设置学习率
    learning_rate = 0.01
    
    # 准备训练文本数据
    text = 'you say goodbye and I say hello. what are you talking about? I love natural language processing.'
    # 预处理文本，获取语料库和词汇映射
    corpus, word_to_id, id_to_word = preprocess(text)
    # 获取词汇表大小
    vocab_size = len(word_to_id)
    
    # 打印数据集信息
    print(f'Vocabulary size: {vocab_size}')
    print(f'Corpus length: {len(corpus)}')
    
    # 创建CBOW训练数据集
    train_data = create_cbow_dataset(corpus, window_size)
    print(f'Training samples: {len(train_data)}')
    
    # 创建CBOW模型实例
    model = CBOWModel(vocab_size, embedding_dim)
    
    # 开始训练模型
    print('Training CBOW model...')
    losses = train_cbow(model, train_data, epochs, learning_rate)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # 从训练好的模型中提取词向量
    word_vectors = get_word_vectors(model, word_to_id)
    
    # 测试相似词查找功能
    test_words = ['you', 'say', 'hello']
    for word in test_words:
        # 检查词汇是否在词汇表中
        if word in word_to_id:
            # 查找相似词汇
            find_similar_words(word, word_vectors, word_to_id, id_to_word)
    
    # 可视化词嵌入
    visualize_embeddings(word_vectors, word_to_id)
    
    # 执行批量完形填空测试
    batch_cloze_test(model, word_to_id, id_to_word, window_size)
    
    # 启动交互式完形填空测试
    interactive_cloze_test(model, word_to_id, id_to_word, window_size)
    
    # 返回训练结果
    return model, word_vectors, word_to_id, id_to_word

# 主程序入口
if __name__ == '__main__':
    # 执行主函数
    model, word_vectors, word_to_id, id_to_word = main() 