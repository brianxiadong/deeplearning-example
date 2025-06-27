import numpy as np
from matplotlib import pyplot as plt
import datasets
from sklearn.utils.extmath import randomized_svd


def preprocess(text):
    word = text.lower().replace('.', ' .').split(' ')

    #带去重，增加id
    word_to_id = {}
    id_to_word = {}
    for w in word:
        if w not in word_to_id:
            tmp = len(word_to_id)
            word_to_id[w] = tmp
            id_to_word[tmp] = w

    #获得ID列表
    corpus = [word_to_id[w] for w in word]

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

# 余弦相似度
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

# 使用余弦相似度，查询词典中最相似的字
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    # ❷ 计算余弦相似度
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    # ❸ 基于余弦相似度，按降序输出值
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M


# 处理PTB文本数据
def process_ptb_data():
    dataset = datasets.load_dataset('common/ptb.py', 'penn_treebank')
    
    # 获取所有句子
    sentences = []
    for split in ['train', 'validation', 'test']:
        for item in dataset[split]:
            sentences.append(item['sentence'])
    
    return sentences


# 处理PTB文本数据
def process_ptb_data():
    dataset = datasets.load_dataset('ptb.py', 'penn_treebank')

    # 获取所有句子
    sentences = []
    for split in ['train', 'validation', 'test']:
        for item in dataset[split]:
            sentences.append(item['sentence'])

    # 合并成一个字符串
    return ' '.join(sentences)

if __name__ == '__main__':

     window_size = 2
     wordvec_size = 100

     text = process_ptb_data()
     # text = 'you say goodbye and I say hello.what are u talking abount?'
     corpus, word_to_id, id_to_word = preprocess(text)
     print(corpus)
     print(id_to_word)
     print(word_to_id)

     co_matrix = create_co_matrix(corpus, len(word_to_id), window_size)
     print(co_matrix)


     c0 = co_matrix[word_to_id['you']]
     c1 = co_matrix[word_to_id['i']]

     print(cos_similarity(c0, c1))

     most_similar('you', word_to_id, id_to_word, co_matrix)

     W = ppmi(co_matrix, verbose=True)
     print(W)
     U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                              random_state=None)
     print(U)
     print(S)
     print(V)

     for word, id in word_to_id.items():
         plt.annotate(word, (U[id, 0], U[id, 1]))

     plt.scatter(U[:,0], U[:,1], alpha=0.5)
     plt.show()