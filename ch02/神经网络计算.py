import numpy as np

from ch01.sigmoid函数 import sigmoid

if __name__ == '__main__':
    def init_network():
        """初始化神经网络参数"""
        network = {}
        # 第一层权重
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        # 第一层偏置
        network['b1'] = np.array([0.1, 0.2, 0.3])
        # 第二层权重
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        # 第二层偏置
        network['b2'] = np.array([0.1, 0.2])
        # 第三层权重
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        # 第三层偏置
        network['b3'] = np.array([0.1, 0.2])
        return network


    def identity_function(a3):
        """恒等函数作为输出层激活函数"""
        return a3


    def forward(network, x):
        """前向传播计算"""
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        # 第一层计算
        a1 = np.dot(x, W1) + b1
        # 应用Sigmoid激活函数
        z1 = sigmoid(a1)
        # 第二层计算
        a2 = np.dot(z1, W2) + b2
        # 应用Sigmoid激活函数
        z2 = sigmoid(a2)
        # 输出层计算
        a3 = np.dot(z2, W3) + b3
        # 应用恒等函数 回归问题用恒等函数，分类问题用softmax函数
        y = identity_function(a3)
        return y


    # 初始化网络
    network = init_network()
    # 输入数据
    x = np.array([1.0, 0.5])
    # 前向传播计算
    y = forward(network, x)
    # 打印输出结果
    print(y)  # [ 0.31682708 0.69627909]