import pickle
import sys, os

import numpy as np
from sklearn.metrics import accuracy_score

from ch01.sigmoid函数 import sigmoid
from ch02.softmax函数 import softmax2, softmax
from mnist import load_mnist
from PIL import Image

if __name__ == '__main__':

    sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
    # 第一次调用会花费几分钟 ……

    def img_show(img):
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.show()

    def get_data():
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                          normalize=True,
                                                          one_hot_label=False)
        return x_test,t_test

    def init_network():
        with open("sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)

        return network

    def predict(network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,W3) + b3
        y = softmax(a3)

        return y

    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
          accuracy_cnt += 1

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))