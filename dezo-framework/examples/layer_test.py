import numpy as np
import sys
import os

# 添加父目录到Python路径，以便导入dezero包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import Variable, Linear, mean_squared_error, sigmoid_simple, Layer

if __name__ == '__main__':
    model = Layer()
    model.l1 = Linear(5)
    model.l2 = Linear(3)

    def predict(x):
        y = model.l1(x)
        y = sigmoid_simple(y)
        y = model.l2(y)
        return y

    for p in model.params():
        print(p)

    model.cleargrads()