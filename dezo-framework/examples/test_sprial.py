"""
测试两层神经网络
演示如何使用TwoLayerNet模型
"""
import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import datasets,MLP,optimizers


def test_sprial():
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    x,t  = datasets.get_sprial(train=True)
    model = MLP((hidden_size,3))
    optimizer = optimizers.SGD(lr).setup(model)


if __name__ == '__main__':
    test_sprial()