import numpy as np
from sympy.physics.control.control_plots import plt

from dezero.core_simple import *
from dezero.utils import plot_dot_graph


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z



if __name__ == '__main__':

    #线性回归
    # Generate toy dataset
    # np.random.seed(0)
    # x = np.random.rand(10000, 1)
    # y = 5 + 2 * x + np.random.rand(10000, 1)
    # x, y = Variable(x), Variable(y)
    #
    # W = Variable(np.zeros((1, 1)))
    # b = Variable(np.zeros(1))
    #
    #
    # def predict(x):
    #     return linear_simple(x, W, b)
    #
    #

    np.random.seed(0)
    x = np.random.rand(1000, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(1000, 1)

    I, H, O =  1, 30 , 1
    # Xavier初始化
    W1 = Variable(np.random.randn(I,H) / np.sqrt(I))
    b1 = Variable(np.zeros(H))
    W2 = Variable(np.random.randn(H,O) / np.sqrt(H))
    b2 = Variable(np.zeros(O))

    def predict(x) :
        y = linear_simple(x, W1, b1)
        y = sigmoid_simple(y)  # 使用tanh替代sigmoid
        y = linear_simple(y, W2, b2)
        return y


    lr = 0.01  # 降低学习率
    iters = 50000

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()

        loss.backward()

        # Update .data attribute (No need grads when updating params)
        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data
        if i % 1000 == 0:
            print(loss)

    # Plot
    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    t = np.arange(0, 1, .01)[:, np.newaxis]
    y_pred = predict(t)
    plt.plot(t, y_pred.data, color='r')
    plt.show()