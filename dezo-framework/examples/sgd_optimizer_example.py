import numpy as np
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dezero import MLP, optimizers, mean_squared_error


if __name__ == '__main__':

    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = MLP((100, 1))
    optimizer = optimizers.Momentum(lr).setup(model)

    for i in range(max_iter):
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        if i % 1000 == 0:
            print(loss)