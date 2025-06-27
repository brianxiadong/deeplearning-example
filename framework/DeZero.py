import numpy as np

from dezero.core_simple import Variable, sin, my_sin, rosenbrock, tanh
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
   x = Variable(np.array(1.0))
   y = tanh(x)
   x.name = 'x'
   y.name = 'y'
   y.backward(create_graph=True)

   iters = 10
   for i in range(iters):
       gx = x.grad
       x.cleargrad()
       gx.backward(create_graph=True)

   gx = x.grad
   gx.name  = 'gx' + str(iters + 1)
   plot_dot_graph(gx, verbose=False, to_file='tanh.png')