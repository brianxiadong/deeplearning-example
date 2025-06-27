import contextlib
import math
import unittest
import weakref

import numpy as np
from onnxslim.third_party.onnx_graphsurgeon import Variable

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

def no_grad():
    return using_config("enable_backprop", False)

class Config:
    enable_backprop = True

class Variable:

    __array_priority__ = 200
    def __init__(self, data ,name =  None):
        if data is not None :
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __mul__(self, other):
        return mul(self, other)
    def __add__(self, other):
        return add(self, other)
    def __len__(self):
        return len(self.data)
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def set_creator(self, func):
        self.creator = func
        self.generation = self.creator.generation + 1

    def cleargrad(self):
        self.grad = None

    def __repr__(self):
        if self is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(%s)' % p

    def backward(self, retain_grad=False , create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        # f = self.creator
        # if f is not None:
        #     x = f.input
        #     x.grad = f.backward(self.grad)
        #     # 递归调用
        #     x.backward()
        # 修改为循环
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs) :
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

                if not retain_grad:
                    for output in f.outputs:
                        output().grad = None

    def clear_grads(self):
        self.grad = None


class Function:
    def __call__(self, *inputs):

        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

class Mul( Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0

def mul(x0,x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# class SquareTest(unittest.TestCase) :
#     def test_forward(self):
#         x = Variable(np.array(2.0))
#         y = square(x)
#         expected = np.array(4.0)
#         self.assertEqual(y.data, expected)
#
#     def test_backward(self):
#         x = Variable(np.array(3.0))
#         y = square(x)
#         y.backward()
#         expected = np.array(6.0)
#         self.assertEqual(x.grad, expected)

class Add(Function):
    def forward(self, x0, x1):
       return x0 + x1

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)



class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x , threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
def tanh(x):
    return Tanh()(x)

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y
def setup_variable():
    Variable.__pow__ = pow
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__rsub__ = rsub
    Variable.__sub__ = sub
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul