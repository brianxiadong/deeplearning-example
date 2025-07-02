import weakref
import numpy as np
from .core import Variable
from .tensor_ops import matmul


class Parameter(Variable):
    """参数类，继承自Variable但用于表示模型参数"""
    pass


class Layer:
    """神经网络层的基类"""
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
       for name in self._params:
           obj = self.__dict__[name]

           if isinstance(obj, Layer):
               yield from obj.params()
           else:
                yield obj


    def cleargrads(self):
        """清除所有参数的梯度"""
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    """线性层（全连接层）"""
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        """Xavier初始化权重"""
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = linear_simple(x, self.W, self.b)
        return y


def linear_simple(x, W, b=None):
    """简单的线性变换函数"""
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None  # 释放中间结果的内存
    return y 