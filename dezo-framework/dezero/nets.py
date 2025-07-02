import numpy as np

from .core import Variable
from .models import Model
from .layers import Linear
from .functions import sigmoid_simple

class TwoLayerNet(Model):
    def __init__(self, hidden_size , out_size):
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.l2 = Linear(out_size)

    def forward(self, x):
        return self.l2(sigmoid_simple(self.l1(x)))

def main():
    """演示两层神经网络的使用"""
    x = Variable(np.random.randn(5, 10), name='x')
    model = TwoLayerNet(100, 10)
    
    # 前向传播
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 生成计算图
    model.plot(x, to_file="../images/two-layer.png")
    print("计算图已保存到 images/two-layer.png")


if __name__ == '__main__':
    main()