from .layers import Layer
from .utils import *

class Model(Layer):
    def plot(self, *inputs, to_file='../images/model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)