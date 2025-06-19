import numpy as np

def step_func(x):
    return np.array(x > 0, dtype=np.int)

if __name__ == '__main__':
    x = np.array([0,1])
    w = np.array([0.5,0.5]) # w1 w2 为权重
    b = -0.7        # θ为偏置
    print(np.sum(w * x))
    print(np.sum(w * x) + b)

