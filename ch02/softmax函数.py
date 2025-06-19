import numpy as np
#softmax函数的输出是0.0到1.0之间的实数。并且，softmax
#函数的输出值的总和是1。输出总和为1是softmax函数的一个重要性质。正
#因为有了这个性质，我们才可以把softmax函数的输出解释为“概率”

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def softmax2(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

if __name__ == '__main__':

    test = softmax([0.1, 0.2, 0.3, 0.4, 0.5])
    print(test)
    test = softmax2([0.1, 0.2, 0.3, 0.4, 0.5])
    print(test)