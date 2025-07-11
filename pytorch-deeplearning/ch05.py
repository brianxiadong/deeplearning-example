import matplotlib
import random
import torch
from d2l import torch as d2l
from torch import nn


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机打乱索引

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

if __name__ == '__main__':

   # torch.normal的作用是生成一个正态分布的随机张量
   w = torch.normal(0,0.01,(2,1), requires_grad=True)
   # 创建一个常数张量
   b = torch.zeros(1, requires_grad=True)

   # 定义模型，全连接层
   def linear(x, w, b):
       return torch.matmul(x,w) + b

   # 定义损失函数
   def loss(y_hat, y):
        # 均方误差 y.view(y_hat.size())作用是将y_hat和y的形状统一，方便计算损失
        return (y_hat - y.view(y_hat.size())) ** 2 / 2


    # 定义优化器
   def sgd(params, lr, batch_size):
       """小批量随机梯度下降"""
       with torch.no_grad():
           for param in params:
               param -= lr * param.grad / batch_size
               # 清零梯度
               param.grad.zero_()

   lr = 0.05
   num_epochs = 20
   batch_size = 100

   # 生成数据
   true_w = torch.tensor([2, -3.4])
   true_b = 4.2
   features, labels = d2l.synthetic_data(true_w, true_b, 1000)

   for epoch in range(num_epochs):
       for X, y in data_iter(batch_size, features, labels):
           l = loss(linear(X, w , b), y)
           l.sum().backward()
           sgd([w, b], lr, batch_size)

       with torch.no_grad():
           train_l = loss(linear(features, w, b), labels)
           print(f'epoch {epoch + 1}, loss {float(train_l.mean()):.6f}')

   print(f'w的估计误差 : {true_w - w.reshape(true_w.shape)}')
   print(f'b的估计误差 : {true_b - b}')


   net = nn.Sequential(nn.Linear(2,1))
   net[0].weight.data.normal_(0,0.01)
   net[0].bias.data.fill_(0)

   loss = nn.MSELoss()
   trainer = torch.optim.SGD(net.parameters(), lr=0.03)

   for epoch in range(num_epochs):
       for X, y in data_iter(batch_size, features, labels):
           l = loss(net(X), y)
           trainer.zero_grad()
           l.backward()
           trainer.step()

       l = loss(net(features), labels)
       print(f'epoch {epoch + 1}, loss {l:f}')