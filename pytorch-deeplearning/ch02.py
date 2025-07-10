import torch
import os
import pandas as pd

if __name__ == '__main__':
    # 初始化一个0-11的张量
    x = torch.arange(12)
    print(x)

    # 张量的形状
    print(x.shape)

    # 张量中元素的总数
    print(x.numel())

    # 一维张量改为3行四列的张量
    x = x.reshape(3, 4)
    print(x)

    # 创建全0
    y = torch.zeros((2, 3, 4))
    print(y)

    # 创建全1
    y = torch.ones((2, 3, 4))
    print(y)

    # 通过提供包含数值的Python列表（或嵌套列表）来为所需张量中的每个元素赋予确定值。
    y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]) # 二维tensor
    z = torch.tensor([[[2,1,4,3],[1,2,3,4],[4,3,2,1]]]) # 三维tensor
    print(y)
    print(z)

    # 张量运算操作
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)  # **运算符是求幂运算

    # e的x次方
    x = torch.tensor([1.0,2,4,8])
    x = torch.exp(x) # e的x次方
    print(x)

    # 张量合并操作
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    m = torch.cat((x, y), dim=0)  # 按行合并起来
    n = torch.cat((x, y), dim=1)  # 按列合并起来
    print(m)
    print(n)

    # 通过逻辑运算符构建二元张量
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(x)
    print(y)
    print(x == y)  # 对应元素相等为 True，否则为 False

    # 张量累加运算
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    print(x.sum())

    # 张量广播运算
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a)
    print(b)
    print(a + b)  # a会复制出一个3*2的矩阵，b复制出一个3*2的矩阵，然后再相加，会得到一个3*2矩阵

    # 张量访问运算 可以用`[-1]`选择最后一个元素，可以用`[1:3]`选择第二个和第三个元素。
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    print(x[-1])
    print(x[1:3])

    # 除读取外，还可以通过指定索引来将元素写入矩阵
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    x[1, 2] = 9
    print(x)

    # 批量初始化数据、设置默认值、创建均匀分布的数据集
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    x[0:2, :] = 12
    print(x)

    # 运行一些操作可能会导致为新结果分配内容
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    before = id(y)
    y = x + y
    print(id(y) == before)  # 运行操作后，赋值后的y的id和原来的id不一样

    # 如果在后续计算中没有重复使用X，即内存不会过多复制，也可以使用`X[:] = X + Y` 或 `X += Y` 来减少操作的内存开销。
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    z = torch.zeros_like(y)  # z 的数据类型、尺寸和y一样，里面的元素全为0
    print("id(z):", id(z))
    z[:] = x + y
    print("id(z):", id(z))

    # 张量转 NumPy。
    #
    # **案例：** 与传统机器学习库(如scikit-learn)交互，或使用matplotlib绘图。
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    A = x.numpy()
    B = torch.tensor(A)
    print(type(A), type(B))

    # 将大小为1的张量转为 Python 标量
    a = torch.tensor([3.5])
    print(a)
    print(a.item()) # 转标量
    print(float(a))
    print(int(a))

    os.makedirs(os.path.join('.', '01_Data'), exist_ok=True)  # 相对路径，创建文件夹
    data_file = os.path.join('.', '01_Data', '01_house_tiny.csv')
    print(data_file)
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')

    data_file = os.path.join('.', '01_Data', '01_house_tiny.csv')
    data = pd.read_csv(data_file, na_values='NA')
    print(data)

    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean())  # 对 NaN 值用均值插值
    print(inputs)