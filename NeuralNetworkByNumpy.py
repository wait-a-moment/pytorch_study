# *-* coding:utf8 *-*
import numpy as np

"""
使用numpy实现两层神经网络
h=W_1*X+b1
a=max(0,h)
y_hat=W_2*a+b_2
"""
size, x_in, h, out = 64, 1000, 200, 10  # size表示一批数据的个数，x_in表示输入数据维度，h表示隐藏层维度（一个隐藏层有多少个神经元），out表示输出维度
# 创建输入和输出数据 randn返回的数据服从标准正态分布
x = np.random.randn(size, x_in)
y = np.random.randn(size, out)

# 初始化权重
w1 = np.random.randn(x_in, h)
w2 = np.random.randn(h, out)

# 学习率
lr = 1e-6

for i in range(500):
    # 前向传播
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算损失
    loss = np.square(y_pred - y).sum()
    print(i, loss)

    # 反向传播
    # loss = (y_pred - y) ** 2
    grad_y_pred = 2.0 * (y_pred - y)
    #
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
