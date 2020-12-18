import torch

dtype = torch.float
device = torch.device('cpu')

size, x_in, h, out = 64, 1000, 200, 10  # size表示一批数据的个数，x_in表示输入数据维度，h表示隐藏层维度（一个隐藏层有多少个神经元），out表示输出维度

x = torch.randn(size, x_in, device=device, dtype=dtype)
y = torch.randn(size, out, device=device, dtype=dtype)

w1 = torch.randn(x_in, h, device=device, dtype=dtype)
w2 = torch.randn(h, out, device=device, dtype=dtype)

lr = 1e-6

for i in range(500):
    # 前向传播
    h = x.mm(w1)
    h_relu = torch.clamp(h, min=0)  # 将小于min的数值设置为min,将大于max的数值设置为max
    y_pred=h_relu.mm(w2)

    # 计算loss
    loss=(y_pred-y).pow(2).sum().item()
    print(i,loss)

    # 反向传播
    grad_y_pred=2.0*(y_pred-y)
    grad_w2=h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 梯度更新
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2