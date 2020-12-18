import torch
import torch.nn

"""
使用PyTorch中nn这个库来构建网络。 用PyTorch autograd来构建计算图和计算gradients， 然后PyTorch会帮我们自动计算gradient。

"""

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out))

loss_fn = torch.nn.MSELoss(reduction='sum')  # 默认为mean

lr = 1e-6

for i in range(3500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(i, loss)

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
