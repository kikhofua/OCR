import torch

x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.FloatTensor(5,3)
torch.add(x, y, out=result)
print(result)
y.add_(x)  # <-- mutates a tensor in place
print(y)
print(x[:, 1])
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

