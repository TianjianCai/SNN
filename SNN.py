import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


line = np.linspace(0,10,1000)
groundtruth = -15*line + 49
noise = np.random.normal(0,21,1000)
noised_output = groundtruth+noise


lr = 1e-3


class layer(nn.Module):
    def __init__(self):
        super(layer, self).__init__()
        self.k = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.k.expand(x.size()) * x + self.b.expand(x.size())


module = layer()

i = 0
old_loss = 0
while True:
    l = torch.tensor(line,dtype=torch.float32)
    output = module(l)
    loss = torch.mean((output - torch.tensor(noised_output, dtype=torch.float32)) ** 2)
    loss.backward()
    print(repr(i) + " loss: "+repr(loss.item()))
    if abs(loss.item() - old_loss) < 1e-5:
        break
    else:
        old_loss = loss.item()
    with torch.no_grad():
        module.k -= lr * module.k.grad
        module.b -= lr * module.b.grad
        module.k.grad.zero_()
        module.b.grad.zero_()
    i = i + 1

output = module(l)
o = output.data.tolist()
print(o)

print(module.k.item(),module.b.item())

plt.scatter(line,noised_output,s=0.5)
plt.plot(line,o,color='r')

plt.show()



