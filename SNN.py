import numpy as np
import torch
import matplotlib.pyplot as plt


line = np.linspace(0,10,100)
groundtruth = 15*line + 21
noise = np.random.normal(0,20,100)
noised_output = groundtruth+noise
'''
f1 = plt.subplot(211)
f1.scatter(line,groundtruth)
f2 = plt.subplot(212)
f2.scatter(line,noised_output)
plt.show()
'''


k = torch.randn(1,dtype=torch.float32,requires_grad=True)
b = torch.randn(1,dtype=torch.float32,requires_grad=True)
l = torch.tensor(line,dtype=torch.float32)

lr = 1e-2

i = 0
old_loss = 0
while True:
    output = k.expand(l.size()) * l + b.expand(l.size())
    loss = torch.mean((output - torch.tensor(noised_output, dtype=torch.float32)) ** 2)
    loss.backward()
    print(repr(i) + " loss: "+repr(loss.item()))
    if abs(loss.item() - old_loss) < 1e-6:
        break
    else:
        old_loss = loss.item()
    with torch.no_grad():
        k -= lr * k.grad
        b -= lr * b.grad
        k.grad.zero_()
        b.grad.zero_()
    i = i + 1

output = k.expand(l.size()) * l + b.expand(l.size())
o = output.data.tolist()
print(o)

print(k.item(),b.item())

plt.plot(line,o)
plt.scatter(line,noised_output)
plt.show()



