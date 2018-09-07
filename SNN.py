import numpy as np
import torch
import matplotlib.pyplot as plt


line = np.linspace(0,10,100)
groundtruth = 5*line + 15
noise = np.random.normal(0,5,100)
noised_output = groundtruth+noise
'''
f1 = plt.subplot(211)
f1.scatter(line,groundtruth)
f2 = plt.subplot(212)
f2.scatter(line,noised_output)
plt.show()
'''


k = torch.tensor(0.,dtype=torch.float32,requires_grad=True)
b = torch.tensor(0.,dtype=torch.float32,requires_grad=True)
l = torch.tensor(line,dtype=torch.float32)

output = k * l + b.expand(l.size())

loss = torch.mean((output - torch.tensor(noised_output,dtype=torch.float32))**2)

lr = 1e-3
for i in range(1000):
    print("this is loss: "+repr(loss))
    loss.backward(retain_graph=True)
    k = k - k.grad
    b = b - b.grad




