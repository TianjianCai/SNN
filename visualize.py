import numpy as np
import matplotlib.pyplot as plt

train_data = np.genfromtxt('cost.txt',delimiter=',')
print(train_data)

fig, ax1 = plt.subplots()
ax1.plot(train_data[:,0],train_data[:,4])
ax1.set_xlabel('iterations')
ax1.set_ylabel('grad')
ax1.tick_params('y')
'''
ax2 = ax1.twinx()
ax2.plot(train_data[:,0],train_data[:,4],'b-')
ax2.set_ylabel('grad',color='b')
ax2.tick_params('y',colors='b')
'''
#fig.tight_layout()
plt.show()
