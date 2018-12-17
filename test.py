import numpy as np
import os
from scipy.signal import lfilter
import matplotlib.pyplot as plt

import MNIST_handle


SAVE_PATH = os.getcwd() + '/weight_mnist'
mnist = MNIST_handle.MnistData(path=["MNIST/t10k-images.idx3-ubyte","MNIST/t10k-labels.idx1-ubyte"])

def snn_forward(x,w,Ts,scale=1):
    bias = np.zeros([1,np.shape(x)[1]])
    bias[0,0] = 1
    x = np.append(x,bias,axis=0)
    x_added = np.dot(np.transpose(w),x)
    current = lfilter([scale],[1,-np.exp(-Ts*scale)],x_added)
    potential = lfilter([Ts],[1,-1],current)
    #fg,ax = plt.subplots(nrows=2,ncols=1)
    #for i in range(np.shape(potential)[0]):
        #ax[1].plot(potential[i])
        #ax[0].plot(current[i])
    #plt.show()
    spikes_index = np.argmax(potential>1,axis=1)
    spikes_index[spikes_index==0] = np.shape(potential)[1]-1
    spikes = np.zeros_like(potential)
    spikes[np.arange(np.shape(potential)[0]),spikes_index] = 1
    return spikes

w1 = np.load(SAVE_PATH + '1.npy')
w2 = np.load(SAVE_PATH + '2.npy')

Ts = 1e-3
scale = 2

xs, ys = mnist.next_batch(1, shuffle=True)
xs = (1-xs[0,:])/Ts
print(ys)

input_mat = np.zeros([784,int(1/Ts*2)])
input_mat[range(784),xs.astype(int)] = 1

l1out = snn_forward(input_mat,w1,Ts,scale)
l2out = snn_forward(l1out,w2,Ts,scale)

plt.figure(1)
plt.plot(l2out[0],label='0')
plt.plot(l2out[1],label='1')
plt.plot(l2out[2],label='2')
plt.plot(l2out[3],label='3')
plt.plot(l2out[4],label='4')
plt.plot(l2out[5],label='5')
plt.plot(l2out[6],label='6')
plt.plot(l2out[7],label='7')
plt.plot(l2out[8],label='8')
plt.plot(l2out[9],label='9')
plt.legend(loc='upper left')
plt.figure(2)
plt.imshow(np.reshape(xs,[28,28]),cmap='gray')

plt.show()