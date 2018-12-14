import numpy as np
import os
from scipy.signal import lfilter
import matplotlib.pyplot as plt


SAVE_PATH = os.getcwd() + '/weight_xor'

def snn_forward(x,w,Ts,scale=1):
    bias = np.zeros([1,np.shape(x)[1]])
    bias[0,0] = 1
    x = np.append(x,bias,axis=0)
    x_added = np.dot(np.transpose(w),x)
    current = lfilter([scale],[1,-np.exp(-Ts*scale)],x_added)
    potential = lfilter([Ts],[1,-1],current)
    fg,ax = plt.subplots(nrows=2,ncols=1)
    for i in range(np.shape(potential)[0]):
        ax[1].plot(potential[i])
        ax[0].plot(current[i])
    plt.show()
    spikes_index = np.argmax(potential>1,axis=1)
    spikes_index[spikes_index==0] = np.shape(potential)[1]-1
    spikes = np.zeros_like(potential)
    spikes[np.arange(np.shape(potential)[0]),spikes_index] = 1
    return spikes

w1 = np.load(SAVE_PATH + '1.npy')
w2 = np.load(SAVE_PATH + '2.npy')

Ts = 1e-3
scale = 2

input_mat = np.zeros([2,int(1/Ts*2)])
input_mat[0,int(1/Ts*0.1)] = 1
input_mat[1,int(1/Ts*1.1)] = 1

l1out = snn_forward(input_mat,w1,Ts,scale)
l2out = snn_forward(l1out,w2,Ts,scale)


plt.plot(l2out[0])
plt.plot(l2out[1])
plt.show()