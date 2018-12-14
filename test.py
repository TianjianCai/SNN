import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

def snn_forward(x,w,decay_param):
    x_added = np.dot(w,x)
    current = lfilter([1],[1,-np.exp(-decay_param)*decay_param],x_added)
    potential = lfilter([1],[1,-1],current)
    spikes_index = np.argmax(potential>1,axis=1)
    spikes_index[spikes_index==0] = np.shape(potential)[1]-1
    spikes = np.zeros_like(potential)
    spikes[np.arange(np.shape(potential)[0]),spikes_index] = 1
    return spikes

weight_mat = np.array([[.1,.4,.7],[.6,.4,.5]])
input_mat = np.zeros([3,100])
input_mat[0,70] = 1
input_mat[1,50] = 1
input_mat[2,30] = 1

y = snn_forward(input_mat,weight_mat,.1)

plt.plot(y[0])
plt.plot(y[1])
plt.show()