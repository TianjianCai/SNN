import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE

w = [[0.1,0.25,0.3,0.5],
     [0.2, 0.25, 0.4, 0.5],
     [0.1, 0.25, 0.3, 0.5]]

b = [0.21, 0.26, 0.01, 0.5]

x = tf.placeholder(tf.float32)
layer = SNN_CORE.SNNLayer_Multi(3,4,2,2,w,b)
layer2 = SNN_CORE.SNNLayer(3,4,w)
y = layer.forward(x)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

input = np.arange(1,5*3*2+1,1)
input = np.reshape(input,[5,3,2])

print(input)

out = sess.run(y,{x:input})
print(out)

