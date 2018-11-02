import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE

x = tf.placeholder(tf.float32)
layer = SNN_CORE.POOLING(10)
y = layer.forward(x)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

input = np.arange(1,101,1)
input = np.reshape(input,[1,10,10,1])

print(input)

out = sess.run(y,{x:input})
print(out)

plt.imshow(out[0,:,:,0])
plt.show()