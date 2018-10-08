import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

import SNN_CORE
import MNIST_handle

x = tf.placeholder(tf.float32)
x_exp = tf.exp(x)
layer1 = SNN_CORE.SNNLayer(784,784)
y = layer1.forward(x_exp)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mnist = MNIST_handle.MnistData()

xs,ys = mnist.next_batch(1)
xs = xs.reshape(-1,784)
out = sess.run(y,{x:xs})

plt.imshow(out.reshape(28,28))
plt.show()
