import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE

K = 100
K2 = 1e-2
TRAINING_BATCH = 30
learning_rate = 1e-3

mnist = MNIST_handle.MnistData()

lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
input_real_bin = tf.where(input_real>0.5,tf.zeros_like(input_real),tf.ones_like(input_real))
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_resize = tf.reshape(1+5*input_real_bin,[TRAINING_BATCH,28,28,1])

pool2 = SNN_CORE.POOLING(2)
pool7 = SNN_CORE.POOLING(7)

layer1 = SNN_CORE.SCNN(kernel_size=3,in_channel=1,out_channel=16,strides=2,wta=False)
layer2 = SNN_CORE.SCNN(kernel_size=3,in_channel=16,out_channel=32,strides=2,wta=False)
layer3 = SNN_CORE.SCNN(kernel_size=3,in_channel=32,out_channel=64,strides=2,wta=False)
layer4 = SNN_CORE.SCNN(kernel_size=3,in_channel=64,out_channel=64,strides=2,wta=False)
layer5 = SNN_CORE.SCNN(kernel_size=3,in_channel=64,out_channel=10,strides=2,wta=False)
layerout1 = layer1.forward(input_real_resize)
layerout2 = layer2.forward(layerout1)
layerout3 = layer3.forward(layerout2)
layerout4 = layer4.forward(layerout3)
layerout5 = tf.reshape(layer5.forward(layerout4),[-1,10])

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
try:
    saver.restore(sess, os.getcwd() + '/save/save.ckpt')
    print('checkpoint loaded')
except BaseException:
    print('cannot load checkpoint')
    exit(-1)

w = sess.run(tf.abs(layer2.kernel.weight))
w = np.swapaxes(w[:144,:],0,1)
w = np.reshape(w,[32,9,16])
w = np.reshape(w,[32,3,3,16])
w = np.swapaxes(w,0,3)

xs, ys = mnist.next_batch(TRAINING_BATCH, shuffle=False)
xs = np.reshape(xs, [-1, 28, 28, 1])

[lo1,lo2,lo3,lo4,lo5] = sess.run([layerout1,layerout2,layerout3,layerout4,layerout5],{input_real:xs})

plt.imshow(np.reshape(np.swapaxes(lo1[0,:,:,:],1,2),[14,224]))
plt.show()
print(lo5)