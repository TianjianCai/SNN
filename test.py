import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE

TRAINING_BATCH = 30

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

layer1 = SNN_CORE.SCNN(kernel_size=3,in_channel=1,out_channel=20,strides=2)
layer2 = SNN_CORE.SCNN(kernel_size=5,in_channel=20,out_channel=20,strides=2)
layer3 = SNN_CORE.SCNN(kernel_size=5,in_channel=20,out_channel=40,strides=2)
layer4 = SNN_CORE.SCNN(kernel_size=5,in_channel=40,out_channel=32,strides=2)
layer5 = SNN_CORE.SCNN(kernel_size=3,in_channel=32,out_channel=10,strides=2)

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

out = sess.run(tf.reduce_sum(layer5.kernel.weight,0))
print(out)