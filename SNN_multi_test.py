import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE

K = 100
K2 = 1e-3
TRAINING_BATCH = 10
learning_rate = 1e-3
input_scale = 1.79
NOISE_SPIKE = 0
INNER_SPIKE = 5

mnist = MNIST_handle.MnistData()

lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
input_real_reshaped = tf.reshape((2-input_real)*input_scale,[TRAINING_BATCH,784,1])
inc = tf.tile(tf.reshape(tf.linspace(0.,2.,784),[1,784,1]),[TRAINING_BATCH,1,1])
input_real_exp_inc = tf.exp(input_real_reshaped + inc)
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)

if not NOISE_SPIKE == 0:
    noise = tf.random_uniform([TRAINING_BATCH,784,NOISE_SPIKE],0.,10.)
    noise_exp = tf.exp(input_scale*noise)
    input_with_noise = tf.concat((input_real_exp_inc,noise_exp),axis=2)
else:
    input_with_noise = input_real_exp_inc

layer1 = SNN_CORE.SNNLayer_Multi(784,800,1+NOISE_SPIKE,INNER_SPIKE)
layer2 = SNN_CORE.SNNLayer_Multi(800,10,INNER_SPIKE,1)

layerout1 = layer1.forward(input_with_noise)
layerout2 = tf.reshape(layer2.forward(layerout1),[TRAINING_BATCH,10])

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
try:
    saver.restore(sess, os.getcwd() + '/save/save.ckpt')
    print('checkpoint loaded')
except BaseException:
    print('cannot load checkpoint')

xs, ys = mnist.next_batch(TRAINING_BATCH, shuffle=True)
img = sess.run(tf.reduce_min(input_with_noise,axis=2),{input_real:xs})
l1o = sess.run(layerout1,{input_real:xs})
print(l1o)
plt.imshow(np.reshape(img[0,:],[28,28]))
plt.show()