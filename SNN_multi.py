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
NOISE_SPIKE = 1

mnist = MNIST_handle.MnistData()

lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
input_real_exp = tf.reshape(tf.exp((1-input_real)*input_scale),[TRAINING_BATCH,784,1])
inc = tf.tile(tf.reshape(tf.linspace(0.,0.1,784),[1,784,1]),[TRAINING_BATCH,1,1])
input_real_exp_inc = input_real_exp + inc
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)

if not NOISE_SPIKE == 0:
    noise = tf.random_uniform([TRAINING_BATCH,784,NOISE_SPIKE],0.,10.)
    noise_exp = tf.exp(noise)
    input_with_noise = tf.concat((input_real_exp_inc,noise_exp),axis=2)
else:
    input_with_noise = input_real_exp_inc

layer1 = SNN_CORE.SNNLayer_Multi(784,800,1+NOISE_SPIKE,1+NOISE_SPIKE)
layer2 = SNN_CORE.SNNLayer_Multi(800,10,1+NOISE_SPIKE,1)

layerout1 = layer1.forward(input_with_noise)
layerout2 = tf.reshape(layer2.forward(layerout1),[TRAINING_BATCH,10])

def loss_func(both):
    """
    function to calculate loss, refer to paper p.7, formula 11
    :param both: a tensor, it put both layer output and expected output together, its' shape
            is [batch_size,out_size*2], where the left part is layer output(real output), right part is
            the label of input(expected output), the tensor both should be looked like this:
            [[2.13,3.56,7.33,3.97,...0,0,1,0,...]
             [3.14,5.56,2.54,15.6,...0,0,0,1,...]...]
                ↑                   ↑
             layer output           label of input
    :return: a tensor, it is a scalar of loss
    """
    output = tf.slice(both, [0], [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    output = tf.divide(output,tf.reduce_min(output)) #to normalize the output
    index = tf.slice(both, [tf.cast(tf.shape(both)[0] / 2, tf.int32)],
                     [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    z1 = tf.exp(tf.subtract(0., tf.reduce_sum(tf.multiply(output, index))))
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0., output)))
    loss = tf.subtract(
        0., tf.log(
            tf.clip_by_value(tf.divide(
                z1, tf.clip_by_value(
                    z2, 1e-10, 1e10)), 1e-10, 1)))
    return loss

layer_real_output = tf.concat([layerout2, output_real], 1)
output_loss = tf.reduce_mean(tf.map_fn(loss_func, layer_real_output))

wsc1 = SNN_CORE.w_sum_cost(layer1.weight)
wsc2 = SNN_CORE.w_sum_cost(layer2.weight)

l21 = SNN_CORE.l2_func(layer1.weight_core)
l22 = SNN_CORE.l2_func(layer2.weight_core)

cost = K*(wsc1+wsc2) + K2*(l21+l22) + output_loss

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)

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

print("training started")
while(True):
    xs, ys = mnist.next_batch(TRAINING_BATCH, shuffle=True)
    [c,lo,_] = sess.run([cost,layerout2,train_op],{input_real:xs,output_real:ys})
    step = sess.run(step_inc_op)
    if step % 10 == 0:
        saver.save(sess, os.getcwd() + '/save/save.ckpt')
        print(repr(step)+', '+repr(c) + '\n' + repr(lo[0,:])+'\n'+repr(ys[0,:]))