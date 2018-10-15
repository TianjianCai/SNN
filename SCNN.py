import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE


K = 1e2
K2 = 1e-3
W1 = 0.9
W2 = 0.1
TRAINING_BATCH = 1
learning_rate = 1e-3

mnist = MNIST_handle.MnistData()

lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
input_real_bin = tf.where(input_real>0.5,tf.ones_like(input_real),tf.zeros_like(input_real))
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_pad = tf.pad(input_real_bin,tf.constant([[0,0],[2,2],[2,2],[0,0]]),"CONSTANT")
input_real_invert = tf.where(input_real_pad>0.5,tf.zeros_like(input_real_pad),tf.ones_like(input_real_pad))
input_exp = tf.reshape(tf.exp(input_real_pad*1.79),[TRAINING_BATCH,32,32,1])
output_real_4d = tf.reshape(output_real,[-1,1,1,10])
zeros_4d = tf.zeros_like(output_real_4d)
output_real_fig = tf.multiply(input_real_pad,output_real_4d)
output_real_front = tf.concat((output_real_fig,tf.zeros([tf.shape(output_real_fig)[0],tf.shape(output_real_fig)[1],tf.shape(output_real_fig)[2],1])),axis=3)
back_4d = tf.concat((zeros_4d,tf.ones([tf.shape(zeros_4d)[0],tf.shape(zeros_4d)[1],tf.shape(zeros_4d)[2],1])),axis=3)
output_real_back = tf.multiply(input_real_invert,back_4d)

layer1 = SNN_CORE.SCNN(kernel_size=5,in_channel=1,out_channel=8,strides=2)
layer2 = SNN_CORE.SCNN(kernel_size=5,in_channel=8,out_channel=16,strides=2)
layer3 = SNN_CORE.SCNN(kernel_size=5,in_channel=16,out_channel=32,strides=2)
layer4 = SNN_CORE.SCNN(kernel_size=5,in_channel=32,out_channel=64,strides=2)
layer5 = SNN_CORE.SCNN_upsample(kernel_size=11,in_channel=64,out_channel=32,strides=4)
layer6 = SNN_CORE.SCNN_upsample(kernel_size=11,in_channel=32,out_channel=11,strides=4)
layerout3 = layer3.forward(layer2.forward(layer1.forward(input_exp)))
layerout6 = layer6.forward(layer5.forward(layer4.forward(layerout3)))

layerout_first = tf.one_hot(tf.argmin(layerout6,axis=3),depth=11)

both_front = tf.concat((layerout6,output_real_front),axis=3)
both_back = tf.concat((layerout6,output_real_back),axis=3)

def loss_func_3d(both):
    output,groundtruth = tf.split(both,[11,11],axis=2)
    z1 = tf.reduce_sum(tf.multiply(tf.exp(tf.subtract(0.,output)),groundtruth),2,keepdims=True)
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0.,output)),2,keepdims=True)
    loss = tf.subtract(0.,tf.log(tf.clip_by_value(tf.divide(z1,tf.clip_by_value(z2,1e-10,1e10)),1e-10,1)))
    return tf.reduce_mean(loss)

front_loss = tf.reduce_mean(tf.map_fn(loss_func_3d,both_front))
back_loss = tf.reduce_mean(tf.map_fn(loss_func_3d,both_back))

wsc1,l21 = layer1.kernel.cost()
wsc2,l22 = layer2.kernel.cost()
wsc3,l23 = layer3.kernel.cost()
wsc4,l24 = layer4.kernel.cost()
wsc5,l25 = layer5.scnn.kernel.cost()
wsc6,l26 = layer6.scnn.kernel.cost()

wsc = wsc1+wsc2+wsc3+wsc4+wsc5+wsc6
l2 = l21+l22+l23+l24+l25+l26

cost = K*wsc + K2*l2 + W1*front_loss + W2*back_loss

g_l1,g_l2,g_l3,g_l4,g_l5,g_l6 = tf.gradients(cost,[layer1.kernel.weight,layer2.kernel.weight,layer3.kernel.weight,layer4.kernel.weight,layer5.scnn.kernel.weight,layer6.scnn.kernel.weight])
grad_sum_sqrt = tf.clip_by_value(
    tf.sqrt(
        tf.reduce_sum(
            tf.square(
                g_l1.values)) +
        tf.reduce_sum(
            tf.square(
                g_l2.values)) +
        tf.reduce_sum(
            tf.square(
                g_l3.values))+
        tf.reduce_sum(
            tf.square(
                g_l4.values)) +
        tf.reduce_sum(
            tf.square(
                g_l5.values)) +
        tf.reduce_sum(
            tf.square(
                g_l6.values))
    ),
    1e-10,
    10)
g_l1_normed = tf.divide(g_l1.values, grad_sum_sqrt)
g_l2_normed = tf.divide(g_l2.values, grad_sum_sqrt)
g_l3_normed = tf.divide(g_l3.values, grad_sum_sqrt)
g_l4_normed = tf.divide(g_l4.values, grad_sum_sqrt)
g_l5_normed = tf.divide(g_l5.values, grad_sum_sqrt)
g_l6_normed = tf.divide(g_l6.values, grad_sum_sqrt)

train_op_1 = tf.scatter_add(
    layer1.kernel.weight, g_l1.indices, -lr * g_l1_normed)
train_op_2 = tf.scatter_add(
    layer2.kernel.weight, g_l2.indices, -lr * g_l2_normed)
train_op_3 = tf.scatter_add(
    layer3.kernel.weight, g_l3.indices, -lr * g_l3_normed)
train_op_4 = tf.scatter_add(
    layer4.kernel.weight, g_l4.indices, -lr * g_l4_normed)
train_op_5 = tf.scatter_add(
    layer5.scnn.kernel.weight, g_l5.indices, -lr * g_l5_normed)
train_op_6 = tf.scatter_add(
    layer6.scnn.kernel.weight, g_l6.indices, -lr * g_l6_normed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

plt.ion()
fig = plt.figure()
p1 = fig.add_subplot(221)
p2 = fig.add_subplot(222)
p3 = fig.add_subplot(223)
p4 = fig.add_subplot(224)


def cal_lr(lr, step_num):
    bias = 1e-4
    return (lr * np.exp(step_num * -1e-5)) + bias


xs, ys = mnist.next_batch(TRAINING_BATCH)
xs = np.reshape(xs, [-1, 28, 28, 1])

while(True):

    [c,o,lo,_,_,_,_,_,_] = sess.run([cost,layerout_first,layerout6,train_op_1,train_op_2,train_op_3,train_op_4,train_op_5,train_op_6], {input_real: xs, output_real: ys,lr: cal_lr(learning_rate, sess.run(global_step))})
    sess.run(step_inc_op)
    print(c)
    p1.imshow(o[0,:,:,0])
    p2.imshow(o[0, :, :, 5])
    p3.imshow(o[0, :, :, 10])
    p4.imshow(o[0, :, :, 9])
    fig.canvas.draw()
    plt.pause(0.1)

