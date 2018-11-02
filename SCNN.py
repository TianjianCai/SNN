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

wsc1 = SNN_CORE.w_sum_cost(layer1.kernel.weight)
wsc2 = SNN_CORE.w_sum_cost(layer2.kernel.weight)
wsc3 = SNN_CORE.w_sum_cost(layer3.kernel.weight)
wsc4 = SNN_CORE.w_sum_cost(layer4.kernel.weight)
wsc5 = SNN_CORE.w_sum_cost(layer5.kernel.weight)

wsc = wsc1 + wsc2 + wsc3 + wsc4 +wsc5

l21 = SNN_CORE.l2_func(layer1.kernel.weight)
l22 = SNN_CORE.l2_func(layer2.kernel.weight)
l23 = SNN_CORE.l2_func(layer3.kernel.weight)
l24 = SNN_CORE.l2_func(layer4.kernel.weight)
l25 = SNN_CORE.l2_func(layer5.kernel.weight)


l2 = l21+l22+l23+l24+l25

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

layer_real_output = tf.concat([layerout5, output_real], 1)
output_loss = tf.reduce_mean(tf.map_fn(loss_func, layer_real_output))

cost = K*wsc + K2*l2 + output_loss

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)



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

print("training started")



while(True):
    xs, ys = mnist.next_batch(TRAINING_BATCH, shuffle=True)
    xs = np.reshape(xs, [-1, 28, 28, 1])
    [c,l,lo,_] = sess.run([cost,output_loss,layerout5,train_op], {input_real: xs, output_real: ys, lr:learning_rate})
    step = sess.run(step_inc_op)
    if step % 10 == 0:
        saver.save(sess, os.getcwd() + '/save/save.ckpt')
        print(repr(step)+', '+repr(c)+', '+repr(l) + '\n'+repr(lo[0,:])+'\n'+repr(ys[0,:]))
    '''
        plt.imshow(lo1[0,:,:,0])
        plt.show()
    '''



