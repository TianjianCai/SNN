import numpy as np
import tensorflow as tf
import os

import SNN_CORE

input = tf.placeholder(tf.float32)
input_exp = tf.exp(input)
groundtruth = tf.placeholder(tf.float32)

layer_in = SNN_CORE.SNNLayer_new(2,4)
layer_out = SNN_CORE.SNNLayer_new(4,2)

layerin_out = layer_in.forward(input_exp)
layerout_out = layer_out.forward(layerin_out)
nnout = tf.log(layerout_out)

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


layerout_groundtruth = tf.concat([layerout_out,groundtruth],1)
loss = tf.reduce_mean(tf.map_fn(loss_func,layerout_groundtruth))

wsc = SNN_CORE.w_sum_cost(layer_in.weight) + SNN_CORE.w_sum_cost(layer_out.weight)
l2c = SNN_CORE.l2_func(layer_in.weight) + SNN_CORE.l2_func(layer_out.weight)

K = 10
K2 = 0.1
learning_rate = 1e-3

cost = loss + K*wsc + K2*l2c

opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

scale = 2
xs = [[0,0],[scale,0],[0,scale],[scale,scale]] + np.random.uniform(0,scale/10,[4,2])
ys = [[1,0],[0,1],[0,1],[1,0]]

print('training started')
step = 1
while(True):
    [out,c,_] = sess.run([nnout,cost,train_op],{input:xs,groundtruth:ys})
    if step % 500 == 1:
        print('step '+repr(step) +', cost='+repr(c))
    if step % 5000 == 1:
        print(out)
    step = step + 1

