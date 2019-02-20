import sys
import numpy as np
sys.path.append("..")
import SNN
import tensorflow as tf
import os


K = 100
K2 = 1e-2
TRAINING_BATCH = 10
learning_rate = 1e-3



lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_real_resize = tf.reshape(tf.exp(input_real),[TRAINING_BATCH,28,28,1])

layer1 = SNN.SCNNLayer(kernel_size=5,in_channel=1,out_channel=32,strides=2)
layer2 = SNN.SCNNLayer(kernel_size=5,in_channel=32,out_channel=16,strides=2)
layer3 = SNN.SNNLayer(in_size=784,out_size=10)
layerout1 = layer1.forward(input_real_resize)
layerout2 = layer2.forward(layerout1)
layerout3 = layer3.forward(tf.reshape(layerout2,[TRAINING_BATCH,784]))


wsc1 = layer1.kernel.w_sum_cost()
wsc2 = layer2.kernel.w_sum_cost()
wsc3 = layer3.w_sum_cost()

wsc = wsc1 + wsc2 + wsc3

l21 = layer1.kernel.l2_cost()
l22 = layer2.kernel.l2_cost()
l23 = layer3.l2_cost()

l2 = l21+l22+l23


layerout_groundtruth = tf.concat([layerout3, output_real], 1)
output_loss = tf.reduce_mean(tf.map_fn(SNN.loss_func, layerout_groundtruth))

cost = K*wsc + K2*l2 + output_loss

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)



config = tf.ConfigProto(
    device_count={'GPU': 1}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


print("training started")

scale = 3
mnist = SNN.MnistData(path=["MNIST/train-images.idx3-ubyte","MNIST/train-labels.idx1-ubyte"])

while(True):
    xs, ys = mnist.next_batch(TRAINING_BATCH, shuffle=True)
    xs = scale * (-xs + 1)
    xs = np.reshape(xs, [-1, 28, 28, 1])
    [c,l,lo,_] = sess.run([cost,output_loss,layerout3,train_op], {input_real: xs, output_real: ys, lr:learning_rate})
    step = sess.run(step_inc_op)
    if step % 100 == 0:
        print(repr(step)+', '+repr(c)+', '+repr(l) + '\n'+repr(lo[0,:])+'\n'+repr(ys[0,:]))



