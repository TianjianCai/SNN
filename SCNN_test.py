import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import psutil
import time

import MNIST_handle
import SNN_CORE

class SCNN(object):
    def __init__(self,kernel_size=3,in_channel=1,out_channel=1,strides=1):
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.strides = strides
        self.kernel = SNN_CORE.SNNLayer(in_size=self.kernel_size*self.kernel_size*self.in_channel,out_size=self.out_channel)

    def forward(self,layer_in):
        input_size = tf.shape(layer_in)
        patches = tf.extract_image_patches(images=layer_in,ksizes=[1,self.kernel_size,self.kernel_size,1],strides=[1,self.strides,self.strides,1],rates=[1,1,1,1],padding="SAME")
        patches_flatten = tf.reshape(patches,[input_size[0],-1,self.in_channel*self.kernel_size*self.kernel_size])
        patches_infpad = tf.where(tf.less(patches_flatten,0.9),SNN_CORE.MAX_SPIKE_TIME*tf.ones_like(patches_flatten),patches_flatten)
        img_raw = tf.map_fn(self.kernel.forward,patches_infpad)
        img_reshaped = tf.reshape(img_raw,[input_size[0],input_size[1]//self.strides,input_size[2]//self.strides,self.out_channel])
        return img_reshaped


TESTING_DATA_SIZE = 100
TESTING_BATCH = 10

SLEEP_TIME = 10


mnist = MNIST_handle.MnistData()

input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_exp = tf.reshape(tf.exp(input_real*1.79),[TESTING_BATCH,28,28,1])

layer1 = SCNN(kernel_size=5,in_channel=1,out_channel=32,strides=2)
layer2 = SCNN(kernel_size=3,in_channel=32,out_channel=16,strides=2)
#layer3 = SNN_CORE.SNNLayer(in_size=784,out_size=10)
layer4 = SNN_CORE.SNNLayer(in_size=784,out_size=10)
l1out = layer1.forward(input_exp)
l2out = layer2.forward(l1out)
#l3out = layer3.forward(tf.reshape(l2out,[-1,784]))
l4out = layer4.forward(tf.reshape(l2out,[-1,784]))


layer_real_output = tf.concat([l4out,output_real],1)

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)


layer_output_pos = tf.argmin(l4out, 1)
real_output_pos = tf.argmax(output_real, 1)
accurate = tf.reduce_mean(
    tf.where(
        tf.equal(
            layer_output_pos, real_output_pos), tf.ones_like(
                layer_output_pos, dtype=tf.float32), tf.zeros_like(
                    layer_output_pos, dtype=tf.float32)))



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

sess.graph.finalize()
process = psutil.Process(os.getpid())

def cal_lr(lr, step_num):
    bias = 1e-4
    return (lr * np.exp(step_num * -1e-5)) + bias


xs, ys = mnist.next_batch(TESTING_BATCH,shuffle=True)
xs = xs.reshape(-1,28,28,1)
print(sess.run(l4out,{input_real: xs, output_real: ys}))
print(ys)



saved_path = os.getcwd() + "/save/checkpoint"
modified_time = None
if os.path.exists(saved_path):
    modified_time = os.stat(saved_path)[8]-1
else:
    print("No tf checkpoint found, Program will exit")
    exit(1)

while True:
    if os.stat(saved_path)[8] > modified_time:
        modified_time = os.stat(saved_path)[8]
        saver.restore(sess, os.getcwd() + '/save/save.ckpt')

        xs, ys = mnist.next_batch(TESTING_BATCH,shuffle=True)
        xs = xs.reshape(-1, 28, 28, 1)
        out = sess.run(l4out, {input_real: xs[0:10], output_real: ys[0:10]})
        print(out)
        print(ys[0:10])
        j = 0
        acc = 0
        while (j < TESTING_DATA_SIZE / TESTING_BATCH):
            xs, ys = mnist.next_batch(TESTING_BATCH)
            xs = xs.reshape(-1, 28, 28, 1)
            acc = acc + sess.run(accurate, {input_real: xs, output_real: ys})
            j = j + 1
        acc = acc / (TESTING_DATA_SIZE / TESTING_BATCH)
        print("------accurate: ", repr(acc))
        with open(os.getcwd() + "/accuracy.txt", "a") as f:
            f.write("\n" + repr(sess.run(global_step)) + ", " + repr(acc))
    time.sleep(SLEEP_TIME)
