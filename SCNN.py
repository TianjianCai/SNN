import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

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
        patches = tf.image.extract_image_patches(images=layer_in,ksizes=[1,self.kernel_size,self.kernel_size,1],strides=[1,self.strides,self.strides,1],rates=[1,1,1,1],padding="SAME")
        patches_flatten = tf.reshape(patches,[input_size[0],-1,self.in_channel*self.kernel_size*self.kernel_size])
        patches_infpad = tf.where(tf.less(patches_flatten,0.9),SNN_CORE.MAX_SPIKE_TIME*tf.ones_like(patches_flatten),patches_flatten)
        img_raw = tf.map_fn(self.kernel.forward,patches_infpad)
        img_reshaped = tf.reshape(img_raw,[input_size[0],input_size[1]//self.strides,input_size[2]//self.strides,self.out_channel])
        return img_reshaped


mnist = MNIST_handle.MnistData()
input_real = tf.placeholder(tf.float32)
input_exp = tf.exp(input_real*1.79)
layer1 = SCNN(kernel_size=5,in_channel=1,out_channel=32,strides=2)
layer2 = SCNN(kernel_size=5,in_channel=32,out_channel=16,strides=2)
layer3 = SCNN(kernel_size=5,in_channel=16,out_channel=8,strides=1)
layer4 = SNN_CORE.SNNLayer(in_size=392,out_size=10)
l1out = layer1.forward(input_exp)
l2out = layer2.forward(l1out)
l3out = layer3.forward(l2out)
l4out = layer4.forward(tf.reshape(l3out,[-1,392]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


xs,ys = mnist.next_batch(10)
xs = xs.reshape(-1,28,28,1)
out = sess.run(l4out,{input_real:xs})
print(out)


