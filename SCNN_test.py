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
TESTING_BATCH = 10
TESTING_DATA_SIZE = 50000
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
input_real_resize = tf.reshape(1+5*input_real_bin,[TESTING_BATCH,28,28,1])

pool2 = SNN_CORE.POOLING(2)
pool7 = SNN_CORE.POOLING(7)

wta = True
layer1 = SNN_CORE.SCNN(kernel_size=5,in_channel=1,out_channel=32,strides=2,wta=wta)
layer2 = SNN_CORE.SCNN(kernel_size=3,in_channel=32,out_channel=32,strides=2,wta=False)
layer3 = SNN_CORE.SCNN(kernel_size=3,in_channel=32,out_channel=64,strides=2,wta=False)
layer4 = SNN_CORE.SCNN(kernel_size=3,in_channel=64,out_channel=64,strides=2,wta=False)
layer5 = SNN_CORE.SCNN(kernel_size=3,in_channel=64,out_channel=10,strides=2,wta=False)
layerout1 = layer1.forward(input_real_resize)
layerout2 = layer2.forward(layerout1)
layerout3 = layer3.forward(layerout2)
layerout4 = layer4.forward(layerout3)
layerout5 = tf.reshape(layer5.forward(layerout4),[-1,10])

layer_output_pos = tf.argmin(layerout5, 1)
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
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


saver = tf.train.Saver()
try:
    saver.restore(sess, os.getcwd() + '/save/save.ckpt')
    print('checkpoint loaded')
except BaseException:
    print('cannot load checkpoint')
    exit(-1)



w = sess.run(layer5.kernel.weight)
w = np.swapaxes(w[:576,:],0,1)
w = np.reshape(w,[64,9,10])
w = np.reshape(w,[64,3,3,10])
w = np.swapaxes(w,2,3)

xs, ys = mnist.next_batch(TESTING_BATCH, shuffle=False)
xs = np.reshape(xs, [-1, 28, 28, 1])

[lo1,lo2,lo3,lo4,lo5] = sess.run([layerout1,layerout2,layerout3,layerout4,layerout5],{input_real:xs})

print(lo5)
print(ys)

f1 = plt.subplot(511)
f2 = plt.subplot(512)
f3 = plt.subplot(513)
f4 = plt.subplot(514)
f5 = plt.subplot(515)
'''
f1.imshow(np.reshape(np.swapaxes(1/lo4[0,:,:,:],1,2),[2,128]))
f2.imshow(np.reshape(np.swapaxes(1/lo4[1,:,:,:],1,2),[2,128]))
f3.imshow(np.reshape(np.swapaxes(1/lo4[2,:,:,:],1,2),[2,128]))
f4.imshow(np.reshape(np.swapaxes(1/lo4[3,:,:,:],1,2),[2,128]))
f5.imshow(np.reshape(np.swapaxes(1/lo4[4,:,:,:],1,2),[2,128]))
'''
f1.imshow(np.reshape(1/lo5[0,:],[1,10]))
f2.imshow(np.reshape(1/lo5[1,:],[1,10]))
f3.imshow(np.reshape(1/lo5[2,:],[1,10]))
f4.imshow(np.reshape(1/lo5[3,:],[1,10]))
f5.imshow(np.reshape(1/lo5[4,:],[1,10]))
plt.show()

plt.imshow(np.reshape(w,[192,30]))
plt.show()


'''
j=0
totalloop = TESTING_DATA_SIZE/TESTING_BATCH
cum = 0
while j < totalloop:
    xs,ys = mnist.next_batch(TESTING_BATCH,shuffle=False)
    cum += sess.run(accurate,{input_real:xs,output_real:ys})
    if j%(totalloop/50)==0:
        print(repr(j)+"/"+repr(totalloop))
    j += 1
acc = cum/totalloop
print("accurate: "+repr(acc))
'''
