import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE


K = 1e2
K2 = 1e-3
TRAINING_BATCH = 10
learning_rate = 1e-3

mnist = MNIST_handle.MnistData()

lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
output_real = tf.placeholder(tf.float32)
'''
Here is a reshape, because TensorFlow DO NOT SUPPORT tf.extract_image_patches gradients operation for VARIABLE SIZE inputs
'''
input_exp = tf.reshape(tf.exp(input_real*1.79),[TRAINING_BATCH,28,28,1])


layer1 = SNN_CORE.SCNN(kernel_size=5,in_channel=1,out_channel=16,strides=2)
layer2 = SNN_CORE.SCNN(kernel_size=3,in_channel=16,out_channel=32,strides=2)
layer3 = SNN_CORE.SCNN(kernel_size=3,in_channel=32,out_channel=64,strides=2)
layer4 = SNN_CORE.SCNN_upsample(kernel_size=3,in_channel=64,out_channel=32,strides=2)
layer5 = SNN_CORE.SCNN_upsample(kernel_size=3,in_channel=32,out_channel=16,strides=2)
layer6 = SNN_CORE.SCNN_upsample(kernel_size=3,in_channel=16,out_channel=2,strides=2)
layerout3 = layer3.forward(layer2.forward(layer1.forward(input_exp)))
layerout6 = layer6.forward(layer5.forward(layer4.forward(layerout3)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
xs,ys = mnist.next_batch(10)
xs = np.reshape(xs,[-1,28,28,1])
l3 = sess.run(layerout3,{input_real:xs})
print(np.shape(l3))
out = sess.run(layerout6,{input_real:xs})
print(np.shape(out))
f1 = plt.subplot(221)
f2 = plt.subplot(222)
f3 = plt.subplot(223)
f4 = plt.subplot(224)
f1.imshow(out[0,:,:,0])
f2.imshow(out[0,:,:,1])
f3.imshow(l3[0,:,:,0])
f4.imshow(l3[0,:,:,1])
plt.show()
exit(0)




'''
Define the Network
'''
layer1 = SNN_CORE.SCNN(kernel_size=5,in_channel=1,out_channel=32,strides=2)
layer2 = SNN_CORE.SCNN(kernel_size=3,in_channel=32,out_channel=16,strides=2)
#layer3 = SNN_CORE.SNNLayer(in_size=784,out_size=10)
layer4 = SNN_CORE.SNNLayer(in_size=784,out_size=10)
l1out = layer1.forward(input_exp)
l2out = layer2.forward(l1out)
#l3out = layer3.forward(tf.reshape(l2out,[-1,784]))
l4out = layer4.forward(tf.reshape(l2out,[-1,784]))


layer_real_output = tf.concat([l4out,output_real],1)
output_loss = tf.reduce_mean(tf.map_fn(SNN_CORE.loss_func,layer_real_output))
wsc1,l21 = layer1.kernel.cost()
wsc2,l22 = layer2.kernel.cost()
#wsc3,l23 = layer3.cost()
wsc4,l24 = layer4.cost()
WC = wsc1+wsc2+wsc4
L2 = l21+l22+l24
cost = K*WC + K2*L2 + output_loss

global_step = tf.Variable(1, dtype=tf.int64)
step_inc_op = tf.assign(global_step, global_step + 1)

grad_l1, grad_l2, grad_l4 = tf.gradients(
    cost, [layer1.kernel.weight, layer2.kernel.weight, layer4.weight], colocate_gradients_with_ops=True)
grad_sum_sqrt = tf.clip_by_value(
    tf.sqrt(
        tf.reduce_sum(
            tf.square(
                grad_l1.values)) +
        tf.reduce_sum(
            tf.square(
                grad_l2.values)) +
        tf.reduce_sum(
            tf.square(
                grad_l4.values))
    ),
    1e-10,
    10)
grad_l1_normed = tf.divide(grad_l1.values, grad_sum_sqrt)
grad_l2_normed = tf.divide(grad_l2.values, grad_sum_sqrt)
#grad_l3_normed = tf.divide(grad_l3.values, grad_sum_sqrt)
grad_l4_normed = tf.divide(grad_l4.values, grad_sum_sqrt)
train_op_1 = tf.scatter_add(
    layer1.kernel.weight, grad_l1.indices, -lr * grad_l1_normed)
train_op_2 = tf.scatter_add(
    layer2.kernel.weight, grad_l2.indices, -lr * grad_l2_normed)
#train_op_3 = tf.scatter_add(
#    layer3.weight, grad_l3.indices, -lr * grad_l3_normed)
train_op_4 = tf.scatter_add(
    layer4.weight, grad_l4.indices, -lr * grad_l4_normed)

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


xs, ys = mnist.next_batch(TRAINING_BATCH)
xs = xs.reshape(-1,28,28,1)
print(sess.run(cost,{input_real: xs, output_real: ys}))

i = 1
while(1):
    xs, ys = mnist.next_batch(TRAINING_BATCH)
    xs = xs.reshape(-1, 28, 28, 1)
    [c,l,_, _, _] = sess.run([cost,output_loss,train_op_1, train_op_2, train_op_4], {
                         input_real: xs, output_real: ys, lr: cal_lr(learning_rate, sess.run(global_step))})
    sess.run(step_inc_op)
    if i % 50 == 0:
        tmpstr = repr(sess.run(global_step)) + ", " + repr(c) + ", "+repr(l)
        with open(os.getcwd() + "/cost.txt", "a") as f:
            f.write("\n" + tmpstr)
        print(tmpstr)
        saver.save(sess, os.getcwd() + '/save/save.ckpt')
        mem = process.memory_percent()
        if mem > 70:
            exit(0)
    i = i + 1
