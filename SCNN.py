import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import psutil

import MNIST_handle
import SNN_CORE


K = 1e2
K2 = 1e-3
W1 = 1
W2 = 1
TRAINING_BATCH = 10
learning_rate = 1e-3

mnist = MNIST_handle.MnistData()

lr = tf.placeholder(tf.float32)
input_real = tf.placeholder(tf.float32)
input_real_bin = tf.where(input_real>0.5,tf.ones_like(input_real),tf.zeros_like(input_real))
output_real = tf.placeholder(tf.float32)
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

layer1 = SNN_CORE.SCNN(kernel_size=3,in_channel=1,out_channel=8,strides=2)
layer2 = SNN_CORE.SCNN(kernel_size=3,in_channel=8,out_channel=16,strides=2)
layer3 = SNN_CORE.SCNN(kernel_size=3,in_channel=16,out_channel=16,strides=2)
layer4 = SNN_CORE.SCNN_upsample(kernel_size=3,in_channel=16,out_channel=16,strides=2)
layer5 = SNN_CORE.SCNN_upsample(kernel_size=3,in_channel=16,out_channel=16,strides=2)
layer6 = SNN_CORE.SCNN_upsample(kernel_size=3,in_channel=16,out_channel=11,strides=2)
layerout3 = layer3.forward(layer2.forward(layer1.forward(input_exp)))
layerout6 = layer6.forward(layer5.forward(layer4.forward(layerout3)))

layerout_first = tf.where(tf.equal(tf.reduce_min(layerout6,axis=3,keepdims=True),layerout6),tf.ones_like(layerout6),tf.zeros_like(layerout6))

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
wsc4,l24 = layer4.scnn.kernel.cost()
wsc5,l25 = layer5.scnn.kernel.cost()
wsc6,l26 = layer6.scnn.kernel.cost()

wsc = wsc1+wsc2+wsc3+wsc4+wsc5+wsc6
l2 = l21+l22+l23+l24+l25+l26

cost = K*wsc + K2*l2 + W1*front_loss + W2*back_loss

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)




config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

plt.ion()
fig = plt.figure()
p = fig.add_subplot(111)

xs, ys = mnist.next_batch(10)
xs = np.reshape(xs, [-1, 28, 28, 1])

while(True):

    [c,o,_] = sess.run([front_loss,layerout_first,train_op], {input_real: xs, output_real: ys})
    print(c)
    p.imshow(o[0,:,:,10])
    fig.canvas.draw()
    plt.pause(0.1)

xs,ys = mnist.next_batch(10)
xs = np.reshape(xs,[-1,28,28,1])
r_out = sess.run(front_loss,{input_real:xs,output_real:ys})
print(r_out)
l3 = sess.run(layerout3,{input_real:xs})
print(np.shape(l3))
out = sess.run(layerout6,{input_real:xs})
print(np.shape(out))
f1 = plt.subplot(221)
f2 = plt.subplot(222)
f3 = plt.subplot(223)
f4 = plt.subplot(224)
f1.imshow(r_out[0,:,:,10])
f2.imshow(r_out[0,:,:,0])
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
