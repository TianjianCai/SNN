import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

import MNIST_handle
import SNN_CORE

lr = tf.placeholder(tf.float32)
real_input = tf.placeholder(tf.float32)
real_input_01 = tf.where(
    real_input > 0.5,
    tf.ones_like(real_input),
    tf.zeros_like(real_input))
real_input_exp = tf.exp(real_input_01 * 1.79)
real_output = tf.placeholder(tf.float32)

layer1 = SNN_CORE.SNNLayer(real_input_exp, 784, 800)
layer2 = SNN_CORE.SNNLayer(layer1.out, 800, 10)

layer_output_pos = tf.argmin(layer2.out, 1)
real_output_pos = tf.argmax(real_output, 1)
accurate = tf.reduce_mean(
    tf.where(
        tf.equal(
            layer_output_pos, real_output_pos), tf.ones_like(
                layer_output_pos, dtype=tf.float32), tf.zeros_like(
                    layer_output_pos, dtype=tf.float32)))

"""
setting up tensorflow sessions
"""
config = tf.ConfigProto(
    device_count={'GPU': 1}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

"""
try to restore from previous saved checkpoint
"""
saver = tf.train.Saver()
try:
    saver.restore(sess, os.getcwd() + '/save/save.ckpt')
    print('checkpoint loaded')
except BaseException:
    print('cannot load checkpoint')

sess.graph.finalize()

mnistData_train = MNIST_handle.MnistData()
mnistData_test = MNIST_handle.MnistData(path=["MNIST/t10k-images.idx3-ubyte","MNIST/t10k-labels.idx1-ubyte"])

plt.imshow(np.transpose((sess.run(layer1.weight))[0:784,10:20]).reshape([280,28]))
plt.show()