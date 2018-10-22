import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import SNN_CORE

layer_input = np.array([1,1,1,2,2,2,3,3,3],dtype=np.float32)
layer_weight = np.array([1,1,1,32],dtype=np.float32)


layer_weight_snn = layer_weight.reshape([-1,1])
layer_input_snn = layer_input.reshape([3,-1])
snn = SNN_CORE.SNNLayer(3,1,layer_weight_snn)
snn_in = tf.placeholder(tf.float32)
snn_out = snn.forward(snn_in)
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
tf_out = sess.run(snn_out,{snn_in:layer_input_snn})
actual_out = tf_out
print(actual_out)



