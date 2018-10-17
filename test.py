import tensorflow as tf
import numpy as np
import SNN_CORE

layer_input = np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.01],dtype=np.float32)
layer_weight = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,1.1],dtype=np.float32)


# Analytical solution

layer_weight_snn = layer_weight.reshape([-1,1])
layer_input_snn = np.exp(layer_input).reshape([1,-1])

snn = SNN_CORE.SNNLayer(10,1,layer_weight_snn)
snn_in = tf.placeholder(tf.float32)
snn_out = snn.forward(snn_in)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf_out = sess.run(snn_out,{snn_in:layer_input_snn})
actual_out = np.log(tf_out)
print(actual_out)


# descrete solution

timeline = np.zeros([1001])

layer_input_descret = np.ceil(np.append(layer_input,1)*1000).astype(int)
np.add.at(timeline,layer_input_descret,layer_weight)

timeline_fft = np.fft.fft(timeline)

linespace = np.arange(0,1.001,0.001)
exp_line = np.exp(-linespace)

expline_fft = np.fft.fft(exp_line)

mul_fft = expline_fft*timeline_fft

conv_result = np.fft.ifft(mul_fft)

integrate_result = np.cumsum(conv_result*0.001)



print(np.argmax(integrate_result>1))