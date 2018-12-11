import numpy as np
import tensorflow as tf

import SNN_CORE

w = [[0.1, 0.15, 0.3, 0.5],
     [0.2, 0.25, 0.4, 0.6],
     [0.3, 0.35, 0.5, 0.7]]

x = tf.placeholder(tf.float32)
layer = SNN_CORE.SNNLayer_new(2,4,w=w)
y = layer.forward(x)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

input = np.array([1,2,3,9,8,7])
input = np.reshape(input,[3,2])

print(input)

out = sess.run(y,{x:input})
print(out)

