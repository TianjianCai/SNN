import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

MAX_SPIKE_TIME = 1e2


class SNNLayer(object):
    def __init__(self, layer_in, in_size, out_size):
        self.weight = tf.Variable(tf.random_normal(
            [in_size, out_size], 2, 1, tf.float32))
        batch_num = tf.shape(layer_in)[0]
        _, input_sorted_indices = tf.nn.top_k(-layer_in, in_size, False)
        map_x = tf.reshape(
            tf.tile(
                tf.reshape(
                    tf.range(
                        start=0, limit=batch_num, delta=1), [
                        batch_num, 1]), [
                    1, in_size]), [
                batch_num, in_size, 1])
        input_sorted_map = tf.concat(
            [map_x, tf.reshape(input_sorted_indices, [batch_num, in_size, 1])], 2)
        input_sorted = tf.gather_nd(params=layer_in, indices=input_sorted_map)
        input_sorted_outsize = tf.tile(
            tf.reshape(
                input_sorted, [
                    batch_num, in_size, 1]), [
                1, 1, out_size])
        weight_sorted = tf.map_fn(
            lambda x: tf.gather(
                self.weight, tf.cast(
                    x, tf.int32)), tf.cast(
                input_sorted_indices, tf.float32))
        weight_input_mul = tf.multiply(weight_sorted, input_sorted_outsize)

        def add_func(index, next_array, source):
            next_array = tf.slice(
                tf.concat(
                    [
                        tf.reduce_sum(
                            tf.slice(source, [0, 0], [index + 1, out_size]), 0, True
                        ), next_array
                    ], 0
                ), [0, 0], [in_size, out_size]
            )
            return [index + 1, next_array, source]

        def loop_cond(index, next_array, source):
            return index < in_size

        def loop_init(source):
            index = tf.constant(0)
            next_array = tf.zeros([in_size, out_size], tf.float32)
            return [index, next_array, source]

        def loop_func(loop_matrix):
            _, result, _ = tf.while_loop(
                loop_cond, add_func, loop_init(loop_matrix))
            return tf.reverse(result, [0])
        weight_sumed = tf.map_fn(loop_func, weight_sorted)
        weight_input_sumed = tf.map_fn(loop_func, weight_input_mul)
        output_spike_all = tf.divide(
            weight_input_sumed, tf.subtract(
                weight_sumed, 1.))
        valid_cond_1 = tf.where(
            weight_sumed > 1,
            tf.ones_like(weight_sumed),
            tf.zeros_like(weight_sumed))
        input_sorted_outsize_left = tf.slice(tf.concat([input_sorted_outsize, MAX_SPIKE_TIME * tf.ones(
            [batch_num, 1, out_size])], 1), [0, 1, 0], [batch_num, in_size, out_size])
        valid_cond_2 = tf.where(
            output_spike_all < input_sorted_outsize_left,
            tf.ones_like(input_sorted_outsize),
            tf.zeros_like(input_sorted_outsize))
        valid_cond_both = tf.where(
            tf.equal(
                valid_cond_1 +
                valid_cond_2,
                2),
            tf.ones_like(valid_cond_1),
            tf.zeros_like(valid_cond_1))
        valid_cond_both_extend = tf.concat(
            [valid_cond_both, tf.ones([batch_num, 1, out_size])], 1)
        output_spike_all_extent = tf.concat(
            [output_spike_all, MAX_SPIKE_TIME * tf.ones([batch_num, 1, out_size])], 1)
        output_valid_both = tf.concat(
            [output_spike_all_extent, valid_cond_both_extend], 1)

        def select_output(both):
            value = tf.transpose(
                tf.slice(
                    both, [
                        0, 0], [
                        in_size + 1, out_size]))
            valid = tf.transpose(
                tf.slice(both, [in_size + 1, 0], [in_size + 1, out_size]))
            pos = tf.cast(tf.where(tf.equal(valid, 1)), tf.int32)
            pos_reduced = tf.concat([tf.reshape(tf.range(0, out_size), [out_size, 1]), tf.reshape(
                tf.segment_min(pos[:, 1], pos[:, 0]), [out_size, 1])], 1)
            return tf.gather_nd(value, pos_reduced)
        layer_out = tf.map_fn(select_output, output_valid_both)
        self.out = layer_out


def w_sum_cost(W):
    part1 = tf.subtract(1., tf.reduce_sum(W, 0))
    part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
    return tf.reduce_sum(part2)


def l2_func(W):
    w_sqr = tf.square(W)
    return tf.reduce_sum(w_sqr)


def loss_func(both):
    output = tf.slice(both, [0], [tf.cast(tf.shape(both)[0] / 2,tf.int32)])
    index = tf.slice(both, [tf.cast(tf.shape(both)[0] / 2,tf.int32)], [tf.cast(tf.shape(both)[0] / 2,tf.int32)])
    z1 = tf.exp(tf.subtract(0., tf.reduce_sum(tf.multiply(output, index))))
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0., output)))
    loss = tf.subtract(
        0., tf.log(
            tf.divide(
                z1, tf.clip_by_value(
                    z2, 1e-5, 1e5))))
    return loss


K = 100
K2 = 0.001
learning_rate = 1e-1

real_input = tf.placeholder(tf.float32)
real_output = tf.placeholder(tf.float32)

layer1 = SNNLayer(real_input,784,400)
layer2 = SNNLayer(layer1.out,400,400)
layer3 = SNNLayer(layer2.out,400,10)

layer_real_output = tf.concat([layer3.out,real_output],1)
output_loss = tf.reduce_mean(tf.map_fn(loss_func,layer_real_output))
WC1 = w_sum_cost(layer1.weight)
WC2 = w_sum_cost(layer2.weight)
WC3 = w_sum_cost(layer3.weight)
L2_1 = l2_func(layer1.weight)
L2_2 = l2_func(layer2.weight)
L2_3 = l2_func(layer3.weight)
cost = K*(WC1+WC2+WC3)+K2*(L2_1+L2_2+L2_3)+output_loss
opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)

config = tf.ConfigProto(
    device_count={'GPU': 1}
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

i=0
while(1):
    xs, ys = mnist.train.next_batch(10)
    print(sess.run(cost,{real_input:xs,real_output:ys}))
    sess.run(train_op,{real_input:xs,real_output:ys})
    if i%10 == 0:
        saver.save(sess, os.getcwd() + '/save/save.ckpt')
        print("checkpoint saved")
    i = i+1
