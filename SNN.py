import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

"""
MAX_SPIKE_TIME determine the maximum firing time. ideally maximum firing time should be inf, 
but 1e5 is plenty to use
"""
MAX_SPIKE_TIME = 1e5


class MnistData(object):
    """
    This class manage the mnist data. when initialized, self.xs_full and self.ys_full contain all the mnist data
    Use next_batch() function to get next batch of data in xs_full and ys_full
    """
    def __init__(self, size, path=["/save/train_data_x", "/save/train_data_y"]):
        """
        This function will try to load saved data from path, if no saved data exists, it will load the data from
        tensorflow's example mnist data and save it to path
        :param size: size is a int, used to determine how many mnist data should be loaded if no saved data exists
        :param path: path is a list with 2 strings, the first string indicate the path of xs(mnist image data), second
                    string indicate the path of ys(label of images)
                    ATTEN: PLEASE CREATE CORRESPONDING DIRECTORY BEFORE RUN THE CODE, OTHERWISE IOERROR MAY OCCUR
        """
        try:
            self.xs_full = np.load(os.getcwd() + path[0] + ".npy")
            self.ys_full = np.load(os.getcwd() + path[1] + ".npy")
            print(path[0]+", "+path[1] + " "+"loaded")
        except:
            self.xs_full, self.ys_full = mnist.train.next_batch(size, shuffle=False)
            np.save(os.getcwd() + path[0], self.xs_full)
            np.save(os.getcwd() + path[1], self.ys_full)
            print("cannot load " + path[0]+", "+path[1] +", get new data")
        self.datasize = size
        self.pointer = 0

    def next_batch(self,batch_size):
        """
        This function can get next batch of data from self.xs_full and self.ys_full
        It uses self.pointer to decide from where to return the xs and ys
        :param batch_size: batch_size is a int, used to determine the batch size of returned xs and ys
        :return: return 2 arrays, the first one is image data, shape is [batch_size,784], last one is label,
                shape is [batch_size,10]
        """
        if self.pointer + batch_size < self.datasize:
            pass
        else:
            self.pointer = 0
            if batch_size >= self.datasize:
                batch_size = self.datasize - 1
        xs = self.xs_full[self.pointer:self.pointer + batch_size, :]
        ys = self.ys_full[self.pointer:self.pointer + batch_size, :]
        self.pointer = self.pointer + batch_size
        return xs, ys


class SNNLayer(object):
    """
    This class draw the graph of a SNN layer
    self.out is the output of SNN layer, its' shape is [batch_size, out_size]
    self.weight is the weight of SNN layer, its' shape is [in_size, out_size]
    """
    def __init__(self, layer_in, in_size, out_size):
        """
        All input, output and weights are tensors.
        :param layer_in: layer_in is a tensor, its' shape should be [batch_size,in_size]
        :param in_size: in_size is a int, determine the size of input
        :param out_size: out_size is a int, determine the size of output
        """
        self.weight = tf.Variable(tf.random_uniform(
            [in_size, out_size], 1. / in_size, 5. / in_size, tf.float32))
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
        weight_sumed = tf.cumsum(weight_sorted,axis=1)
        weight_input_sumed = tf.cumsum(weight_input_mul,axis=1)
        output_spike_all = tf.divide(
            weight_input_sumed, tf.clip_by_value(tf.subtract(
                weight_sumed, 1.), 1e-10, 1e10))
        self.outspikeall = output_spike_all
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
    """
    function to calculate weight sum cost.
    :param W: a tensor, like SNNLayer.weight
    :return: a tensor, it is a scalar
    """
    part1 = tf.subtract(1., tf.reduce_sum(W, 0))
    part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
    return tf.reduce_sum(part2)


def l2_func(W):
    """
    function to calculate l2 weight regularzation
    :param W: a tensor, like SNNLayer.weight
    :return: a tensor, it is a scalar
    """
    w_sqr = tf.square(W)
    return tf.reduce_sum(w_sqr)


def loss_func(both):
    """
    function to calculate loss, refer to paper p.7, formula 11
    :param both: a tensor, it put both layer output and expected output together, its' shape
            is [batch_size,out_size*2], where the left part is layer output(real output), right part is
            the label of input(expected output), the tensor both should be looked like this:
            [[2.13,3.56,7.33,3.97,...0,0,1,0,...]
             [3.14,5.56,2.54,15.6,...0,0,0,1,...]...]
                ↑                   ↑
             layer output           label of input
    :return: a tensor, it is a scalar of loss
    """
    output = tf.slice(both, [0], [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    index = tf.slice(both, [tf.cast(tf.shape(both)[0] / 2, tf.int32)],
                     [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    z1 = tf.exp(tf.subtract(0., tf.reduce_sum(tf.multiply(output, index))))
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0., output)))
    loss = tf.subtract(
        0., tf.log(
            tf.clip_by_value(tf.divide(
                z1, tf.clip_by_value(
                    z2, 1e-10, 1e10)),1e-10,1)))
    return loss


def cal_lr(lr, step_num):
    bias = 1e-4
    return (lr*np.exp(step_num*-1e-4))+bias


"""
K and K2 are used to calculate cost, see paper p.6
learning_rate will decrease exponentially with the increase of step count, see paper p.8
"""
K = 100
K2 = 0.001
learning_rate = 1e-1

TRAINING_DATA_SIZE = 50000
TESTING_DATA_SIZE = 1000
TRAINING_BATCH = 10
TESTING_BATCH = 200

"""
lr is learning rate to be used when training
real_input and real_output are the layer input and expected output
"""
lr = tf.placeholder(tf.float32)
real_input = tf.placeholder(tf.float32)
real_input_exp = tf.exp(real_input*1.79)
real_output = tf.placeholder(tf.float32)

"""
drawing the graph of SNN

"""
layer1 = SNNLayer(real_input_exp, 784, 400)
layer2 = SNNLayer(layer1.out, 400, 400)
layer3 = SNNLayer(layer2.out, 400, 10)

"""
draw the graph to calculate cost to be optimized
"""
layer_real_output = tf.concat([layer3.out, real_output], 1)
output_loss = tf.reduce_mean(tf.map_fn(loss_func, layer_real_output))
WC = w_sum_cost(layer1.weight) + w_sum_cost(layer2.weight) + w_sum_cost(layer3.weight)
L2 = l2_func(layer1.weight) + l2_func(layer2.weight) + l2_func(layer3.weight)
cost = K * WC + K2 * L2 + output_loss

"""
the step count itself and the operation to increase step count
"""
global_step = tf.Variable(1,dtype=tf.int64)
step_inc_op = tf.assign(global_step,global_step+1)

"""
draw the graph to calculate gradient and update wight operations 
"""
grad_l1,grad_l2,grad_l3 = tf.gradients(cost,[layer1.weight,layer2.weight,layer3.weight],colocate_gradients_with_ops=True)
grad_sum_sqrt = tf.sqrt(tf.reduce_sum(tf.square(grad_l1.values)) + tf.reduce_sum(tf.square(grad_l2.values)) + tf.reduce_sum(tf.square(grad_l3.values)))
grad_l1_normed = tf.divide(grad_l1.values,grad_sum_sqrt)
grad_l2_normed = tf.divide(grad_l2.values,grad_sum_sqrt)
grad_l3_normed = tf.divide(grad_l3.values,grad_sum_sqrt)
train_op_1 = tf.scatter_add(layer1.weight,grad_l1.indices,-lr*grad_l1_normed)
train_op_2 = tf.scatter_add(layer2.weight,grad_l2.indices,-lr*grad_l2_normed)
train_op_3 = tf.scatter_add(layer3.weight,grad_l3.indices,-lr*grad_l3_normed)

"""
draw the graph to calculate accurate
"""
layer_output_pos = tf.argmin(layer3.out, 1)
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
    device_count={'GPU': 0}
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

"""
set up mnist data to be used when training and testing(arrays, NOT tensors)
"""
mnistData_train = MnistData(size=TRAINING_DATA_SIZE,path=["/save/train_data_x","/save/train_data_y"])
mnistData_test = MnistData(size=TESTING_DATA_SIZE,path=["/save/test_data_x","/save/test_data_y"])


"""
training using mnist data
"""
i = 1
while(1):
    xs, ys = mnistData_train.next_batch(TRAINING_BATCH)
    [c, _, _, _] = sess.run([cost, train_op_1, train_op_2, train_op_3], {real_input: xs, real_output: ys, lr:cal_lr(learning_rate,sess.run(global_step))})
    sess.run(step_inc_op)
    tmpstr = repr(sess.run(global_step)) + ", " + repr(c)
    print(tmpstr)
    with open(os.getcwd() + "/cost.txt", "a") as f:
        f.write("\n" + tmpstr)
    if i % 10 == 0:
        saver.save(sess, os.getcwd() + '/save/save.ckpt')
        print("checkpoint saved")
        
    i = i + 1
