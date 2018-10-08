import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

    def next_batch(self,batch_size,shuffle=False):
        """
        This function can get next batch of data from self.xs_full and self.ys_full
        It uses self.pointer to decide from where to return the xs and ys
        :param batch_size: batch_size is a int, used to determine the batch size of returned xs and ys
        :return: return 2 arrays, the first one is image data, shape is [batch_size,784], last one is label,
                shape is [batch_size,10]
        """
        if shuffle:
            index = np.random.randint(self.datasize, size=batch_size)
            xs = self.xs_full[index, :]
            ys = self.ys_full[index, :]
            return xs, ys
        else:
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

    def __init__(self, layer_in, in_size, out_size,firstlayer=False):
        """
        All input, output and weights are tensors.
        :param layer_in: layer_in is a tensor, its' shape should be [batch_size,in_size]
        :param in_size: in_size is a int, determine the size of input
        :param out_size: out_size is a int, determine the size of output
        """
        in_size = in_size + 1
        self.weight = tf.Variable(tf.random_uniform(
            [in_size, out_size], 1. / in_size, 5. / in_size, tf.float32))
        batch_num = tf.shape(layer_in)[0]
        bias_layer_in = tf.ones([batch_num, 1])
        layer_in = tf.concat([layer_in, bias_layer_in], 1)
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
        weight_sumed = tf.cumsum(weight_sorted, axis=1)
        weight_input_sumed = tf.cumsum(weight_input_mul, axis=1)
        output_spike_all = tf.divide(
            weight_input_sumed, tf.clip_by_value(tf.subtract(
                weight_sumed, 1.), 1e-10, 1e10))
        self.outspikeall = output_spike_all
        valid_cond_1 = tf.where(
            weight_sumed > 1,
            tf.ones_like(weight_sumed),
            tf.zeros_like(weight_sumed))

        def mov_left(input):
            input_unique,input_unique_index,_ = tf.unique_with_counts(input)
            input_unique_left = tf.slice(
                tf.concat((input_unique,[MAX_SPIKE_TIME]),0),[1],[tf.shape(input_unique)[0]])
            return tf.gather(input_unique_left,input_unique_index)
        #input_sorted_outsize_left = tf.slice(tf.concat([input_sorted_outsize, MAX_SPIKE_TIME * tf.ones(
         #   [batch_num, 1, out_size])], 1), [0, 1, 0], [batch_num, in_size, out_size])
        input_sorted_outsize_left = tf.tile(
            tf.reshape(tf.map_fn(mov_left, input_sorted), [
                batch_num, in_size, 1]), [
                1, 1, out_size])
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


TESTING_DATA_SIZE = 1000
TESTING_BATCH = 100

SLEEP_TIME = 10

lr = tf.placeholder(tf.float32)
real_input = tf.placeholder(tf.float32)
real_input_01 = tf.where(real_input>0.5, tf.ones_like(real_input), tf.zeros_like(real_input))
real_input_exp = tf.exp(real_input_01*1.79)
real_output = tf.placeholder(tf.float32)

layer1 = SNNLayer(real_input_exp, 784, 800)
layer2 = SNNLayer(layer1.out, 800, 10)

global_step = tf.Variable(1,dtype=tf.int64)

layer_output_pos = tf.argmin(layer2.out, 1)
real_output_pos = tf.argmax(real_output, 1)
accurate = tf.reduce_mean(
    tf.where(
        tf.equal(
            layer_output_pos, real_output_pos), tf.ones_like(
                layer_output_pos, dtype=tf.float32), tf.zeros_like(
                    layer_output_pos, dtype=tf.float32)))

config = tf.ConfigProto(
    device_count={'GPU': 1}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saved_path = os.getcwd() + "/save/checkpoint"
data_path = ["/save/test_data_x", "/save/test_data_y"]

mnistData_test = MnistData(size=TESTING_DATA_SIZE,path=data_path)

print("testing started...")
saver.restore(sess, os.getcwd() + '/save/save.ckpt')
j = 0
x = np.arange(0,4,0.01)
y1 = np.zeros_like(x)
y2 = np.zeros_like(x)
while (j < TESTING_DATA_SIZE / TESTING_BATCH):
    xs, ys = mnistData_test.next_batch(TESTING_BATCH)
    l1out = sess.run(layer1.out, {real_input: xs, real_output: ys})
    l2out = sess.run(layer2.out, {real_input: xs, real_output: ys})
    l1out = np.log(np.array(l1out))
    l2out = np.log(np.array(l2out))
    l2out = np.min(l2out,axis=1)
    l1hist,_ = np.histogram(l1out,bins=x)
    l2hist,_ = np.histogram(l2out, bins=x)
    l1hist = np.resize(l1hist, np.size(x))
    l2hist = np.resize(l2hist, np.size(x))
    y1 += l1hist
    y2 += l2hist
    j = j + 1
y1 /= np.sum(y1)
y2 /= np.sum(y2)
plt.bar(x,y1,width=0.01,color='#ff0000af')
plt.bar(x,y2,width=0.01,color='#0000ffaf')
plt.show()