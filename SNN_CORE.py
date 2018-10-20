import tensorflow as tf

MAX_SPIKE_TIME = 1e5

class SNNLayer(object):
    """
    This class draw the graph of a SNN layer
    self.out is the output of SNN layer, its' shape is [batch_size, out_size]
    self.weight is the weight of SNN layer, its' shape is [in_size, out_size]
    """
    def __init__(self, in_size, out_size,w=None):
        self.out_size = out_size
        self.in_size = in_size + 1
        if w is None:
            self.weight = tf.Variable(tf.random_uniform([self.in_size, self.out_size], 1. / self.in_size, 48. / self.in_size, tf.float32))
        else:
            self.weight = tf.Variable(w,dtype=tf.float32)

    def forward(self,layer_in):
        batch_num = tf.shape(layer_in)[0]
        bias_layer_in = tf.ones([batch_num, 1])
        layer_in = tf.concat([layer_in, bias_layer_in], 1)
        _, input_sorted_indices = tf.nn.top_k(-layer_in, self.in_size, False)
        map_x = tf.reshape(
            tf.tile(
                tf.reshape(
                    tf.range(
                        start=0, limit=batch_num, delta=1), [
                        batch_num, 1]), [
                    1, self.in_size]), [
                batch_num, self.in_size, 1])
        input_sorted_map = tf.concat(
            [map_x, tf.reshape(input_sorted_indices, [batch_num, self.in_size, 1])], 2)
        input_sorted = tf.gather_nd(params=layer_in, indices=input_sorted_map)
        input_sorted_outsize = tf.tile(
            tf.reshape(
                input_sorted, [
                    batch_num, self.in_size, 1]), [
                1, 1, self.out_size])
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
            input_unique, input_unique_index, _ = tf.unique_with_counts(input)
            input_unique_left = tf.slice(
                tf.concat((input_unique, [MAX_SPIKE_TIME]), 0), [1], [tf.shape(input_unique)[0]])
            return tf.gather(input_unique_left, input_unique_index)

        # input_sorted_outsize_left = tf.slice(tf.concat([input_sorted_outsize, MAX_SPIKE_TIME * tf.ones(
        #   [batch_num, 1, out_size])], 1), [0, 1, 0], [batch_num, in_size, out_size])
        input_sorted_outsize_left = tf.tile(
            tf.reshape(tf.map_fn(mov_left, input_sorted), [
                batch_num, self.in_size, 1]), [
                1, 1, self.out_size])
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
            [valid_cond_both, tf.ones([batch_num, 1, self.out_size])], 1)
        output_spike_all_extent = tf.concat(
            [output_spike_all, MAX_SPIKE_TIME * tf.ones([batch_num, 1, self.out_size])], 1)
        output_valid_both = tf.concat(
            [output_spike_all_extent, valid_cond_both_extend], 1)

        def select_output(both):
            value = tf.transpose(
                tf.slice(
                    both, [
                        0, 0], [
                        self.in_size + 1, self.out_size]))
            valid = tf.transpose(
                tf.slice(both, [self.in_size + 1, 0], [self.in_size + 1, self.out_size]))
            pos = tf.cast(tf.where(tf.equal(valid, 1)), tf.int32)
            pos_reduced = tf.concat([tf.reshape(tf.range(0, self.out_size), [self.out_size, 1]), tf.reshape(
                tf.segment_min(pos[:, 1], pos[:, 0]), [self.out_size, 1])], 1)
            return tf.gather_nd(value, pos_reduced)

        layer_out = tf.map_fn(select_output, output_valid_both)
        return layer_out

    def cost(self):
        wsc = w_sum_cost(self.weight)
        l2 = l2_func(self.weight)
        return wsc,l2


class SCNN(object):
    def __init__(self, kernel_size=3, in_channel=1, out_channel=1, strides=1):
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.strides = strides
        self.kernel = SNNLayer(in_size=self.kernel_size * self.kernel_size * self.in_channel,
                                        out_size=self.out_channel)

    def forward(self, layer_in):
        input_size = tf.shape(layer_in)
        patches = tf.extract_image_patches(images=layer_in, ksizes=[1, self.kernel_size, self.kernel_size, 1],
                                           strides=[1, self.strides, self.strides, 1], rates=[1, 1, 1, 1],
                                           padding="SAME")
        patches_flatten = tf.reshape(patches,
                                     [input_size[0], -1, self.in_channel * self.kernel_size * self.kernel_size])
        patches_infpad = tf.where(tf.less(patches_flatten, 0.9),
                                  MAX_SPIKE_TIME * tf.ones_like(patches_flatten), patches_flatten)
        img_raw = tf.map_fn(self.kernel.forward, patches_infpad)
        img_reshaped = tf.reshape(img_raw, [input_size[0], tf.cast(tf.math.ceil(input_size[1] / self.strides),tf.int32), tf.cast(tf.math.ceil(input_size[2] / self.strides),tf.int32),
                                            self.out_channel])
        return img_reshaped


class SCNN_upsample(object):
    def __init__(self, kernel_size=3, in_channel=1, out_channel=1, strides=1):
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.strides = strides
        self.scnn = SCNN(self.kernel_size, self.in_channel, self.out_channel, 1)

    def upsample(self, in_image, strides):
        in_size = tf.shape(in_image)
        img_reshaped = tf.reshape(in_image, [in_size[0], in_size[1], 1, in_size[2], 1, in_size[3]])
        img_large1 = tf.concat((img_reshaped,MAX_SPIKE_TIME*tf.ones([in_size[0],in_size[1],1,in_size[2],strides-1,in_size[3]])),axis=4)
        img_large2 = tf.concat((img_large1,MAX_SPIKE_TIME*tf.ones([in_size[0],in_size[1],strides-1,in_size[2],strides,in_size[3]])),axis=2)
        #img_large = tf.tile(img_reshaped, [1, 1, strides, 1, strides, 1])
        img_large_reshaped = tf.reshape(img_large2, [in_size[0], in_size[1] * strides, in_size[2] * strides, in_size[3]])
        return img_large_reshaped

    def forward(self, layer_in):
        layer_in_upsample = self.upsample(layer_in, self.strides)
        return self.scnn.forward(layer_in_upsample)


def w_sum_cost(W):
    part1 = tf.subtract(1., tf.reduce_sum(W, 0))
    part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
    return tf.reduce_mean(part2)


def l2_func(W):
    s = tf.shape(W)
    W1 = tf.slice(W,[0,0],[s[0]-1,s[1]])
    w_sqr = tf.square(W1)
    return tf.reduce_mean(w_sqr)


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
                    z2, 1e-10, 1e10)), 1e-10, 1)))
    return loss
