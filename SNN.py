import tensorflow as tf

class Layer(object):
    def __init__(self,input,n_in,n_out,sess, W=None):
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        if W is None:
            W = tf.Variable(tf.random_normal([n_in,n_out], 1, .5, tf.float32))
        i = tf.Variable(0)
        sum_z = tf.Variable(tf.zeros([n_out,n_in],tf.float32))
        sum_W = tf.Variable(tf.zeros([n_out,n_in],tf.float32))
        c_zero = tf.constant(0.,tf.float32)
        c_one = tf.constant(1.,tf.float32)
        
        init = tf.global_variables_initializer()
        
        sess.run(init)
        
        self.W = W
        self.i = i
        self.sum_z = sum_z
        self.sum_W = sum_W
        self.c_zero = c_zero
        self.c_one = c_one
        def cal_out():
            r, ind = tf.nn.top_k(self.input, self.n_in)
            r_ind = tf.reverse(ind,[0])
            nx = tf.gather(self.input,r_ind)
            nW = tf.transpose(tf.gather(self.W,r_ind))
            nxW = tf.multiply(nx, nW)
            
            def body_z(i,z):
                z = tf.slice(
                    tf.concat(
                        [
                            tf.cast(
                                tf.reduce_sum(
                                    tf.slice(nxW,[0,0],[self.n_out,i+1]),
                                    1,True),
                                tf.float32),
                            z],
                        1),
                    [0,0],[self.n_out,self.n_in])
                return [i+1,z]
            
            def body_W(i,z):
                z = tf.slice(
                    tf.concat(
                        [
                            tf.cast(
                                tf.reduce_sum(
                                    tf.slice(nW,[0,0],[self.n_out,i+1]),
                                    1,True),
                                tf.float32),
                            z],
                        1),
                    [0,0],[self.n_out,self.n_in])
                return [i+1,z]

            def condition(i,z):
                return i<self.n_in
            
            r1,n_sum_z=tf.while_loop(condition, body_z, [self.i,self.sum_z])
            r2,n_sum_W=tf.while_loop(condition, body_W, [self.i,self.sum_W])
            f_sum_z = tf.reverse(n_sum_z,[1])
            f_sum_W = tf.reverse(n_sum_W,[1])
            
            out_all = tf.divide(f_sum_z, tf.subtract(f_sum_W,1))
            out_all_2 = tf.concat(
                [
                    out_all,
                    tf.transpose(
                        [
                            tf.tile(
                                [tf.divide(self.c_one,self.c_zero)],
                                [self.n_out])
                            ]
                        )
                    ]
                ,1)
            
            out_ok = tf.where(
                tf.logical_and(
                    tf.less(
                        tf.cast(
                            tf.tile(
                                [tf.concat([nx,[1]],0)],
                                [self.n_out,1]),
                            tf.float32)
                        ,
                        out_all_2),
                    tf.greater_equal(
                        tf.cast(
                            tf.tile(
                                [tf.slice(
                                    tf.concat(
                                        [nx,[tf.divide(self.c_one,self.c_zero),tf.divide(self.c_one,self.c_zero)]],0),
                                    [1],
                                    [self.n_in+1])
                                    ],
                                [self.n_out,1]),
                            tf.float32),
                        out_all_2)))

            out_idx = tf.transpose(tf.concat([[tf.range(0,self.n_out)],[tf.cast(tf.segment_min(out_ok[:, 1], out_ok[:, 0]),tf.int32)]],0))
            out = tf.gather_nd(out_all_2,out_idx)
            return out
        self.output = cal_out()
        
if __name__ == '__main__':
    sess = tf.Session()
    input = tf.placeholder(tf.float32)
    l1 = Layer(input,2,4,sess)
    l2 = Layer(l1.output,4,2,sess)
    print(sess.run(l2.output,{input:[1,3]}))
    print(sess.run(l1.W,{input:[1,2]}))
    sess.run(tf.assign(l1.W,[[.05,.1,.5,.7],[.04,.2,.6,.8]]))
    print(sess.run(l1.W,{input:[1,2]}))
    print(sess.run(l2.W,{input:[1,2]}))