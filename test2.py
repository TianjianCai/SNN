import tensorflow as tf
import math

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
            output = tf.where(out>100,tf.multiply(tf.ones_like(out),100),out)
            return output
        self.output = cal_out()
        
if __name__ == '__main__':
    W1 = [[0.34282106, 1.4719542,  1.7410532,  0.44637924],[1.4091486,  1.0784113,  1.8950409,  1.6238755 ]]
    W2 = [[0.22391653, 0.7389356 ],[1.2561495,  0.472723  ],[1.0924255,  1.8378254 ],[1.4406638,  0.690428  ]]
    
    sess = tf.Session()
    input = tf.placeholder(tf.float32)
    
    l1 = Layer(input,2,4,sess)
    l2 = Layer(l1.output,4,2,sess)
    
    sess.run(tf.assign(l1.W,W1))
    sess.run(tf.assign(l2.W,W2))
    
    test_input = [[math.exp(0),math.exp(0)],[math.exp(0),math.exp(1)],[math.exp(1),math.exp(0)],[math.exp(1),math.exp(1)]]
    
    i=0
    while i<4:
        print('input is:',end=' ')
        print(test_input[i])
        print(sess.run(l2.output,{input:test_input[i]}))
        i=i+1