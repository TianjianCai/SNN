import tensorflow as tf

input_size = 5
output_size = 4
x = tf.Variable([1.,7,4,2,3],tf.float32)
W = tf.Variable([[.1,.3,.5,.7],[.2,.4,.6,.8],[.3,.5,.7,.9],[.4,.6,.8,.2],[.5,.7,.8,.3]],tf.float32)
i = tf.Variable(0)
sum_z = tf.Variable(tf.zeros([output_size,input_size],tf.float32))
sum_W = tf.Variable(tf.zeros([output_size,input_size],tf.float32))
c_zero = tf.constant(0.,tf.float32)
c_one = tf.constant(1.,tf.float32)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

r, ind = tf.nn.top_k(x, input_size)
r_ind = tf.reverse(ind,[0])
nx = tf.gather(x,r_ind)
nW = tf.transpose(tf.gather(W,r_ind))
nxW = tf.multiply(nx, nW)

def body_z(i,z):
    z = tf.slice(tf.concat([tf.cast(tf.reduce_sum(tf.slice(nxW,[0,0],[output_size,i+1]),1,True),tf.float32),z],1),[0,0],[output_size,input_size])
    return [i+1,z]

def body_W(i,z):
    z = tf.slice(tf.concat([tf.cast(tf.reduce_sum(tf.slice(nW,[0,0],[output_size,i+1]),1,True),tf.float32),z],1),[0,0],[output_size,input_size])
    return [i+1,z]

def condition(i,z):
    return i<input_size

#with tf.control_dependencies(tf.while_loop(condition, body, [i])):
#    print(sess.run(i))
r1,n_sum_z=tf.while_loop(condition, body_z, [i,sum_z])
r2,n_sum_W=tf.while_loop(condition, body_W, [i,sum_W])
f_sum_z = tf.reverse(n_sum_z,[1])
f_sum_W = tf.reverse(n_sum_W,[1])

out_all = tf.divide(f_sum_z, tf.subtract(f_sum_W,1))
out_all_2 = tf.concat([out_all,tf.transpose([tf.tile([tf.divide(c_one,c_zero)],[output_size])])],1)

out_ok = tf.where(
    tf.logical_and(
        tf.less(
            tf.cast(
                tf.tile(
                    [tf.concat([nx,[1]],0)],
                    [output_size,1]),
                tf.float32)
            ,
            out_all_2),
        tf.greater(
            tf.cast(
                tf.tile(
                    [tf.slice(
                        tf.concat(
                            [nx,[tf.divide(c_one,c_zero),tf.divide(c_one,c_zero)]],0),
                        [1],
                        [input_size+1])
                    ],
                    [output_size,1]),
                tf.float32),
            out_all_2)))

out_idx = tf.transpose(tf.concat([[tf.range(0,output_size)],[tf.cast(tf.segment_min(out_ok[:, 1], out_ok[:, 0]),tf.int32)]],0))


out = tf.gather_nd(out_all,out_idx)

print(sess.run(nx))
#print(sess.run(f_sum_z))
#print(sess.run(f_sum_W))
print(sess.run(out_all_2))
#print(sess.run(out_ok))
print(sess.run(out_idx))
print(sess.run(out))
