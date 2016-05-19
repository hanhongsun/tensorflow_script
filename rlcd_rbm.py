from mt_reward import muscleTorqueScore
import tensorflow as tf
import Image
import numpy as np
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
from utils import tile_raster_images
from tensorflow.python.ops import control_flow_ops
import matplotlib.pyplot as plt

# size_x is the size of the visiable layer
# size_h is the size of the hidden layer
side_h = 10
side_x = 10
size_x = side_x * side_x
size_h = side_h * side_h
size_bt = 100 # batch size ## TODO support only batch == 1
k = 16
# helper function

def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def scalePosNegOne(vec):
    return tf.add(1.0, tf.mul(2.0, vec))

def scalePosNegOneInt(vec):
    return tf.add(1, tf.mul(2, vec))

# define parameters
b = tf.Variable(tf.random_uniform([size_h, 1], -0.05, 0.05))
W = tf.Variable(tf.random_uniform([size_x, size_h], -0.05, 0.05))
c = tf.Variable(tf.random_uniform([size_x, 1], -0.05, 0.05))
H0 = tf.Variable(tf.zeros([size_h, size_bt], tf.float32))
X1 = tf.Variable(tf.zeros([size_x, size_bt], tf.float32))
H1 = tf.Variable(tf.zeros([size_h, size_bt], tf.float32))

a = tf.placeholder(tf.float32)
coldness = tf.placeholder(tf.float32) # coldness is 1/Temperature

sc = tf.placeholder(tf.float32, [1, size_bt]) # place holder for returned score
# define the Simulated annealing sampling graph
# cold_target = tf.placeholder(tf.float32)
# the const number for this
norm_const = tf.Variable([20.00])

an_step = tf.constant(0.2)

x_o = sample(tf.ones([size_x, size_bt]) * 0.5)
h_o = sample(tf.sigmoid(tf.matmul(tf.transpose(W)*0, x_o) + tf.tile(b*0, [1, size_bt])))

def simAnnealingGibbs(xx, hh, temp_inv):
    xk = sample(tf.sigmoid(tf.matmul(W*temp_inv, hh) + tf.tile(c*temp_inv, [1, size_bt])))
    hk = sample(tf.sigmoid(tf.matmul(tf.transpose(W*temp_inv), xk) + tf.tile(b*temp_inv, [1, size_bt])))
    return xk, hk, temp_inv + an_step

def isColdEnough(xx, hh, temp_inv):
    return temp_inv < 1.0

[x0, h0, _] = control_flow_ops.While(isColdEnough, simAnnealingGibbs, [x_o, h_o, coldness], 1, False)

x1 = sample(tf.sigmoid(tf.matmul(W, h0) + tf.tile(c, [1, size_bt])))
h1 = sample(tf.sigmoid(tf.matmul(tf.transpose(W), x1) + tf.tile(b, [1, size_bt])))

sample_data = [H0.assign(h0), X1.assign(x1), H1.assign(h1)]

def logMargX(x, h, W, c):
    prob_all1 = tf.matmul(W, h) + tf.tile(c, [1, size_bt])
    # log_matrix has the DIM[0] the position of batch in h and DIM[1] the position of batch in x
    log_matrix = tf.reduce_sum(tf.tile(tf.expand_dims(prob_all1, -1), [1, 1, size_bt]) * \
        tf.transpose(tf.tile(tf.expand_dims(scalePosNegOne(x), -1), [1, 1, size_bt]), [0, 2, 1]), 1)
    return tf.log(tf.reduce_mean(tf.sigmoid(log_matrix), 0, True))

# define the update rule
updt_value = sc - logMargX(X1, H0, W, c) - tf.tile(tf.expand_dims(norm_const, -1), [1, size_bt])
update_value = tf.minimum(tf.maximum(tf.exp(updt_value) - 2.8, -2.8), 100)
update_value_norm = tf.minimum(tf.maximum(updt_value, -2.8), 100)

norm_const_ = tf.mul(tf.reduce_mean(update_value_norm, 1), 2*a)

# x2 = sample(tf.sigmoid(tf.matmul(W, H1) + tf.tile(c, [1, size_bt])))
# h2 = sample(tf.sigmoid(tf.matmul(tf.transpose(W), x2) + tf.tile(b, [1, size_bt])))

def rbmGibbs(xx, hh, count, k):
    xk = sampleInt(tf.sigmoid(tf.matmul(W, hh) + tf.tile(c, [1, size_bt])))
    hk = sampleInt(tf.sigmoid(tf.matmul(tf.transpose(W), xk) + tf.tile(b, [1, size_bt])))
    # assh_in1 = h_in.assign(hk)
    return xk, hk, count+1, k

def lessThanK(xk, hk, count, k):
    return count <= k

# tf.convert_to_tensor(X1, name="X1") tf.convert_to_tensor(H1, name="H1")
[x2, h2, _, _] = control_flow_ops.While(lessThanK, rbmGibbs, [X1, H1, tf.convert_to_tensor(1), tf.convert_to_tensor(k)], 1, False)


debug_1 = tf.tile(tf.reshape(update_value, [size_bt, 1, 1]), [1, size_x, size_h])
debug_2 = tf.batch_matmul(tf.expand_dims(tf.transpose(X1), 2), tf.expand_dims(tf.transpose(H1), 1))
debug_3 = tf.batch_matmul(tf.expand_dims(tf.transpose(x2), 2), tf.expand_dims(tf.transpose(h2), 1))
debug_4 = tf.tile(tf.reshape(update_value, [size_bt, 1, 1]), [1, size_x, size_h]) * tf.batch_matmul(tf.expand_dims(tf.transpose(X1), 2), tf.expand_dims(tf.transpose(H1), 1))


W_ = a/float(size_bt) * tf.reduce_mean(tf.sub(tf.batch_matmul(tf.expand_dims(tf.transpose(X1), 2), tf.expand_dims(tf.transpose(H1), 1)),\
                                              tf.batch_matmul(tf.expand_dims(tf.transpose(x2), 2), tf.expand_dims(tf.transpose(h2), 1)))\
                                       * tf.tile(tf.reshape(update_value, [size_bt, 1, 1]), [1, size_x, size_h]), 0)
b_ = a * tf.reduce_mean(tf.mul(tf.sub(H1, h2), tf.tile(update_value, [size_h, 1])), 1, True)
c_ = a * tf.reduce_mean(tf.mul(tf.sub(X1, x2), tf.tile(update_value, [size_x, 1])), 1, True)

# norm_const
# TODO not done, add the self engergy function estimation
updt = [W.assign_add(W_), b.assign_add(b_), c.assign_add(c_), norm_const.assign_add(norm_const_)]

# run session

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

norm_hist = []
sample_score_hist = []
# loop with batch
for i in range(1, 10002):
    alpha = min(0.05, 100.0/float(i))
    sess.run(sample_data, feed_dict={coldness: 0.0})
    x1_ = sess.run(X1)
    score = 10.0 * muscleTorqueScore(size_x, side_x, x1_)
    sample_score_hist.extend(score.tolist())
    # print sample_score_hist
    # print "shape of W_", sess.run(tf.shape(W_), feed_dict={ sc: score, a: alpha, coldness: 0.0}), sess.run(tf.shape(W))
    # print "shape of b_", sess.run(tf.shape(b_), feed_dict={ sc: score, a: alpha, coldness: 0.0}), sess.run(tf.shape(b))
    # print "shape of c_", sess.run(tf.shape(c_), feed_dict={ sc: score, a: alpha, coldness: 0.0}), sess.run(tf.shape(c))
    # print "shape of norm_const_", sess.run(tf.shape(norm_const_), feed_dict={ sc: score, a: alpha, coldness: 0.0}), sess.run(tf.shape(norm_const))
    # print "logMargX", sess.run(logMargX(x1, h1, W, c), feed_dict={ sc: score, a: alpha, coldness: 0.00})
    # print "shape of updt_value", sess.run(tf.shape(updt_value), feed_dict={ sc: score, a: alpha, coldness: 0.0}), "value ", sess.run(updt_value, feed_dict={ sc: score, a: alpha, coldness: 0.0})
    sess.run(updt, feed_dict={ sc: score, a: alpha})
    print i, ' mean score ', np.mean(score), ' max score ', np.max(score), ' step size ', alpha, ' norm_const ', sess.run(norm_const)
    # vidualization
    if i % 500 == 1:
        image = Image.fromarray(tile_raster_images(sess.run(W).T,
                                                   img_shape=(side_x, side_x),
                                                   tile_shape=(side_h, side_h),
                                                   tile_spacing=(2, 2)))
        image.show()
        image = Image.fromarray(tile_raster_images(sess.run(X1).T,
                                                   img_shape=(side_x, side_x),
                                                   tile_shape=(side_h, side_h),
                                                   tile_spacing=(2, 2)))
        image.show()
        image = Image.fromarray(tile_raster_images(sess.run(c).T,
                                                   img_shape=(side_x, side_x),
                                                   tile_shape=(2, 2),
                                                   tile_spacing=(2, 2)))
        image.show()
        print 'norm_const ', sess.run(norm_const).T
        print 'W ', sess.run(W).T
        print 'c ', sess.run(c).T
        # print 'tiled 3d scalar', sess.run(debug_1, feed_dict={sc: score, a: alpha}).T
        # print 'W update pos side 3d', sess.run(debug_2, feed_dict={sc: score, a: alpha}).T
        # print 'W update neg side re-sample 3d', sess.run(debug_3, feed_dict={sc: score, a: alpha}).T
        # print 'W update neg side weighted 3d', sess.run(debug_4, feed_dict={sc: score, a: alpha}).T

        # plt.plot(sample_score_hist)
        # print 'c ', sess.run(c).T
        # print 'b ', sess.run(b).T
        # print 'W ', sess.run(W).T
        
plt.plot(sample_score_hist)
plt.show()




