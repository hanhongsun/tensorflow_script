# this file is a backup file for continue using the simulated annealing method with temperature.

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
side_h = 7
side_x = 10
size_x = side_x * side_x
size_h = side_h * side_h
size_bt = 1 # batch size ## TODO support only batch == 1

# helper function

def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# define parameters
b = tf.Variable(tf.random_uniform([size_h, 1], -0.05, 0.05))
W = tf.Variable(tf.random_uniform([size_x, size_h], -0.05, 0.05))
c = tf.Variable(tf.random_uniform([size_x, 1], -0.05, 0.05))
# x = tf.placeholder(tf.float32, [size_x, size_bt])
a = tf.placeholder(tf.float32)
coldness = tf.placeholder(tf.float32) # coldness is 1/Temperature

sc = tf.placeholder(tf.float32, [size_bt, 1])
# define the Simulated annealing sampling graph
cold_target = tf.placeholder(tf.float32)
# the const number for this
norm_const = tf.Variable(100.00)

an_step = tf.constant(0.2)

x = sample(tf.ones([size_x, size_bt]) * 0.5)
h = sample(tf.sigmoid(tf.matmul(tf.transpose(tf.mul(W, coldness)), x) + tf.tile(tf.mul(b, coldness), [1, size_bt])))

def simAnnealingGibbs(xx, hh, temp_inv):
    xk = sample(tf.sigmoid(tf.matmul(tf.mul(W, temp_inv), hh) + tf.tile(tf.mul(c, temp_inv), [1, size_bt])))
    hk = sample(tf.sigmoid(tf.matmul(tf.transpose(tf.mul(W, temp_inv)), xk) + tf.tile(tf.mul(b, temp_inv), [1, size_bt])))
    return xk, hk, temp_inv + an_step

def isColdEnough(xx, hh, temp_inv):
    return temp_inv < cold_target

[x1, h1, _] = control_flow_ops.While(isColdEnough, simAnnealingGibbs, [x, h, coldness], 1, False)
x2 = sample(tf.sigmoid(tf.matmul(tf.mul(W, cold_target), h1) + tf.tile(tf.mul(c, cold_target), [1, size_bt])))
h2 = sample(tf.sigmoid(tf.matmul(tf.transpose(tf.mul(W, cold_target)), x2) + tf.tile(tf.mul(b, cold_target), [1, size_bt])))

# define the update rule

[W_, b_, c_] = [tf.mul(a/float(size_bt), tf.sub(tf.matmul(x1, tf.transpose(h1)), tf.matmul(x2, tf.transpose(h2)))),\
        tf.mul(a/float(size_bt), tf.reduce_sum(tf.sub(h1, h2), 1, True)),\
        tf.mul(a/float(size_bt), tf.reduce_sum(tf.sub(x1, x2), 1, True))]

neg_self_engergy = tf.mul(cold_target, (tf.matmul(tf.transpose(x1), tf.matmul(W, h1))\
                                  + tf.matmul(tf.transpose(b), h1)\
                                  + tf.matmul(tf.transpose(c), x1)))
updt_value = tf.exp(tf.squeeze(cold_target * sc - neg_self_engergy - norm_const))

# norm_const
# TODO not done, add the self engergy function estimation
updt = [W.assign_add(tf.mul(W_, updt_value)),\
        b.assign_add(tf.mul(b_, updt_value)),\
        c.assign_add(tf.mul(c_, updt_value)),\
        norm_const.assign_add(tf.mul(updt_value - 1.0, a))]

# run session

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

norm_hist = []
sample_score_hist = []
# loop with batch
for i in range(1, 10002):
    alpha = min(0.05, 100.0/float(i))
    x1_ = sess.run(x1, feed_dict={cold_target: 5.0, coldness: 0.00})
    score = muscleTorqueScore(size_x, side_x, x1_)
    sample_score_hist.append(np.asscalar(score))
    # print sample_score_hist
    # print "shape of W_", sess.run(tf.shape(W_), feed_dict={ sc: score, a: alpha, cold_target: 1.0, coldness: 0.0}), sess.run(tf.shape(W))
    # print "shape of b_", sess.run(tf.shape(b_), feed_dict={ sc: score, a: alpha, cold_target: 1.0, coldness: 0.0}), sess.run(tf.shape(b))
    # print "shape of c_", sess.run(tf.shape(c_), feed_dict={ sc: score, a: alpha, cold_target: 1.0, coldness: 0.0}), sess.run(tf.shape(c))
    print "neg_self_engergy", sess.run(neg_self_engergy, feed_dict={ sc: score, a: alpha, cold_target: 5.0, coldness: 0.00})
    # print "shape of updt_value", sess.run(tf.shape(updt_value), feed_dict={ sc: score, a: alpha, cold_target: 1.0, coldness: 0.0}), "value ", sess.run(updt_value, feed_dict={ sc: score, a: alpha, cold_target: 1.0, coldness: 0.0})
    sess.run(updt, feed_dict={ sc: score, a: alpha, cold_target: 5.0, coldness: 0.0})
    print i, ' step size ', alpha
    # vidualization
    if i % 200 == 1:
        image = Image.fromarray(tile_raster_images(sess.run(W).T,
                                                   img_shape=(side_x, side_x),
                                                   tile_shape=(side_h, side_h),
                                                   tile_spacing=(2, 2)))
        image.show()
        image = Image.fromarray(tile_raster_images(sess.run(x1, feed_dict={ sc: score, a: alpha, cold_target: 5.0, coldness: 0.0}).T,
                                                   img_shape=(side_x, side_x),
                                                   tile_shape=(side_h, side_h),
                                                   tile_spacing=(2, 2)))
        image.show()
        print 'norm_const ', sess.run(norm_const).T
        plt.plot(sample_score_hist)
        plt.show() 
        # print 'c ', sess.run(c).T
        # print 'b ', sess.run(b).T
        # print 'W ', sess.run(W).T
