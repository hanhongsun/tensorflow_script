import tensorflow as tf
import Image
import numpy as np
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
from utils import tile_raster_images
from tensorflow.python.ops import control_flow_ops

# size_x is the size of the visiable layer
# size_h is the size of the hidden layer
side_h = 10
size_x = 28*28
size_h = side_h * side_h
size_bt = 100 # batch size

k = tf.constant(1)

#### we do the first test on the minst data again

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
    mnist.test.images, mnist.test.labels


# helper function

def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

# variables and place holder

b = tf.Variable(tf.random_uniform([size_h, 1], -0.005, 0.005))
W = tf.Variable(tf.random_uniform([size_x, size_h], -0.005, 0.005))
c = tf.Variable(tf.random_uniform([size_x, 1], -0.005, 0.005))
x = tf.placeholder(tf.float32, [size_x, size_bt])
a = tf.placeholder(tf.float32)

# define graph/algorithm

# sample h x1 h1 ..
h = sample(tf.sigmoid(tf.matmul(tf.transpose(W), x) + tf.tile(b, [1, size_bt])))

# CD-k 
# we use tf.while_loop to achieve the multiple (k - 1) gibbs sampling  

# set up tf.while_loop()

def rbmGibbs(xx, hh, count, k):
    xk = sampleInt(tf.sigmoid(tf.matmul(W, hh) + tf.tile(c, [1, size_bt])))
    hk = sampleInt(tf.sigmoid(tf.matmul(tf.transpose(W), xk) + tf.tile(b, [1, size_bt])))
    # assh_in1 = h_in.assign(hk)
    return xk, hk, count+1, k

def less_than_k(xx, hk, count, k):
    return count <= k

ct = tf.constant(1)

[xk1, hk1, _, _] = control_flow_ops.While(less_than_k, rbmGibbs, [x, h, ct, k], 1, False)

# update rule
[W_, b_, c_] = [tf.mul(a/float(size_bt), tf.sub(tf.matmul(x, tf.transpose(h)), tf.matmul(xk1, tf.transpose(hk1)))),\
        tf.mul(a/float(size_bt), tf.reduce_sum(tf.sub(h, hk1), 1, True)),\
        tf.mul(a/float(size_bt), tf.reduce_sum(tf.sub(x, xk1), 1, True))]

# wrap session
updt = [W.assign_add(W_), b.assign_add(b_), c.assign_add(c_)]

# stop gradient to save time and mem
tf.stop_gradient(h)
tf.stop_gradient(xk1)
tf.stop_gradient(hk1)
tf.stop_gradient(W_)
tf.stop_gradient(b_)
tf.stop_gradient(c_)

# run session

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# loop with batch
for i in range(1, 10002):
    tr_x, tr_y  = mnist.train.next_batch(size_bt)
    tr_x = np.transpose(tr_x)
    tr_y = np.transpose(tr_y)
    alpha = min(0.05, 100/float(i))
    sess.run(updt, feed_dict={x: tr_x, a: alpha})
    print i, ' step size ', alpha
    # vidualization
    if i % 5000 == 1:
        image = Image.fromarray(tile_raster_images(sess.run(W).T,
                                                   img_shape=(28, 28),
                                                   tile_shape=(side_h, side_h),
                                                   tile_spacing=(2, 2)))
        image.show()
        print 'c ', sess.run(c).T
        print 'b ', sess.run(b).T
        print 'W ', sess.run(W).T
        print 'x ', np.transpose(tr_x)
        print 'h ', sess.run(h, feed_dict={x: tr_x}).T
        # print 'x1 ', sess.run(x1, feed_dict={x: tr_x}).T
        # print 'h1 ', sess.run(h1, feed_dict={x: tr_x}).T        
        imagex = Image.fromarray(tile_raster_images(np.transpose(tr_x),
                                                   img_shape=(28, 28),
                                                   tile_shape=(10, 10),
                                                   tile_spacing=(2, 2)))
        imagex.show()
        imagexk = Image.fromarray(tile_raster_images(sess.run(xk1, feed_dict={x: tr_x}).T,
                                                   img_shape=(28, 28),
                                                   tile_shape=(10, 10),
                                                   tile_spacing=(2, 2)))
        imagexk.show()


