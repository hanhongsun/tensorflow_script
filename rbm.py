import tensorflow as tf
import Image
import numpy as np
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
# sys.path.append('/home/hanhong/Projects/python27/DeepLearningToolbox/util')
from utils import tile_raster_images

# size_x is the size of the visiable layer
# size_h is the size of the hidden layer
side_h = 10
size_x = 28*28
size_h = side_h * side_h
size_bt = 200

#### we do the first test on the minst data again

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels,\
    mnist.test.images, mnist.test.labels


# set up graph and algorithm

def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

b = tf.Variable(tf.random_uniform([size_h, 1], -0.005, 0.005))
W = tf.Variable(tf.random_uniform([size_x, size_h], -0.005, 0.005))
c = tf.Variable(tf.random_uniform([size_x, 1], -0.005, 0.005))
x = tf.placeholder(tf.float32, [size_x, size_bt])
a = tf.placeholder(tf.float32)


# h = tf.Variable(tf.zeros(tf.float32, [size_h, size_bt]))

# b_update = tf.Variable(tf.zeros([size_h, 1]))
# W_update = tf.Variable(tf.zeros([size_x, size_h]))
# c_update = tf.Variable(tf.zeros([size_x, 1])) 

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# print 'x ', sess.run(tf.size(x)) , sess.run(tf.shape(x))
print 'W ', sess.run(tf.size(W)), sess.run(tf.shape(W)),\
  'b ', sess.run(tf.size(b)), sess.run(tf.shape(b)),\
  'c ', sess.run(tf.size(c)), sess.run(tf.shape(c))

h = sample(tf.sigmoid(tf.matmul(tf.transpose(W), x) + tf.tile(b, [1, size_bt])))
x1 = sample(tf.sigmoid(tf.matmul(W, h) + tf.tile(c, [1, size_bt])))
h1 = tf.sigmoid(tf.matmul(tf.transpose(W), x1) + tf.tile(b, [1, size_bt]))

# set up the session
  # this session has multiple output to b_update W_update c_update at one round

# def sess_fun():
# c_ = tf.reduce_sum(x - x1, 1, True)
# W_ = tf.matmul(x, tf.transpose(h)) - tf.matmul(x1, tf.transpose(h1))
# b_ = tf.reduce_sum(h - h1, 1, True)
[W_, b_, c_] = [tf.mul(a/float(size_bt), tf.sub(tf.matmul(x, tf.transpose(h)), tf.matmul(x1, tf.transpose(h1)))),\
        tf.mul(a/float(size_bt), tf.reduce_sum(tf.sub(h, h1), 1, True)),\
        tf.mul(a/float(size_bt), tf.reduce_sum(tf.sub(x, x1), 1, True))]

tf.stop_gradient(h)
tf.stop_gradient(x1)
tf.stop_gradient(h1)
tf.stop_gradient(W_)
tf.stop_gradient(b_)
tf.stop_gradient(c_)

# run session
# W_ = tf.placeholder(tf.float32, [size_x, size_h])
# b_ = tf.placeholder(tf.float32, [size_h, 1])
# c_ = tf.placeholder(tf.float32, [size_x, 1])
updt = [W.assign(W + W_), b.assign(b + b_), c.assign(c + c_)]

sess.run(init)


#print 'tr_x batch ', size_bt, ' shape ', sess.run(tf.shape(tr_x))

for i in range(1, 10002):
    tr_x, tr_y  = mnist.train.next_batch(size_bt)
    tr_x = np.transpose(tr_x)
    tr_y = np.transpose(tr_y)
    alpha = min(0.005, 10/float(i))
    # b_update = sess.run(b_, feed_dict={x: tr_x})
    # W_update = sess.run(W_, feed_dict={x: tr_x})
    # c_update = sess.run(c_, feed_dict={x: tr_x})
    # [W_update, b_update, c_update] = sess.run(wbc_, feed_dict={x: tr_x, a: alpha})
    # sess.run(apdt, feed_dict = {W_: W_update, b_: b_update, c_: c_update})
    # [W_update, b_update, c_update] = sess.run(wbc_, feed_dict={x: tr_x, a: alpha})
    sess.run(updt, feed_dict={x: tr_x, a: alpha})
    print i, ' step size ', alpha
    # vidualization
    if i % 500 == 1:
        image = Image.fromarray(tile_raster_images(sess.run(W).T, #W_update.T, #np.transpose(tr_x), #
                                                   img_shape=(28, 28),
                                                   tile_shape=(side_h, side_h),
                                                   tile_spacing=(2, 2)
                                                   )
                                )
        image.show()
        # print 'c shape ', sess.run(tf.shape(c)), 'c_update shape ', sess.run(tf.shape(c_update))
        print 'c ', sess.run(c).T
        # print 'c_update ', c_update.T
        # print 'b shape ', sess.run(tf.shape(b)), 'b_update shape ', sess.run(tf.shape(b_update))
        print 'b ', sess.run(b).T
        # print 'b_update ', b_update.T
        # print 'W shape ', sess.run(tf.shape(W)), 'W_update shape ', sess.run(tf.shape(W_update))
        # print 'W_update ', W_update.T
        print 'W ', sess.run(W).T
        print 'x ', np.transpose(tr_x)
        print 'h ', sess.run(h, feed_dict={x: tr_x}).T
        print 'x1 ', sess.run(x1, feed_dict={x: tr_x}).T
        print 'h1 ', sess.run(h1, feed_dict={x: tr_x}).T        
        # imageh = Image.fromarray(tile_raster_images(sess.run(x1, feed_dict={x: tr_x}).T, #sess.run(sample(x), feed_dict={x: tr_x}).T,#np.transpose(tr_x), #c_update.T,#
        #                                            img_shape=(28, 28),
        #                                            tile_shape=(9, 9),
        #                                            tile_spacing=(2, 2)
        #                                            )
        #                         )
        # imageh.show()

test_ph = tf.placeholder(tf.float32,[2, 2]);
print 'test sample 1 ', sess.run(sample(test_ph), feed_dict={test_ph: [[1, 1], [1, 1]]})
print 'test sample 0.5 ', sess.run(sample(test_ph), feed_dict={test_ph: [[0.5, 0.5], [0.5, 0.5]]})
print 'test sample 0.2 ', sess.run(sample(test_ph), feed_dict={test_ph: [[0.2, 0.2], [0.2, 0.2]]})
print 'test sample 0 ', sess.run(sample(test_ph), feed_dict={test_ph: [[0.0, 0.0], [0.0, 0.0]]})
# set up the algorithm



# set up visualization



# trainning
