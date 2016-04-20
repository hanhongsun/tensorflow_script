import tensorflow as tf
import Image
import numpy as np
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
# sys.path.append('/home/hanhong/Projects/python27/DeepLearningToolbox/util')
from utils import tile_raster_images

# size_x is the size of the visiable layer
# size_h is the size of the hidden layer
size_x = 784
size_h = 49
size_bt = 50

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

b_update = tf.Variable(tf.zeros([size_h, 1]))
W_update = tf.Variable(tf.zeros([size_x, size_h]))
c_update = tf.Variable(tf.zeros([size_x, 1])) 

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
wbc_ = [tf.mul(a/float(size_bt), tf.matmul(x, tf.transpose(h)) - tf.matmul(x1, tf.transpose(h1))),\
        tf.mul(a/float(size_bt), tf.reduce_sum(h - h1, 1, True)),\
        tf.mul(a/float(size_bt), tf.reduce_sum(x - x1, 1, True))]

tf.stop_gradient(h)
tf.stop_gradient(x1)
tf.stop_gradient(h1)
tf.stop_gradient(wbc_[0])
tf.stop_gradient(wbc_[1])
tf.stop_gradient(wbc_[2])

# run session

sess.run(init)


#print 'tr_x batch ', size_bt, ' shape ', sess.run(tf.shape(tr_x))

for i in range(1, 5001):
    tr_x, tr_y  = mnist.train.next_batch(size_bt)
    tr_x = np.transpose(tr_x)
    tr_y = np.transpose(tr_y)
    alpha = min(0.002, 10/float(i))
    # b_update = sess.run(b_, feed_dict={x: tr_x})
    # W_update = sess.run(W_, feed_dict={x: tr_x})
    # c_update = sess.run(c_, feed_dict={x: tr_x})
    [W_update, b_update, c_update] = sess.run(wbc_, feed_dict={x: tr_x, a: alpha})
    b += b_update
    W += W_update
    c += c_update
    print i, ' step size ', alpha
    # vidualization
    if i % 500 == 1:
        image = Image.fromarray(tile_raster_images(sess.run(W).T, #W_update.T, #np.transpose(tr_x), #
                                                   img_shape=(28, 28),
                                                   tile_shape=(7, 8),
                                                   tile_spacing=(2, 2)
                                                   )
                                )
        image.show()
        print 'c shape ', sess.run(tf.shape(c)), 'c_update shape ', sess.run(tf.shape(c_update))
        # print 'c ', sess.run(c).T
        print 'b shape ', sess.run(tf.shape(b)), 'b_update shape ', sess.run(tf.shape(b_update))
        # print 'b ', sess.run(b).T
        print 'W shape ', sess.run(tf.shape(W)), 'W_update shape ', sess.run(tf.shape(W_update))
        # print 'W ', sess.run(W).T
                
        imageh = Image.fromarray(tile_raster_images(sess.run(sample(x), feed_dict={x: tr_x}).T,#np.transpose(tr_x), #c_update.T,#
                                                   img_shape=(28, 28),
                                                   tile_shape=(1, 1),
                                                   tile_spacing=(2, 2)
                                                   )
                                )
        # imageh.show()


# set up the algorithm



# set up visualization



# trainning
