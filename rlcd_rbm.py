from mt_reward import muscleTorqueScore
import tensorflow as tf
import Image
import numpy as np
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
from utils import tile_raster_images
from tensorflow.python.ops import control_flow_ops

# size_x is the size of the visiable layer
# size_h is the size of the hidden layer
side_h = 20
side_x = 100
size_x = side_x * side_x
size_h = side_h * side_h
size_bt = 1 # batch size

# helper function

def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# define parameters
b = tf.Variable(tf.random_uniform([size_h, 1], -0.05, 0.05))
W = tf.Variable(tf.random_uniform([size_x, size_h], -0.05, 0.05))
c = tf.Variable(tf.random_uniform([size_x, 1], -0.05, 0.05))
x = tf.placeholder(tf.float32, [size_x, size_bt])
# a = tf.placeholder(tf.float32)

# define the Simulated annealing sampling graph
cold = tf.Variable(tf.ones([1])*0.05) # cold is 1/Temperature
an_step = tf.constant(0.05)

h = sample(tf.sigmoid(tf.matmul(tf.transpose(W * cold), x) + tf.tile(b * cold, [1, size_bt])))

def rbmGibbs(xx, hh, count, cold):
    xk = sampleInt(tf.sigmoid(tf.matmul(W * cold, hh) + tf.tile(c * cold, [1, size_bt])))
    hk = sampleInt(tf.sigmoid(tf.matmul(tf.transpose(W * cold), xk) + tf.tile(b * cold, [1, size_bt])))
    # assh_in1 = h_in.assign(hk)
    return xk, hk, count + 1, cold + an_step

def less_than_k(xx, hk, count, k):
    return count <= k


# define the update rule


muscleTorqueScore(size_x, side_x, x_test)


