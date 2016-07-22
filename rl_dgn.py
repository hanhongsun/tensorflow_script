# in this file, we are trying to program a naive reinforcement learning deep generated model
# Author Di Sun

# TODO train the normal_const
# TODO set the train rule
# TODO tf.stop_gradient(tensor)

# score function
from mt_reward import muscleTorqueScore, muscleDirectTorqueScore

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
import math, copy
#??? how to use tensorboard to visualize

# display choice 1
import matplotlib.pyplot as plt
# display choice 0
import Image
import sys
sys.path.append('/home/hanhong/Projects/python27/DeepLearningTutorials/code/')
from utils import tile_raster_images


# we will try to use tensorboard for display

# we will try to write the python class with tensorflow in order to increase the readablilty


np.random.seed(0)
tf.set_random_seed(0)

def scalePosNegOne(vec):
    return tf.add(-1.0, tf.mul(2.0, tf.to_float(vec)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

# Each sample from top level will be used to estimate a probability of generating each of the 
# sample from next level. Sum based on the weight on the previous level.
def compute_importance_weight(sp_h, sp, spw_h, w, b_h, batch_size):
    # we use 3d matrices to represent both the tiled input sample and tiled output sample,
    # then we know for each entry of 3d matrix of input sample, the log prob that it would samples a one
    # would be dependent to the parameters.
    # The -log_prob_of_one would be the log_prob for 0
    # The sum over the vector of top sample will be the prob for one top sample generating one bottom.

    # TODO double check
    log_prob_of_all_one = tf.matmul(w, sp_h) + tf.tile(b_h, [1, batch_size])
    # supporting only sigmoid function right now
    log_matrix = tf.reduce_sum(tf.tile(tf.expand_dims(log_prob_of_all_one, -1), [1, 1, batch_size]) * \
        tf.transpose(tf.tile(tf.expand_dims(scalePosNegOne(sp), -1), [1, 1, batch_size]), [0, 2, 1]), 0)
    # sigmoid then unnormalized_weight 
    weighted_sigmoid_matrix = tf.sigmoid(log_matrix) * tf.tile(tf.transpose(spw_h), [1, batch_size])
    # weighted mean readuce_mean
    return tf.clip_by_value(tf.reduce_mean(weighted_sigmoid_matrix, 0, True),1e-10, 1e35)


# hope tensor will take this overload form.
def derivative_sigmoid_x(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

# input the error: e
# param weight: W, bias: b
# data prob1: p, sample from p: s, sample from higher level: s_h
# control arguement, size low: size_l, last_layer == True then only b_ update
# return upper level of error: e_1, additive update of W: W_, additive update of b: b_
def back_propagate_one_layer(W, p, s, s_h, e, last_layer):
    # e must be the size of (size_l, size_bt)
    delta = tf.mul(tf.mul(e, scalePosNegOne(s)), derivative_sigmoid_x(p))
    # we assert when e[i] != 0 then s[i] == 1 because s[i] was used to compute e[i]
    b_ = tf.reduce_mean(delta, 1, True)
    if last_layer:
        W_ = None
        e_h = None
    else:
        # we need to do a cross product of delta and s_h  W_bt_ and mean on the batch to get W_
        # size(W) = (size_l, size_h), size(delta) = (size_l, size_bt), size(s_h) = (size_h, size_bt)
        # delta ==> delta_t with size (size_bt, size_l, 1)
        # s_h ==> s_h_t with size (size_bt, 1, size_h)
        # W_bt_t_ (size_bt, size_l, sizeh) = tf.batch_matmul()
        delta_t = tf.expand_dims(tf.transpose(delta), 2)
        s_h_t = tf.expand_dims(tf.transpose(s_h), 1)
        W_bt_t_ = tf.batch_matmul(delta_t, s_h_t)
        W_ = tf.reduce_mean(W_bt_t_, 0)
        e_h = tf.matmul(tf.transpose(W), delta)
    return  b_, W_, e_h

class ReinforcementLearningDGN(object):
    # Define the network
    # Define the parameter variables
    # Define the value variables
    # Initialize the network
    # Generate by sample Then save to value variables
    # Evaluate marginal p(x) by importance sampling of batched value variables
    # Define the error value with p(x) and with score(x) and with Normalize z
    # Train with backpropogation from the error value
    
    def __init__(self,
                 network_architecture=[100, 100, 40],
                 transfer_fun=tf.nn.sigmoid,
                 learning_rate=0.05, batch_size=100):
        self.sess = tf.Session()
        self.network_architecture = network_architecture
        self.size_x = network_architecture[0]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.transfer_fun = transfer_fun
        # create the tensorflow network
        self._create_network()
        self.exp_norm_const = tf.Variable(10*tf.ones([1] , dtype=tf.float32))
        self.tf_ph_score = tf.placeholder(tf.float32, [1, batch_size])
        # create training algorithm
        self.error_list, self.log_q_x, self.log_p_x= self._create_target_optimizer()
        # initialize
        init = tf.initialize_all_variables()
        self.sess.run(init)
        return

    def _create_network(self):
        # Initialize the weights and biases for DGN
        self._initialize_params()
        
        # Initialize the batched samples and weight variables
        self._define_batch_samples()
        
        # Build the forward generate rule with importance sampling p(h) estimation
        self.sample_handle = self._tf_sample_generator()

        return

    def _initialize_params(self):
        archit = self.network_architecture
        weights_list = []
        bias_list = [tf.Variable(tf.zeros([archit[0], 1], dtype=tf.float32))]
        updt_w_list = []
        updt_b_list = [tf.Variable(tf.zeros([archit[0], 1], dtype=tf.float32))]
        depth = len(archit) - 1
        for i in range(depth):
            n = archit[i]
            m = archit[i+1]
            w = tf.Variable(tf.random_uniform([n, m], -0.05, 0.05, dtype=tf.float32))
            b = tf.Variable(tf.zeros([m, 1], dtype=tf.float32))
            w_updt_v = tf.Variable(tf.zeros([n, m], dtype=tf.float32))
            b_updt_v = tf.Variable(tf.zeros([m, 1], dtype=tf.float32))
            weights_list.append(w)
            bias_list.append(b)
            updt_w_list.append(w_updt_v)
            updt_b_list.append(b_updt_v)
        self.updt_w_list = updt_w_list
        self.updt_b_list = updt_b_list
        self.weights_list = weights_list
        self.bias_list = bias_list
        return

    def _define_batch_samples(self):
        archit = self.network_architecture
        samp_prob1_var_list = [tf.Variable(tf.zeros([archit[0], self.batch_size]))]
        samp_var_list = [tf.Variable(tf.zeros([archit[0], self.batch_size]))]
        samp_w_var_list = [tf.Variable(tf.ones([1, self.batch_size]))]
        depth = len(archit) - 1
        for i in range(depth):
            n = archit[i]
            m = archit[i+1]
            pb = tf.Variable(tf.zeros([m, self.batch_size], dtype=tf.float32))
            sp = tf.Variable(tf.zeros([m, self.batch_size])) # binaris eventually, type can be optimized
            spw = tf.Variable(tf.zeros([1, self.batch_size])) # top layer don't have a weight
            samp_prob1_var_list.append(pb)
            samp_var_list.append(sp)
            samp_w_var_list.append(spw)
        self.samp_prob1_var_list = samp_prob1_var_list
        self.samp_var_list = samp_var_list
        self.samp_w_var_list = samp_w_var_list
        return

    # The generative model sample from top down.
    # For multiple layers, this requires a batch of samples to estimate the marginal probability
    # of each samples and forward with the weight vector.
    # An alternative would be using a particle filter resampling mechanism. which should be much more
    # acurate but break the auto BP so that we need to provide our own BP.
    def _tf_sample_generator(self):
        archit = self.network_architecture
        depth = len(archit) - 1
        self.samp_prob1_tfhl_list = [tf.tile(self.transfer_fun(self.bias_list[depth]), [1, self.batch_size])] # top layer is just the bias
        self.sample_tfhl_list = [sampleInt(self.samp_prob1_tfhl_list[0])]
        self.samp_w_tfhl_list = [tf.ones([1, self.batch_size])]
        sample_handle = [self.samp_var_list[depth].assign(self.sample_tfhl_list[0]),\
            self.samp_w_var_list[depth].assign(self.samp_w_tfhl_list[0]),\
            self.samp_prob1_var_list[depth].assign(self.samp_prob1_tfhl_list[0])]
        # sample from top to the bottom
        for i in range(depth-1, -1, -1): # not include top one
            n = archit[i]
            m = archit[i+1]
            spb = self.transfer_fun(tf.matmul(self.weights_list[i], self.sample_tfhl_list[0]) +\
                  tf.tile(self.bias_list[i], [1, self.batch_size]))
            # we need to save the prob of sample
            sp = sampleInt(spb)
            spb_assign_handle = self.samp_prob1_var_list[i].assign(spb)
            sp_assign_handle = self.samp_var_list[i].assign(sp)
            #compute_importance_weight(Hi+1, Hi, H_wi+1, W, b)
            spw = compute_importance_weight(self.sample_tfhl_list[0],
                                            sp,
                                            self.samp_w_var_list[0],
                                            self.weights_list[i],
                                            self.bias_list[i],
                                            self.batch_size)

            sample_handle.extend([sp_assign_handle, self.samp_w_var_list[i].assign(spw), spb_assign_handle])
            self.samp_prob1_tfhl_list.insert(0, spb)
            self.sample_tfhl_list.insert(0, sp)
            self.samp_w_tfhl_list.insert(0, spw)
        return sample_handle


    # the target optimizer will be used in the BP algorithm handled by tenserflow later.
    def _create_target_optimizer(self):
        archit = self.network_architecture
        depth = len(archit) - 1
        # 
        self.debug_norm_const = self.exp_norm_const
        log_q_x = self.tf_ph_score - tf.tile(tf.expand_dims(tf.log(self.exp_norm_const), -1), [1, self.batch_size])
        log_p_x = tf.log(self.samp_w_var_list[0])
        error_KLdiv = log_q_x - log_p_x
        #error = [tf.tile(error_KLdiv, [archit[0], 1])]
        error = [tf.tile(tf.mul(tf.exp(error_KLdiv), error_KLdiv), [archit[0], 1])] # top level, with exponential adjustment
        update_handle = []
        for i in range(depth): # not include top one

            b_, W_, e_h = back_propagate_one_layer(self.weights_list[i], \
                               self.samp_prob1_var_list[i], \
                               self.samp_var_list[i], \
                               self.samp_var_list[i+1],
                               error[i], False)
            error.append(e_h)
            update_handle.extend([self.updt_b_list[i].assign(tf.mul(self.learning_rate, b_)), \
                self.updt_w_list[i].assign(tf.mul(self.learning_rate, W_))])

        [b_, _, _] = back_propagate_one_layer(None, \
                           self.samp_prob1_var_list[depth], \
                           self.samp_var_list[depth], \
                           None,
                           error[depth], True)
        # error.append(e_h)
        update_handle.append(self.updt_b_list[depth].assign(tf.mul(self.learning_rate, b_)))
        # this is the update for the norm
        new_exp_norm_const = tf.exp(self.tf_ph_score - log_p_x)
        exp_norm_const_ = tf.clip_by_value(tf.mul(tf.reduce_mean(new_exp_norm_const -\
                                                                 tf.tile(tf.expand_dims(self.exp_norm_const, -1),\
                                                                            [1, self.batch_size]),\
                                                        1), 2*self.learning_rate), 1e-10, 1e35)
        update_handle.append(self.exp_norm_const.assign_add(tf.mul(self.learning_rate, exp_norm_const_)))
        self.update_handle = update_handle
        self.exp_norm_const_ = exp_norm_const_
        return error, log_q_x, log_p_x
       
    def _update(self, side_x):
        # check_op = tf.add_check_numerics_ops()
        # run the sample_handle will do a new round of sample and save into variables
        self.sess.run(self.sample_handle)
        # read from the variables the bottom level x and ask for score
        x = self.sess.run(self.samp_var_list[0])
        score = 10.0 * muscleDirectTorqueScore(side_x*side_x, side_x, x, 0)
        # feed the optimizer with the score and update the weight and bias and the exp_norm_const
        self.sess.run(self.update_handle, feed_dict={self.tf_ph_score: score})
        return score

def display(M, side_i, side_t):
    image = Image.fromarray(tile_raster_images(M,
                                               img_shape=(side_i, side_i),
                                               tile_shape=(side_t, side_t),
                                               tile_spacing=(2, 2)))
    image.show()

def train(network_architecture, learning_rate,
          batch_size, training_epochs=100, display_step=100):
    # TODO learning rate is wrong, should be defined in loop 
    rldgn = ReinforcementLearningDGN(network_architecture,
                                     tf.nn.sigmoid,
                                     learning_rate,
                                     batch_size)

    side_b = int(math.sqrt(batch_size)+ 0.1)
    side_x = int(math.sqrt(network_architecture[0])+ 0.1)
    side_h1 = int(math.sqrt(network_architecture[1])+ 0.1)
    # side_h2 = int(math.sqrt(network_architecture[2])+ 0.1)
    for epoch in range(training_epochs):
        # update one batch
        print "norm_const before update", rldgn.sess.run(rldgn.debug_norm_const)
        score = rldgn._update(side_x)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "score=", score
            display(rldgn.sess.run(rldgn.samp_var_list[0]).T, side_x, side_b)
            display(rldgn.sess.run(rldgn.weights_list[0]).T, side_x, side_h1)
            # display(rldgn.sess.run(rldgn.weights_list[1]).T, side_h1, side_h2)
            print "weight_var 0", rldgn.sess.run(rldgn.samp_w_var_list[0])
            print "weight_var 1", rldgn.sess.run(rldgn.samp_w_var_list[1])
            print "updt weight 0", rldgn.sess.run(rldgn.updt_w_list[0])
            print "updt bias 0", rldgn.sess.run(tf.transpose(rldgn.updt_b_list[0]))
            # print "updt weight 1", rldgn.sess.run(rldgn.updt_w_list[1])
            # print "updt bias 1", rldgn.sess.run(tf.transpose(rldgn.updt_b_list[1]))
            print "error 1", rldgn.sess.run(rldgn.error_list[1], feed_dict={rldgn.tf_ph_score: score})
            print "exp_norm_const_", rldgn.sess.run(rldgn.exp_norm_const_, feed_dict={rldgn.tf_ph_score: score})
            print "log_p_x", rldgn.sess.run(rldgn.log_p_x, feed_dict={rldgn.tf_ph_score: score})
            print "log_q_x", rldgn.sess.run(rldgn.log_q_x, feed_dict={rldgn.tf_ph_score: score})
        # in training display go here as a function.
        print "epoch ", epoch
    return rldgn

def main():
    # unit test some functions
    test()
    # train
    train([100, 9], 0.001, 49, training_epochs = 2001, display_step = 100)
    # post display

def test():
    # test one by one

    # test this
    if not test_compute_importance_weight():
        return False
    return

def sigmoid(x):
  return 1.0 / (1.0 + math.exp(float(-x)))

def sigmoid_list(X):
  return [sigmoid(x) for x in X ]

def sigmoid_dlist(X):
  return [sigmoid_list(x) for x in X ]

def test_compute_importance_weight():
    # sizes
    batch_size = 5
    high_size = 3
    low_size = 4
    # place holder for test
    sp_h = tf.placeholder(tf.float32, [high_size, batch_size])
    sp = tf.placeholder(tf.float32, [low_size, batch_size])
    spw_h = tf.placeholder(tf.float32, [1, batch_size])
    w = tf.placeholder(tf.float32, [low_size, high_size])
    b_h = tf.placeholder(tf.float32, [low_size, 1])
    # actual handle setup
    spw = compute_importance_weight(sp_h, sp, spw_h, w, b_h, batch_size)
    # set up
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    # set up test case value
    Sp_h_t = [ [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]]
    # Sp_h = [[0, 0, 1, 1, 0], [0, 1, 0, 1, 1], [1, 0, 0, 0, 1]]
    Sp_h = zip(*Sp_h_t)
    Sp_t = [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 1, 0]]
    Sp = zip(*Sp_t)
    Spw_h = [[1, 1, 1, 1, 1]]
    W = [[1, 0, 0], [0, 1, -1], [-1, 0 ,1], [0, -1, 1]]
    B_h_t = [[0, 0, 0, 0]]
    B_h = zip(*B_h_t)
    # test
    output = sess.run(spw, feed_dict={sp_h: Sp_h, sp: Sp, spw_h: Spw_h, w: W, b_h: B_h})
    print ("The output", output)
    true_result_pre_mean = sigmoid_dlist([[1, 0, -2, -2, 1], [-3, 2, 0, 2, -1], [1, -2, 2, 0, -1], [-1, 0, 2, 2, -1], [-1, 2, -2, 0, 1]])
    for i in range(batch_size):
        assert abs(sum(true_result_pre_mean[i])/float(batch_size) - output[0][i]) < 1.0e-5
    # with actual weight
    Spw_h = [[1.0, 0.2, 1.0, 0.2, 1.0]]
    output = sess.run(spw, feed_dict={sp_h: Sp_h, sp: Sp, spw_h: Spw_h, w: W, b_h: B_h})
    print ("The output", output)
    true_result_pre_mean = sigmoid_dlist([[1, 0, -2, -2, 1], [-3, 2, 0, 2, -1], [1, -2, 2, 0, -1], [-1, 0, 2, 2, -1], [-1, 2, -2, 0, 1]])
    for i in range(batch_size):
        true_result_pre_mean_weighted = copy.copy(true_result_pre_mean[i])
        for j in range(batch_size):
            true_result_pre_mean_weighted[j] *= Spw_h[0][j]
        assert abs(sum(true_result_pre_mean_weighted)/float(batch_size) - output[0][i]) < 1.0e-5
    return

if __name__ == "__main__":
    main()
