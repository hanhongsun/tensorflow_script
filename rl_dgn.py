# in this file, we are trying to program a naive reinforcement learning deep generated model
# Author Di Sun

# TODO train the normal_const
# TODO set the train rule

# score function
from mt_reward import muscleTorqueScore

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
import math
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
    return tf.reduce_mean(weighted_sigmoid_matrix, 0, True)

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
        self.exp_norm_const = tf.Variable(tf.zeros([1] , dtype=tf.float32))
        self.tf_ph_score = tf.placeholder(tf.float32, [1, batch_size])
        # create training algorithm
        self._create_target_optimizer()
        # initialize
        init = tf.initialize_all_variables()
        self.sess.run(init)
        return

    def _create_network(self):
        # Initialize the weights and biases for DGN
        network_params = self._initialize_params()
        self.tf_weights_list = network_params[0]
        self.tf_bias_list = network_params[1]
        
        # Initialize the batched samples and weight variables
        batch_sample_vars = self._define_batch_samples()
        self.tf_sp_var_list = batch_sample_vars[0]
        self.tf_spw_var_list = batch_sample_vars[1]
        
        # Build the forward generate rule with importance sampling p(h) estimation
        self.sample_handle = self._tf_sample_generator()

        return

    def _initialize_params(self):
        archit = self.network_architecture
        weights_list = []
        bias_list = [tf.Variable(tf.zeros([archit[0], 1], dtype=tf.float32))]
        depth = len(archit) - 1
        for i in range(depth):
            n = archit[i]
            m = archit[i+1]
            w = tf.Variable(tf.random_uniform([n, m], -0.05, 0.05, dtype=tf.float32))
            b = tf.Variable(tf.zeros([m, 1], dtype=tf.float32))
            weights_list.append(w)
            bias_list.append(b)
        return weights_list, bias_list

    def _define_batch_samples(self):
        archit = self.network_architecture
        sample_var_list = [tf.Variable(tf.zeros([archit[0], self.batch_size]))]
        samp_w_var_list = [tf.Variable(tf.ones([1, self.batch_size]))]
        depth = len(archit) - 1
        for i in range(depth):
            n = archit[i]
            m = archit[i+1]
            sp = tf.Variable(tf.zeros([m, self.batch_size])) # binaris eventually, type can be optimized
            spw = tf.Variable(tf.zeros([1, self.batch_size])) # top layer don't have a weight
            sample_var_list.append(sp)
            samp_w_var_list.append(spw)
        return sample_var_list, samp_w_var_list

    # The generative model sample from top down.
    # For multiple layers, this requires a batch of samples to estimate the marginal probability
    # of each samples and forward with the weight vector.
    # An alternative would be using a particle filter resampling mechanism. which should be much more
    # acurate but break the auto BP so that we need to provide our own BP.
    def _tf_sample_generator(self):
        archit = self.network_architecture
        depth = len(archit) - 1
        self.sample_tfhl_list = [sampleInt( self.transfer_fun(self.tf_bias_list[depth]))]
        self.samp_w_tfhl_list = [tf.ones([1, self.batch_size])]
        sample_handle = []
        # sample from top to the bottom
        for i in range(depth-1, -1, -1): # not include top one
            n = archit[i]
            m = archit[i+1]
            sp = sampleInt(self.transfer_fun(tf.matmul(self.tf_weights_list[i], self.sample_tfhl_list[0])
                                             + tf.tile(self.tf_bias_list[i], 
                                                       [1, self.batch_size])))
            assign_handle = self.tf_sp_var_list[i].assign(sp)
            #compute_importance_weight(Hi+1, Hi, H_wi+1, W, b)
            spw = compute_importance_weight(self.tf_sp_var_list[i + 1],
                                            self.tf_sp_var_list[i],
                                            self.tf_spw_var_list[i + 1],
                                            self.tf_weights_list[i],
                                            self.tf_bias_list[i],
                                            self.batch_size)

            sample_handle.extend([assign_handle, self.tf_spw_var_list[i].assign(spw)])
            self.sample_tfhl_list.insert(0, sp)
            self.samp_w_tfhl_list.insert(0, spw)
        return sample_handle

    # the target optimizer will be used in the BP algorithm handled by tenserflow later.
    def _create_target_optimizer(self):
        
        log_q_x = self.tf_ph_score - tf.tile(tf.expand_dims(tf.log(self.exp_norm_const), -1), [1, self.batch_size])
        log_p_x = tf.log(self.tf_spw_var_list[0])
        error_KLdiv = log_q_x - log_p_x
        self.cost = tf.reduce_mean(\
            tf.square(tf.mul(tf.exp(error_KLdiv), error_KLdiv)),
            1)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        new_exp_norm_const = tf.exp(self.tf_ph_score - log_p_x)
        self.exp_norm_const_ = tf.mul(tf.reduce_mean(new_exp_norm_const -\
                                                    tf.tile(tf.expand_dims(self.exp_norm_const, -1),\
                                                                           [1, self.batch_size]),\
                                                1),\
                                 4*self.learning_rate)
        return
       
    def _update(self, side_x):
        # run the sample_handle will do a new round of sample and save into variables
        self.sess.run(self.sample_handle)
        # read from the variables the bottom level x and ask for score
        x = self.sess.run(self.tf_sp_var_list[0])
        score = 1.0 * muscleTorqueScore(side_x*side_x, side_x, x)
        # feed the optimizer with the score and update the weight and bias
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.tf_ph_score: score})
        # update the exp_norm_const from the variables stored
        update_norm_const = self.exp_norm_const.assign_add(self.exp_norm_const_)
        return cost, score

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

    avg_cost = 0.0
    side_b = int(math.sqrt(batch_size)+ 0.1)
    side_x = int(math.sqrt(network_architecture[0])+ 0.1)
    side_h1 = int(math.sqrt(network_architecture[1])+ 0.1)
    side_h2 = int(math.sqrt(network_architecture[2])+ 0.1)
    for epoch in range(training_epochs):
        # update one batch
        cost, score = rldgn._update(side_x)
        # Compute average loss
        avg_cost += float(cost) / (epoch + 1)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost), \
                  "score=", score
            display(rldgn.sess.run(rldgn.tf_sp_var_list[0]).T, side_x, side_b)
            display(rldgn.sess.run(rldgn.tf_weights_list[0]).T, side_x, side_h1)
            display(rldgn.sess.run(rldgn.tf_weights_list[1]).T, side_h1, side_h2)

        # in training display go here as a function.

    return rldgn

def main():
    # unit test some functions
    test()
    # train
    train([100, 49, 9], 0.01, 81, training_epochs = 10, display_step = 1)
    # post display

def test():
    # test one by one

    # test this
    if not test_compute_importance_weight():
        return False
    return

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
    # 
    #print (Sp, Sp_h, W)
    print ("The output", sess.run(spw, feed_dict={sp_h: Sp_h, sp: Sp, spw_h: Spw_h, w: W, b_h: B_h}))

    return

if __name__ == "__main__":
    main()
