from collections import namedtuple

import numpy as np
import tensorflow as tf
#from util_conv import *

from CVAE_general import CVAE

class CVAE_type1(CVAE):
    def _create_input_nodes(self, input_nodes):
        if input_nodes is not None: # pair of (full, partial) data expected
            self.use_placeholders = False
            self.inputs, self.targets, self.zadd = input_nodes[1], input_nodes[0], None
        else:
            self.use_placeholders = True
            self.inputs, self.targets, self.zadd = self._create_placeholders()

        # boolean value indicating whether it is for training or testing
        self.is_training = tf.placeholder(tf.bool, name='is_training') # for batch_norm

    def _check_hps(self, encoder, decoder): # collection of assertions, called in hps_processing()
        assert encoder.hps.latent_size == decoder.hps.latent_size

    ########################
    # Functions for training
    ########################
    def _get_feed_dict(self, targets, data):
        feed_dict = {self.is_training:True}
        #data -> encoder
        if targets is not None:
            feed_dict[self.targets] = targets
        if data is not None:
            feed_dict[self.inputs] = data
        return feed_dict

    ########################
    # Functions for testing
    ########################       
    # Return posterior mean, log(var.), and reconstruction result
    def reconstruct(self, sess, X, z = None, get_add_info=False):
        if z is None:
            feed_dict={self.inputs: X, self.is_training:False}
        else:
            feed_dict={self.z: z, self.is_training: False}
        output_nodes = [self.z_mean, self.z_log_sigma_sq, self.z, self.recon_result]
        if get_add_info is True and self.additional_test_node is not None:
            if type(additional_test_node) is not list:
                output_nodes += [self.additional_test_node]
            else:
                output_nodes += self.additional_test_node

        return sess.run(output_nodes, feed_dict=feed_dict)
    # Return posterior mean, log(var.), and the reconstruction at the posterior mean
    def reconstruct_best(self, sess, X):
        z_mean, z_log_var = self.encode(sess, X)

        feed_dict={self.z: z_mean, self.is_training:False}
        reconstructed = sess.run(self.recon_result, feed_dict=feed_dict)
        return z_mean, z_log_var, reconstructed 
    
    # Return the Reconstruction with Y (truth) to check it can actually generate Y
    # In this model, Y (truth) can be be utilised to generate reconstructions
    def reconstruct_with_full_data(self, sess, Y, X):
        return self.reconstruct_best(sess, X)
    
    def encode(self, sess, X):
        feed_dict={self.inputs: X, self.is_training:False}
        return sess.run([self.z_mean, self.z_log_sigma_sq], feed_dict=feed_dict)
    def decode(self, sess, z, zadd = None):
        feed_dict={self.z: z, self.is_training:False}
        return sess.run(self.recon_result, feed_dict=feed_dict)
    def generate_z(self, sess, X):
        feed_dict={self.inputs: X, self.is_training:False}
        return sess.run([self.z, self.z_mean, self.z_log_sigma_sq], feed_dict=feed_dict)
        
    ########################
    # Other functions
    ########################       

    def create_name(self):
        name = "CVAE_type1_{}_latent{}".format(self.shape_in[0], self.latent_size)
        self.name = name + self.add_name
