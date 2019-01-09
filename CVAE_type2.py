from collections import namedtuple

import numpy as np
import tensorflow as tf
#from util_conv import *

from CVAE_general import CVAE

# decoder input = concatenate(encoder_output, zadd)
# self.latent_size : latent_size of the encoder. Additional data will be concatenated for the decoder 
# zadd_len: an integer, length of zadd
class CVAE_type2(CVAE):
    HParams = namedtuple('HParams', CVAE.HParams._fields + ('zadd_len',)) # Add a hyperparam

    def _create_input_nodes(self, input_nodes):
        if input_nodes is not None: # pair of (full, partial) data expected
            self.use_placeholders = False
            self.inputs, self.targets, self.zadd = input_nodes[0], input_nodes[0], input_nodes[1]
        else:
            self.use_placeholders = True
            self.inputs, self.targets, self.zadd = self._create_placeholders()

        # boolean value indicating whether it is for training or testing
        self.is_training = tf.placeholder(tf.bool, name='is_training') # for batch_norm

    def _check_hps(self, encoder, decoder): # collection of assertions, called in hps_processing()
        assert self.shape_in == self.shape_out # input = target for this model
        assert decoder.hps.latent_size == encoder.hps.latent_size + self.hps.zadd_len # check the decoder size
                      
    ########################
    # Functions for training
    ########################
    def _get_feed_dict(self, targets, data):
        feed_dict = {self.is_training:True}
        #data -> zadd, full data -> encoder
        if targets is not None:
            feed_dict[self.inputs] = targets
            feed_dict[self.targets] = targets
        if data is not None and self.zadd is not None:
            feed_dict[self.zadd]= data
        return feed_dict

    ########################
    # Functions for testing
    ########################       
    # Return posterior mean, var., and reconstruction result
    # In this model, posterior = prior
    def reconstruct(self, sess, zadd=None, z=None, get_add_info=False):
        if z is None:
            z_sample = self.z_prior_sample.eval() # Generate z from the prior distribution
        else:
            z_sample = z
        batch_size = self.batch_size
        if zadd is None:
            zadd = np.zeros((batch_size,)+tuple(self.zadd.shape[1:].as_list()),dtype=np.float32)
        feed_dict={self.z:z_sample, self.zadd: zadd, self.is_training:False}
        output_nodes = [self.prior_mu, self.prior_log_var, self.z, self.recon_result]
        # self.additional_test_node for contextloss model
        if get_add_info is True and self.additional_test_node is not None:
            if type(additional_test_node) is not list:
                output_nodes += [self.additional_test_node]
            else:
                output_nodes += self.additional_test_node
        return sess.run(output_nodes, feed_dict=feed_dict)
    # Return posterior mean, log(var.), and the reconstruction at the posterior mean
    # In this model, posterior = prior, hence, posterior mean = prior mean
    def reconstruct_best(self, sess, zadd): # reconstruction with z=0
        batch_size = self.batch_size
        latent_size = int(self.z.shape[1])
        print(zadd.shape)
        z_sample = np.zeros((batch_size,latent_size))
        feed_dict={self.z:z_sample, self.zadd: zadd, self.is_training:False}
        return sess.run([self.prior_mu, self.prior_log_var, self.recon_result], feed_dict=feed_dict)

    # Return the Reconstruction with Y (truth) to check it can actually generate Y
    def reconstruct_with_full_data(self, sess, Y, zadd=None ):
        batch_size = Y.shape[0]
        if zadd is None:
            zadd = np.zeros((batch_size,)+tuple(self.zadd.shape[1:].as_list()),dtype=np.float32)
        z_mean, z_log_var = self.encode(sess, Y) # Encode the full data 
        feed_dict={self.z_mean:z_mean, self.z_log_sigma_sq:z_log_var, self.zadd:zadd, self.is_training:False}
        reconstructed = sess.run(self.recon_result, feed_dict=feed_dict)
        return z_mean, z_log_var, reconstructed
    
    def encode(self, sess, X):
        feed_dict={self.inputs: X, self.is_training:False}
        return sess.run([self.z_mean, self.z_log_sigma_sq], feed_dict=feed_dict)
    def decode(self, sess, z, zadd=None):
        batch_size = z.shape[0]
        if zadd is None:
            zadd = np.zeros((batch_size,)+tuple(self.zadd.shape[1:].as_list()),dtype=np.float32)
        feed_dict={self.z: z, self.zadd:zadd, self.is_training:False}
        return sess.run(self.recon_result, feed_dict=feed_dict)
    def decode_distribution(self, sess, z_mean, z_logvar, zadd=None):
        batch_size = z_mean.shape[0]
        if zadd is None:
            zadd = np.zeros((batch_size,)+tuple(self.zadd.shape[1:].as_list()),dtype=np.float32)
        feed_dict={self.z_mean:z_mean, self.z_log_sigma_sq:z_logvar, self.zadd:zadd, self.is_training:False}
        return sess.run(self.recon_result, feed_dict=feed_dict)
    def generate_z(self, sess, X):
        # z of type2 does not depend on X
        return self.z_prior_sample.eval()

    def test_z_with_observed(self, sess, z, zadd, observed, mask, get_add_info=False):
        # observed: for calculating pixelwise distance from reconstructions
        batch_size = self.batch_size
        if len(zadd.shape) is 1:
            zadd = np.stack([zadd]*batch_size,axis=0)

        feed_dict={self.z: z, self.zadd:zadd,
                self.observed: observed, self.mask: mask, self.is_training:False}
        output_nodes = [self.recon_result, self.distance_test]

        # self.additional_test_node for contextloss model
        if get_add_info is True and self.additional_test_node is not None:
            if type(additional_test_node) is not list:
                output_nodes += [self.additional_test_node]
            else:
                output_nodes += self.additional_test_node
        result = sess.run(output_nodes,feed_dict=feed_dict)
        return result
        
    ########################
    # Other functions
    ########################       
    def get_endpoints(self):
        endpoints = super().get_endpoints()
        endpoints.append(self.zadd)
        return endpoints

    # make sure keeping the order of variables in end_points same for get_endpoints and load_endpoints
    def load_endpoints(self, endpoints):
        num_prev = super().load_endpoints(endpoints)
        self.zadd = endpoints[num_prev+0]
        return num_prev+1 # the number of variables assigned

    def create_name(self):
        name = "CVAE_type2_{}_latent{}".format(self.shape_in[0], self.latent_size)
        self.name = name + self.add_name
