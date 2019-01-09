from collections import namedtuple

import numpy as np
import tensorflow as tf
from util_conv import *

import time
import os
import pickle

# Abstract class, not to be instantiated
class CVAE(object):
    HParams = namedtuple('HParams', 'batch_size, loss_func, weight_decay_rate')

    def _create_placeholders(self): # called in __init__()
        inputs = tf.placeholder(tf.float32, (None,)+self.shape_in, name="inputs")  # encoder input
        targets = tf.placeholder(tf.float32, (None,)+self.shape_out, name="targets") # reconstruction target

        if 'zadd_len' in self.hps._fields and self.hps.zadd_len is not None:
            print('zadd enabled')
            zadd = tf.placeholder(tf.float32, (None, self.hps.zadd_len), name="zext") # another decoder input (will be concatenated with z) 
        else:
            zadd = None
        return inputs, targets, zadd

    # if input_nodes (expected from dataset api) are None, then create placeholders
    def _create_input_nodes(self, input_nodes):
        if input_nodes is not None:
            raise ValueError("This class(CVAE) cannot be built on the input nodes. Use a subclass implementing it")
        else:
            self.inputs, self.targets, self.zadd = self._create_placeholders()

        # boolean value indicating whether it is for training or testing
        self.is_training = tf.placeholder(tf.bool, name='is_training') # for batch_norm

    def build_net(self, hps, encoder, decoder, input_nodes=None, add_name=""):
        self.hps_processing(hps, encoder, decoder,  add_name) # Setting hyper-parameters
        self._create_input_nodes(input_nodes)

        # select the training function
        if input_nodes is not None:
            self.train = self.train_from_pipeline
        else:
            self.train = self.train_placeholders
        
        with tf.variable_scope("model"):
            self.z_mean, self.z_log_sigma_sq, self.z, self.recon_result = \
                    self.build_CVAE_net(self.inputs, encoder, decoder, self.zadd, is_training=self.is_training)
        with tf.variable_scope("prior"):
            self.prior_mu, self.prior_log_var = self.create_prior_distribution(0.0,5.0)
        # loss and optimizer 
        self.loss, self.train_op = self._create_loss_and_optimizer(self.targets,
                                                                    self.recon_result, 
                                                                    self.z_log_sigma_sq,
                                                                    self.z_mean)
        # get tensors for testing
        endpoints = self.get_endpoints()
        # add to a collection for retrieving later 
        for t in endpoints:
            tf.add_to_collection('endpoints', t)

        self.create_name()


    def _check_hps(self, encoder, decoder): # collection of assertions, called in hps_processing()
        pass

    def hps_processing(self, hps, encoder, decoder,  add_name=""):
        self.add_name = add_name
        self.hps = hps

        self.shape_in = encoder.get_shape() # Except batch_size
        self.shape_out = decoder.get_shape() # Except batch_size

        self.latent_size = encoder.hps.latent_size 
        
        if 'loss_func' in hps._fields:
            # Loss function for the reconstruction result
            if self.hps.loss_func == 'cross_entropy':
                self.elementwise_loss = self.cross_entropy_elementwise
            elif self.hps.loss_func == 'L1':
                self.elementwise_loss = self.L1_elementwise
            elif self.hps.loss_func == 'L2':
                self.elementwise_loss = self.L2_elementwise
            else:
                assert False
        self._check_hps(encoder, decoder)
        
    def create_prior_distribution(self, mean, std, is_trainable=False):
        #prior distribution
        if is_trainable:
            #for trainable prior parameters
            raise NotImplementedError # TODO: Not implemented yet
            prior_mu = tf.get_variable("prior_mu", dtype=tf.float32, initializer=prior_mu)
            prior_log_var=tf.constant("prior_log_var", dtype=tf.float32, initializer=prior_log_var)
        else:
            prior_mu = tf.constant(mean,shape=[self.latent_size], dtype=tf.float32, name='prior_mu_init')
            prior_log_var=tf.constant(np.log(std*std),shape=[self.latent_size], dtype=tf.float32, name='prior_log_var_init')
        return prior_mu, prior_log_var
    

    def cross_entropy_elementwise(self, target, recon):
        return -(target * tf.log(tf.clip_by_value(recon, 1e-10, 1.0))
                              + (1.0 - target) * tf.log(tf.clip_by_value(1.0 - recon, 1e-10, 1.0)))

    def L1_elementwise(self, target, recon):
        return tf.abs(target - recon)
    def L2_elementwise(self, target, recon):
        return tf.square(target-recon)
    
    def reconstr_loss_func(self, target, recon, elementwise_loss=None, mask=None):
        if elementwise_loss is None:
            elementwise_loss = self.elementwise_loss
        dim = len(target.get_shape())
        dims = [i for i in range(1,dim)]
        elementwise = elementwise_loss(target,recon)
        if mask is not None:
            elementwise = elementwise * mask
        return tf.reduce_sum(elementwise , dims)
    
    def KL_dist_from_stdnormal(self, z_log_sigma_sq, z_mean):
        return -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq
                                    - tf.square(z_mean)
                                    - tf.exp(z_log_sigma_sq), 1)
    #KL( N(mu_0,SIG_0)|| N(mu_1,SIG_1)) 
    def KL_dist_diagonal(self, mu0, log_var0, mu1, log_var1):
        return 0.5 * tf.reduce_sum((tf.exp(log_var0)+tf.square(mu1-mu0))/tf.exp(log_var1) - 1.0 +
                                   log_var1 - log_var0, 1)
    
    def _create_loss_and_optimizer(self, target, output, z_log_sigma_sq, z_mean):
        with tf.variable_scope('loss'):
            reconstr_loss = self.reconstr_loss_func(target, output)
            #latent_loss = self.KL_dist_from_stdnormal(z_log_sigma_sq, z_mean)
            latent_loss = self.KL_dist_diagonal(z_mean, z_log_sigma_sq, self.prior_mu, self.prior_log_var)
            loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
            loss = loss + self._decay() # add l2_loss
        
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch_norm ops
            with tf.control_dependencies(update_ops): # for updating batch_norm
                train_op = optimizer.minimize(loss)
        
        return loss, train_op
    
    # add L2 of the variables having 'DW' in their names
    def _decay(self):
        t_vars = tf.trainable_variables()
        l2_loss_list = [tf.nn.l2_loss(var) for var in t_vars if 'DW' in var.name]
        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(l2_loss_list))
    
    def _get_encoded(self, x, encoder, is_training):
        z_mean, z_log_sigma_sq = encoder.build_encoder(x, is_training)
        # Draw random samples
        #x_batch_size = z_log_sigma_sq.get_shape().as_list()[0]
        #eps_batch =  x_batch_size if x_batch_size is not None else self.hps.batch_size
        #eps = tf.random_normal([eps_batch, self.latent_size], 0.0, 1.0, dtype=tf.float32)
        eps = tf.random_normal(tf.shape(z_log_sigma_sq), 0.0, 1.0, dtype=tf.float32)

        z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)) # posterior z

        return z_mean, z_log_sigma_sq, z
                      
    def build_CVAE_net(self, x, encoder, decoder, zadd=None, is_training=True):
        #with tf.variable_scope('encoder'):
        z_mean, z_log_sigma_sq, z = self._get_encoded(x,encoder,is_training)

        if zadd is not None : # zadd is for extension
            z_ext = tf.concat([z,zadd], 1) # Concatenate z and zadd
        else:
            z_ext = z

        # Get the reconstructed mean from the decoder
        #with tf.variable_scope('decoder'):
        x_reconstr_mean = decoder.build_decoder(z_ext,is_training)
                      
        return z_mean, z_log_sigma_sq, z, x_reconstr_mean

    ########################
    # Functions for training
    ########################
    def _get_feed_dict(self, targets, data): # data is only for training with placeholders
        feed_dict = {self.is_training:True}
        # Derived classes should define whether data goes to zadd or self.inputs
        raise NotImplementedError
        return feed_dict

    def partial_fit(self, sess, targets=None, data=None):
        feed_dict = self._get_feed_dict(targets,data)
        _, cost  = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return cost

    def train_from_pipeline(self, sess, display_iter=200):
        i=0
        start_time = time.time()
        cost_list = []
        try:
            while True:
                # Fit training using batch data
                cost = self.partial_fit(sess)
                # Compute average loss
                cost_list.append(cost)

                if (i+1) % display_iter == 0:
                    elapsed_time = time.time() - start_time
                    print ("[Iteration: {:4d}] ".format(i+1) +
                    "current cost = {:.9f} | ".format(cost) +
                    "avg cost = {:.9f} ".format(np.average(cost_list)) +
                      "({:.1f} sec.)".format(elapsed_time))

                    start_time = time.time()
                    cost_list = []
                i += 1
        except tf.errors.OutOfRangeError:
            print("Training completed")


    def train_placeholders(self, sess, source, input_gen=None, training_epochs=10, display_step=1):
        batch_size = self.hps.batch_size
        
        n_samples = source.train.num_examples
        for epoch in range(training_epochs):
            cost_list = []
            total_batch = int(n_samples / batch_size)
            start_time = time.time()
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = source.train.next_batch(batch_size)
                if input_gen is not None:
                    input_net = input_gen(batch_xs)
                else:
                    input_net = batch_xs

                # Fit training using batch data
                cost = self.partial_fit(sess, batch_xs, input_net)
                # Compute average loss
                cost_list.append(cost)
                
            elapsed_time = time.time() - start_time

            # Display logs per epoch step
            if epoch % display_step == 0:
                print ("[Epoch: {:4d}] ".format(epoch+1) +
                    "current cost = {:.9f} | ".format(cost) +
                    "avg cost = {:.9f} ".format(np.average(cost_list)) +
                      "({:.1f} sec.)".format(elapsed_time))

    def create_name(self):
        #model name
        name = "CVAE_latent{}".format(self.latent_size)
        self.name = name + self.add_name
    def get_name(self):
        return self.name

    ########################
    # Functions for testing
    ########################       
    # Multivariate Gaussian with diagonal covariance matrix is assumed
    def build_prior_density(self, prior_mu, prior_log_var):
        var = tf.exp(prior_log_var)
        log_prior_density_each = -tf.square(self.z - prior_mu) / (2.0*var) - 0.5*tf.log(2.0*np.pi*var)
        self.log_prior_density = tf.reduce_sum(log_prior_density_each, axis=1)
        self.prior_density = tf.exp(self.log_prior_density)
    # Nodes for testing
    def create_testing_nodes(self, batch_size):
        # batch_size: for testing
        #with tf.variable_scope("test"):
        # Node for sampling from the prior distribution
        latent_size = int(self.z.shape[1])
        eps = tf.random_normal([batch_size, latent_size], 0.0, 1.0, dtype=tf.float32)
        self.z_prior_sample = tf.add(self.prior_mu, tf.multiply(tf.sqrt(tf.exp(self.prior_log_var)), eps))

        # Node for prior density of z
        self.build_prior_density(self.prior_mu, self.prior_log_var)

        # Distance metric
        shape_out = tuple(self.recon_result.shape[1:])
        self.observed = tf.placeholder(tf.float32, (1,)+shape_out, name="observed") 
        self.mask = tf.placeholder(tf.float32, (1,)+shape_out, name="mask") 

        self.distance_test = self.reconstr_loss_func(self.observed, self.recon_result,
                                                     elementwise_loss = self.L1_elementwise,
                                                     mask=self.mask)
        self.additional_test_node = None
        self.batch_size = batch_size
    
    def get_prior_density(self, sess, z):
        return  sess.run([self.log_prior_density, self.prior_density], feed_dict={self.z:z})


    ########################
    # Other functions
    ########################       
    # Save and load a model
    def get_endpoints(self):
        endpoints = [self.inputs, self.is_training, self.z, self.z_mean, self.z_log_sigma_sq, 
                self.prior_mu, self.prior_log_var, self.recon_result]
        return endpoints

    # make sure keeping the order of variables in end_points same for get_endpoints and load_endpoints
    def load_endpoints(self, endpoints):
        self.inputs = endpoints[0]
        self.is_training = endpoints[1]
        self.z = endpoints[2]
        self.z_mean = endpoints[3]
        self.z_log_sigma_sq = endpoints[4]
        self.prior_mu = endpoints[5]
        self.prior_log_var = endpoints[6]
        self.recon_result = endpoints[7]
        return 8 # the number of variables assigned

    def save(self, sess, filename):
        print ('saving cvae model to %s...' % filename)
        saver = tf.train.Saver()
        saver.save(sess, filename)

    def load(self, sess, model_dir, file_name=None):
        #saver = tf.train.Saver()

        self.name = file_name # model name
        model_path = os.path.join(model_dir,file_name)  
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)

        # New models should have the 'endpoints' collection
        endpoints = tf.get_collection('endpoints')

        # Old models have no collection named endpoints. Load the endpoints from its tensor name
        if len(endpoints) == 0:
            fpath = model_path + '.endpoints'
            if not os.path.exists(fpath):
                raise ValueError('Cannot find endpoints.')
            infile = open(fpath,'rb')
            names = pickle.load(infile)
            infile.close()
            graph = tf.get_default_graph()
            endpoints = []
            for name in names:
                endpoints.append(graph.get_tensor_by_name(name))

            
        self.load_endpoints(endpoints)

