from collections import namedtuple

import numpy as np
import tensorflow as tf
from util_conv import *

from CVAE_type2 import CVAE_type2
from discriminator_model import DiscriminatorNet

import time, os

class CVAE_contextloss(CVAE_type2):
    def build_net(self, hps, encoder, decoder, input_nodes=None, add_name=""):
        super().hps_processing(hps, encoder, decoder, add_name=add_name)
        self._create_input_nodes(input_nodes)

        # set the training function
        self.train = self.train_from_pipeline
        
        with tf.variable_scope("model"):
            with tf.variable_scope('CVAE') as scope_cvae:
                self.z_mean, self.z_log_sigma_sq, self.z, self.recon_result = \
                        self.build_CVAE_net(self.inputs, encoder, decoder, self.zadd, is_training=self.is_training)
               
        dnet = DiscriminatorNet(self.shape_out[0])
        with tf.variable_scope('c_loss') as scope_c_loss:
            logit_gen, self.layers_gen = dnet.build_net_logit(self.recon_result, 2, is_training=self.is_training)
            self.look_like_orig = tf.nn.softmax(logit_gen) # for testing
        with tf.variable_scope('c_loss', reuse=True):
            logit_orig, self.layers_orig = dnet.build_net_logit(self.targets, 2, is_training=self.is_training)
        with tf.variable_scope("prior"):
            self.prior_mu, self.prior_log_var = super().create_prior_distribution(0.0,5.0)
            
        # loss and optimizer 
        with tf.variable_scope('loss') as scope_loss:
            self.d_loss, self.d_accuracy = self._create_discriminator_loss(logit_gen, logit_orig)
            self.d_train_op = self._create_train_op(self.d_loss, scope_c_loss)

            self.g_loss = self._create_contextual_loss(self.layers_gen, self.layers_orig) + \
                          self._create_latent_loss(self.z_log_sigma_sq, self.z_mean, self.prior_mu, self.prior_log_var)

            self.weight_adv = tf.Variable(1.0, tf.float32)
            self.g_loss = self.g_loss + self.weight_adv*self._create_adversarial_loss(logit_gen) # adversarial loss

            #self.g_loss = self.g_loss + self._create_adversarial_loss(logit_gen) # adversarial loss
            self.g_train_op = self._create_train_op(self.g_loss, scope_cvae)

        # get tensors for testing
        endpoints = self.get_endpoints()
        # add to a collection for retrieving later 
        for t in endpoints:
            tf.add_to_collection('endpoints', t)
        self.create_name()

    def _get_loss_and_accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            #accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)),
            #        'float32'))
            accuracy = tf.reduce_mean(tf.cast( tf.equal(labels, tf.argmax(logits, 1, output_type=tf.int32)),
                    'float32'))
        with tf.name_scope('loss'):
            #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        return loss, accuracy
    
    def _create_train_op(self, loss, scope):
        scope = scope.name 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope) # batch_norm ops in the scope
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        
        #print('n. of train. vars={}'.format(len(train_vars)))
        #print(train_vars)
        
        # Add l2 regularization term to loss
        loss = loss + self._decay(train_vars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        with tf.control_dependencies(update_ops): # for updating batch_norm
            train_op = optimizer.minimize(loss, var_list=train_vars)
        
        return train_op
    
    def _create_discriminator_loss(self, logit_gen, logit_orig):
        # The objective of the discriminator is to distinguish generated and original data
        batch_size = tf.shape(logit_gen)[0:1]
        #label_orig = tf.one_hot(tf.ones(batch_size, dtype=tf.int32),2,name='label_original') # label 1: original
        #label_gen  = tf.one_hot(tf.zeros(batch_size, dtype=tf.int32),2,name='label_genereated') # label 0: generated
        label_orig = tf.ones(batch_size, dtype=tf.int32, name='label_original') # label 1: original
        label_gen = tf.zeros(batch_size, dtype=tf.int32, name='label_genereated') # label 0: generated
        
        loss_orig, acc_orig = self._get_loss_and_accuracy(logit_orig, label_orig)
        loss_gen, acc_gen = self._get_loss_and_accuracy(logit_gen, label_gen)
        
        loss = loss_orig + loss_gen
        return loss, 0.5 * (acc_orig + acc_gen)

    def _create_adversarial_loss(self, logit_gen):
        batch_size = tf.shape(logit_gen)[0:1]
        #label_orig = tf.one_hot(tf.ones(batch_size, dtype=tf.int32),2,name='label_original_') # label 1: original
        label_orig = tf.ones(batch_size, dtype=tf.int32, name='label_original') # label 1: original
        loss_gen, acc_gen = self._get_loss_and_accuracy(logit_gen, label_orig)
        return loss_gen

    def _create_latent_loss(self, z_log_sigma_sq, z_mean, prior_mu, prior_log_var):
        self.loss_latent = tf.reduce_mean(super().KL_dist_diagonal(z_mean, z_log_sigma_sq, prior_mu, prior_log_var))
        #self.loss_latent = tf.reduce_mean(super().KL_dist_from_stdnormal(z_log_sigma_sq, z_mean))
        return self.loss_latent
    
    def _create_contextual_loss(self, layers_gen, layers_orig):
        dist_list = [tf.reduce_mean(self.reconstr_loss_func(orig, gen)) for (orig, gen) in zip(layers_orig, layers_gen)] 
        self.dist_tensor = tf.stack(dist_list)
        
        self.dist_divider = tf.placeholder(tf.float32, self.dist_tensor.get_shape(), name="dist_divider")
        
        loss_recon = tf.reduce_sum(self.dist_tensor / self.dist_divider)
        return loss_recon
    
    # add L2 of the variables having 'DW' in their names
    def _decay(self, t_vars):
        l2_loss_list = [tf.nn.l2_loss(var) for var in t_vars if 'DW' in var.name]
        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(l2_loss_list))

    def create_testing_nodes(self, batch_size):
        super().create_testing_nodes(batch_size)
        self.additional_test_node = self.look_like_orig

    ########################
    # Functions for training
    ########################
    
    def _set_initial_divider(self, layers):
        shape_list = [tensor.get_shape().as_list() for tensor in layers]
        num_elements_list = [np.prod(shape[1:]) for shape in shape_list]
        min_val = np.amin(num_elements_list)
        lambda_list = [num_elements/min_val for num_elements in num_elements_list]
        return lambda_list
    
    def partial_fit(self, sess, targets, data, dist_divider, freeze_D=False):
        feed_dict = self._get_feed_dict(targets,data)
        feed_dict[self.dist_divider] = dist_divider # set weights for loss

        if targets is None and data is None:
            if freeze_D:
                tensors_eval = [self.inputs, self.targets, self.zadd, self.d_accuracy]
            else:
                tensors_eval = [self.inputs, self.targets, self.zadd, self.d_accuracy, self.d_train_op]
            output = sess.run(tensors_eval, feed_dict=feed_dict)
            inputs, targets, zadd, d_accuracy = output[0], output[1], output[2], output[3]
            
            # save to reuse the same data for training the generative model
            # if we do not save and reuse, data will be dequeued multiple times for a single step
            feed_dict[self.inputs] = inputs
            feed_dict[self.targets] = targets
            feed_dict[self.zadd] = zadd
        else:
            if freeze_D:
                tensors_eval = [self.d_accuracy]
            else:
                tensors_eval = [self.d_accuracy, self.d_train_op]
            output = sess.run(tensors_eval, feed_dict=feed_dict)
            d_accuracy  = output[0]

        # run the g_train_op twice
        sess.run([self.g_train_op], feed_dict=feed_dict)
        sess.run([self.g_train_op], feed_dict=feed_dict)
        _, g_loss, dist_vec  = sess.run([self.g_train_op, self.g_loss, self.dist_tensor], feed_dict=feed_dict)
        
        return g_loss, d_accuracy, dist_vec

    def train_from_pipeline(self, sess, display_iter=200, lambda_update_iter=1000, weight_adv=1.0, freeze_on=True, model_folder="models", save_iter=10000):
        self.weight_adv.load(weight_adv,session=sess)

        model_filename = os.path.join(model_folder, self.get_name())
        #self.saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=24)
        self.saver = tf.train.Saver(max_to_keep=5)
        
        i=0
        start_time = time.time()
        g_loss_list, d_acc_list = [], []
        dist_list = []
        dist_divider = self._set_initial_divider(self.layers_gen)
        freeze_D = False
        try:
            while True:
                g_loss, d_accuracy, dist_vec = self.partial_fit(sess, None, None, dist_divider, freeze_D)
                g_loss_list.append(g_loss)
                d_acc_list.append(d_accuracy)
                dist_list.append(dist_vec)

                if (i+1) % display_iter == 0: # display, and determine whether freeze the discriminator or not
                    d_acc_ave = np.average(d_acc_list)
                    elapsed_time = time.time() - start_time
                    print ("[Iteration: {:4d}] ".format(i+1) +
                    "g_loss = {:.9f} | ".format(np.average(g_loss_list)) +
                    "d_accuracy = {:.9f} ".format(d_acc_ave) +
                      "({:.1f} sec.)".format(elapsed_time))
                    g_loss_list, d_acc_list = [], []

                    if d_acc_ave > 0.95 and freeze_on:
                        freeze_D_ = freeze_D
                        freeze_D = True
                        if freeze_D != freeze_D_:
                            print('freeze discriminator')
                    else:
                        freeze_D_ = freeze_D
                        freeze_D = False
                        if freeze_D != freeze_D_:
                            print('release discriminator')
                    start_time = time.time()
                if (i+1) % save_iter == 0:
                    self.saver.save(sess, model_filename, global_step=i+1)

                if (i+1) % lambda_update_iter == 0:
                    dist_divider = np.average(dist_list,0)
                    dist_divider = dist_divider/np.amin(dist_divider) + 1.0e-06
                    dist_list = []

                i += 1
        except tf.errors.OutOfRangeError:
            print("All data consumed, iteration: ", i)


    ########################
    # Other functions
    ########################       
    def get_endpoints(self):
        endpoints = super().get_endpoints()
        endpoints.append(self.look_like_orig)
        return endpoints

    # make sure keeping the order of variables in end_points same for get_endpoints and load_endpoints
    def load_endpoints(self, endpoints):
        num_prev = super().load_endpoints(endpoints)
        self.look_like_orig = endpoints[num_prev+0]
        return num_prev+1 # the number of variables assigned

    def create_name(self):
        name = "CVAE_closs_{}_latent{}".format(self.shape_in[0], self.latent_size)
        self.name = name + self.add_name
