from collections import namedtuple

import numpy as np
import tensorflow as tf

from util_conv import *

import time

class DiscriminatorNet(object):
    def __init__(self, resol):
        self.resol = resol
        # Depth parameter for the network to incorporate various input sizes
        assert resol == 32 or resol == 128 # 32 and 128 are tested
        n = [n for n in range(1,11) if 1<<n == resol] # 1<<n means 2^n
        self.depth = n[0]# list containing a single number -> number
        #2^depth = resol
        
    # labels: batch_size X one-hot vector
    def build_net_logit(self, inputs, num_classes, is_training=True, activ=tf.nn.elu, do_BN=True):
        # The number of filters
        filters = [64, 128] # For initial, adaptive, and final layers, respectively
        current_depth = self.depth
                        
        layers = [inputs]
        # If the input size >= 128, 5x5 kernel + stride
        if current_depth >= 7:
            layers.append(conv_unit('init_conv', layers[-1], 5, filters[0], [1, 2, 2, 1],
                                    do_batch_norm=do_BN, is_training=is_training, activ_func=activ))
            current_depth = current_depth - 1
        # Reduce to 4x4
        for i in range(current_depth-2):
            x_temp = conv_unit('unit_{}_1'.format(i+1), layers[-1], 3, filters[1], [1, 1, 1, 1], 
                               do_batch_norm=False, is_training=is_training, activ_func=activ)
            layers.append(conv_unit('unit_{}_stride'.format(i+1), x_temp, 3, filters[1], [1, 2, 2, 1], 
                                    do_batch_norm=do_BN, is_training=is_training, activ_func=activ))
            

        layers.append(global_avg_pool(layers[-1])) # Average pooling
        logits = fully_connected('linear',layers[-1], num_classes)
        """    
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)),
                    'float32'))

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            # TODO: applying decay?
        """    
        return logits, layers

    def build_net_logit_layernorm(self, inputs, num_classes, is_training=True, activ=tf.nn.elu):
        # The number of filters
        filters = [64, 128] # For initial, adaptive, and final layers, respectively
        current_depth = self.depth
                        
        layers = [inputs]
        # If the input size >= 128, 5x5 kernel + stride + max pooling 
        if current_depth >= 7:
            layers.append(conv_unit_layernorm('init_conv', layers[-1], 5, filters[0], [1, 2, 2, 1],
                                     is_training=is_training, activ_func=activ))
            current_depth = current_depth - 1
        # Reduce to 4x4
        for i in range(current_depth-2):
            x_temp = conv_unit_layernorm('unit_{}_1'.format(i+1), layers[-1], 3, filters[1], [1, 1, 1, 1], 
                                is_training=is_training, activ_func=activ)
            layers.append(conv_unit_layernorm('unit_{}_stride'.format(i+1), x_temp, 3, filters[1], [1, 2, 2, 1], 
                                     is_training=is_training, activ_func=activ))
            

        layers.append(global_avg_pool(layers[-1])) # Average pooling
        logits = fully_connected('linear',layers[-1], num_classes)
        return logits, layers
    
    def build_full_net(self, img_depth, num_classes):
        self.num_classes = num_classes
        self.img_depth = img_depth
        with tf.variable_scope('classifier'):
            self.inputs = tf.placeholder(tf.float32, [None, self.resol, self.resol, img_depth], name="inputs") # input & mask
            self.is_training = tf.placeholder(tf.bool, name='is_training') # for batch_norm 
        
            self.logits, _ = self.build_net_logit(self.inputs, num_classes, self.is_training, do_BN=True)
        
            self.labels = tf.placeholder(tf.float32, [None, num_classes], name="labels")
        
            self.labels_predicted = tf.argmax(self.logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(self.labels, 1), self.labels_predicted), 'float32'))
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(xent)
            self.loss += self._decay(decay_rate=0.001)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-03)
        # for batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        
    def _decay(self, decay_rate=0.0001):
        t_vars = tf.trainable_variables()
        l2_loss_list = [tf.nn.l2_loss(var) for var in t_vars if 'DW' in var.name]
        return tf.multiply(decay_rate, tf.add_n(l2_loss_list))
    
    def partial_fit(self, sess, inputs, labels):
        feed_dict = {self.inputs: inputs, self.labels: labels, self.is_training:True}
        _, accuracy, cost  = sess.run([self.train_op, self.accuracy, self.loss], feed_dict=feed_dict)
        return accuracy, cost
    
    def train(self, sess, source, batch_size, noise_gen=None, training_epochs=10, display_step=1):
        height = self.resol
        width = self.resol
        
        n_samples = source.train.num_examples
        for epoch in range(training_epochs):
            cost_list = []
            accu_list = []
            total_batch = int(n_samples / batch_size)
            start_time = time.time()
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, labels = source.train.next_batch(batch_size)
                batch_xs = np.reshape(batch_xs, (-1,height,width,self.img_depth))#reshape
                if noise_gen is not None:
                    if type(noise_gen) is list:
                        for func in noise_gen:
                            batch_xs, _ = func(batch_xs)
                    else:
                        batch_xs, _ = noise_gen(batch_xs)

                # Fit training using batch data
                accu, cost = self.partial_fit(sess, batch_xs, labels)
                # Compute average loss
                cost_list.append(cost)
                accu_list.append(accu)
                
            elapsed_time = time.time() - start_time

            # Display logs per epoch step
            if epoch % display_step == 0:
                print ("[Epoch: {:4d}] ".format(epoch+1) +
                    "current cost = {:.6f} | ".format(cost) +
                    "avg cost = {:.6f} ".format(np.average(cost_list)) +
                    "avg accuracy(train) = {:.3f} ".format(np.average(accu_list)) +
                      "({:.1f} sec.)".format(elapsed_time))
            if (epoch+1) % (display_step*1) == 0:    
                chart = self.test(sess, source, batch_size)
        return chart
            
    
    def test(self, sess, source, batch_size):
        height = width = self.resol
        n_test = source.test.num_examples
        chart = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for i in range(n_test):
            batch_xs, labels = source.test.next_batch(batch_size) # get duplicated batch of a single example
            batch_xs = np.reshape(batch_xs, (-1,height,width, self.img_depth))#reshape
            
            feed_dict = {self.inputs: batch_xs, self.is_training:False}
            labels_predicted = sess.run([self.labels_predicted], feed_dict=feed_dict)
            
            label_true = np.argmax(labels[0])
            label_pred = labels_predicted[0]
            chart[label_true, label_pred] += 1
            
        print(chart)
        return chart
    
    def get_name(self):
        name = "classifier_%d" % self.resol
        return name
    
    def save(self, sess, filename):
        print ('saving cvae model to %s...' % filename)
        saver = tf.train.Saver()
        saver.save(sess, filename)
        
    def load(self, sess, model_foler):
        saver = tf.train.Saver()
        print ('restoring cvae model from %s...' % model_foler)
        saver.restore(sess, tf.train.latest_checkpoint(model_foler))
    
