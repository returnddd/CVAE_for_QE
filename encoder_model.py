import numpy as np
import tensorflow as tf

from collections import namedtuple
from util_conv import *

class encoder_interface(object):
    def get_activ_func(self, str_activ):
        if str_activ == 'elu':
            activ = tf.nn.elu
        elif str_activ == 'relu':
            activ = tf.nn.relu
        else:
            activ = tf.nn.elu # Default
        return activ
    def get_shape(self):
        raise NotImplementedError # expected to return (height, width, layers)

# Fully connected encoder
class encoder_FC(encoder_interface): 
    # input_shape : (data_length,): tuple with a single integer
    # layers_units : sequence of ints. It defines the number of layers and the number of units
    # latent_size : integer, dimension of the latent space 
    # activ_unit : string, will be converted to a tf function
    # do_batch_norm : True or False
    HParams = namedtuple('HParams', 
                         'input_shape, layers_units, latent_size, activ_unit')
    def __init__(self, hps):
        self.hps = hps
        self.activ = self.get_activ_func(hps.activ_unit) # string to function
    def get_shape(self):
        return self.hps.input_shape
    def build_encoder(self, x, is_training):
        latent_size = self.hps.latent_size
        activ = self.activ
        
        return self.encoder(x, latent_size, activ, is_training)
    def encoder(self, x, latent_size, activ=tf.nn.elu, is_training=True):
        depth = len(self.hps.layers_units)
        for i in range(depth):
            x = fully_connected('linear_{}'.format(i+1),x, self.hps.layers_units[i])
            # Batch norm?
        params = fully_connected('final', x, latent_size*2)
        mean = params[:, :latent_size]
        stddev = params[:, latent_size:]
        return mean, stddev

class encoder_pic(encoder_interface):
    # input_shape : (height,width,depth)
    HParams = namedtuple('HParams', 
                         'input_shape, latent_size, activ_unit, do_batch_norm')
    def __init__(self, hps):
        self.hps = hps
        self.activ = self.get_activ_func(hps.activ_unit) # string to function
    def get_shape(self):
        return self.hps.input_shape
    def get_depth(self):
        return convert_shape_to_depth(self.hps.input_shape[0:2])

    
    # x: tensor, shape = [batch_size, N, N, num_channels]
    # returns (mean, std) tensors
    # shape of mean, std: [batch_size, latent_size]
    def build_encoder(self, x, is_training):
        latent_size = self.hps.latent_size
        activ = self.activ
        do_BN = self.hps.do_batch_norm
        
        return self.encoder(x, latent_size, activ, is_training, do_BN)
    
    def encoder(self, x, latent_size, activ=tf.nn.elu, is_training=True, do_BN=False):
        input_hw = x.get_shape().as_list()[1:3] # (height, width) can be inferred from x (NHWD assumed)
        current_depth = convert_shape_to_depth(input_hw)
        filters = [64, 128] # For initial, adaptive, and final layers, respectively
        
        # If the input size >= 128, 5x5 kernel + stride + max pooling 
        if current_depth >= 7:
            x = conv_unit('conv_large', x, 5, filters[0], 2, 
                          is_training=is_training, activ_func=activ, do_batch_norm=do_BN) # convolution while keeping the resolution
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            current_depth = current_depth - 2
            
        # Reduce to 4x4
        for i in range(current_depth-2):
            x = conv_unit('conv_{}'.format(i+1), x, 3, filters[1], 1, 
                          is_training=is_training, activ_func=activ, do_batch_norm=do_BN) # convolution while keeping the resolution
            x = conv_unit('conv_{}_reduce'.format(i+1), x, 3, filters[1], 2, 
                          is_training=is_training, activ_func=activ, do_batch_norm=do_BN) # convolution while halving the resolution
            
        params = fully_connected('linear', x, latent_size*2)
            
        mean = params[:, :latent_size]
        stddev = params[:, latent_size:]
        return mean, stddev
