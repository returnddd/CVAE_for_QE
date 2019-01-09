import numpy as np
import tensorflow as tf

from collections import namedtuple
from util_conv import *

def map_str_to_func(str_activ):
    if str_activ == 'elu':
        activ = tf.nn.elu
    elif str_activ == 'relu':
        activ = tf.nn.relu
    elif str_activ == 'tanh':
        activ = tf.tanh
    elif str_activ == 'sigmoid':
        activ = tf.sigmoid
    else:
        raise ValueError("Unsupported activation function ({})".format(str_activ))
    return activ

def get_activ_func(activ):
    if activ is None:
        return None
    elif type(activ) is str:
        return map_str_to_func(activ)
    elif type(activ) is list:
        return [map_str_to_func(x) for x in activ]
    else:
        raise ValueError("Definition of activation units should be string or list of string")

class decoder_interface(object):
    def get_shape(self):
        raise NotImplementedError # expected to return (height, width, layers)

class decoder_pic(decoder_interface):
    HParams = namedtuple('HParams',
                         'latent_size, output_shape, activ_unit, last_activ_unit, do_batch_norm')

    def __init__(self, hps):
        self.hps = hps
        self.activ = get_activ_func(self.hps.activ_unit)
        self.last_active = get_activ_func(self.hps.last_activ_unit)

    def get_shape(self):
        return self.hps.output_shape
    def get_depth(self):
        return convert_shape_to_depth(self.hps.output_shape[0:2])

    def build_decoder(self, z, is_training, concat_dict=None):
        latent_size = self.hps.latent_size
        output_shape = self.hps.output_shape
        activ = self.activ
        last_active = self.last_active
        do_BN = self.hps.do_batch_norm

        if len(output_shape) != 3:
            raise ValueError('output_shape should be (height, width, channel)')
        if type(last_active) is list:
            if len(last_active) != output_shape[2]:
                raise ValueError("# of elements of last_active should be same with that of output_shape")

        return self.decoder(z, latent_size, output_shape, activ=activ, last_active=last_active, 
                            is_training=is_training, concat_dict=concat_dict, do_BN=do_BN)

    # concat_dict: python dictionary of (height, data), it will be concatenated (along the 3rd dim.) with a layer having same height
    # assume height == width
    def decoder_unit(self, layer_name, x, out_filters, concat_dict=None, is_training=True, activ=tf.nn.elu, do_BN=False):
        height, width = x.get_shape().as_list()[1:3] # height and width of input
        assert height == width
        size = height
        
        # concatenate the provided data if available
        if (concat_dict is not None) and size in concat_dict.keys():
            data_concat = concat_dict[size]
            x = tf.concat([x, data_concat], 3) # concatenate with the resized partial input
            do_concat = True
        # increase the resolution
        x = conv_transpose_unit('conv_t_' + layer_name, x, 3, out_filters, 2, 
                                is_training=is_training, activ_func=activ, do_batch_norm=do_BN)
        #additional convolution
        x = conv_unit('conv_shuf_' + layer_name, x, 3, out_filters, 1, 
                      is_training=is_training, activ_func=activ, do_batch_norm=do_BN) # convolution while keeping the resolution
        return x
    
    def decoder_unit_interp(self, layer_name, x, out_filters, concat_dict=None, is_training=True, activ=tf.nn.elu, do_BN=False):
        height, width = x.get_shape().as_list()[1:3] # height and width of input
        assert height == width
        size = height
        size_up = size * 2
        
        x = tf.image.resize_images(x, [size_up, size_up])
        # concatenate the provided data if available
        if (concat_dict is not None) and size_up in concat_dict.keys():
            print(x.shape,concat_dict[size_up].shape)
            data_concat = concat_dict[size_up]
            x = tf.concat([x, data_concat], 3)
        # increase the resolution
        x = conv_unit('conv_shuf_{}_1'.format(layer_name), x, 3, out_filters, 1, 
                                is_training=is_training, activ_func=activ, do_batch_norm=do_BN)
        x = conv_unit('conv_shuf_{}_2'.format(layer_name), x, 3, out_filters, 1, 
                      is_training=is_training, activ_func=activ, do_batch_norm=do_BN) 
        return x
    def decoder(self, z, latent_size, output_shape, activ=tf.nn.elu, last_active=None,  is_training=True, concat_dict=None, do_BN=False):
        depth = convert_shape_to_depth(output_shape[0:2])
        last_layers = output_shape[2]
        #filters = {2:128, 4:128, 8:64, 16:64, 32:64, 64:64, 128:32}
        #filters = {2:512, 4:256, 8:128, 16:64, 32:64, 64:64, 128:32}
        filters = {2:1024, 4:512, 8:256, 16:128, 32:64, 64:64, 128:32}
        #size = 4
        #x = fully_connected('linear_decoder', z, 2048)
        #x = tf.reshape(x, [-1, size, size, 128]) # convert the code to 1x1 image
        size = 1
        x = tf.reshape(z, [-1, size, size, latent_size]) # convert the code to 1x1 image

        # transposed convolutions
        for i in range(depth):
            size = 2<<(i) # target resolution(just for naming the operation)
            out_filters = filters[size]
            #if i < 3:
            #    x = self.decoder_unit(str(size), x, out_filters, concat_dict, is_training, activ, do_BN)
            #else:
            #    x = self.decoder_unit_interp(str(size), x, out_filters, concat_dict, is_training, activ, do_BN)

            x = self.decoder_unit(str(size), x, out_filters, concat_dict, is_training, activ, do_BN)
            #x = self.decoder_unit_interp(str(size), x, out_filters, concat_dict, is_training, activ, do_BN)

            
        # final deconvolution
        #x = self.decoder_unit_interp(str(1<<depth), x, filters[2],  concat_dict, is_training, activ, do_BN)
        
        # final 1x1 convolution for final channels
        if last_layers == 1:
            x = conv_unit('final_conv', x, 1, last_layers, 1, 
                          is_training=is_training, activ_func=last_active, do_batch_norm=False)
        else:
            x = conv_unit('final_conv', x, 1, last_layers, 1, 
                          is_training=is_training, activ_func=None, do_batch_norm=False)
            xs = list(tf.split(x, last_layers, axis=3))
            for i, _ in enumerate(xs):
                xs[i] = last_active[i](xs[i])
            x = tf.concat(xs,axis=3)

        return x
