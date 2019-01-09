import os
import numpy as np
import tensorflow as tf
import h5py

class Data_from_HDF5(object):
    # Single file per each label
    def read_from_files(self, folderpath, fname_list, var_name = 'data'):
        data_list = []
        label_list = []
        
        num_label = len(fname_list) # The number of labels = the number of files
        num_data_per_label = []
            
        for index, fname in enumerate(fname_list):
            filepath = os.path.join(folderpath, fname)
            #print(filepath)
            with h5py.File(filepath,'r') as f:
                data_list.append( np.array(f[var_name]).astype('float32') )
                num_data = data_list[-1].shape[0]
                num_data_per_label.append(num_data)
                    
                label_new = np.zeros((num_data,num_label), dtype=np.float32)
                label_new[:,index] = 1.0 # the labels of all data in a single file are same
                label_list.append(label_new)
        return data_list, label_list, num_data_per_label

    def data_reshape_and_concat(self, data_list, label_list, data_shape=None):
        data = np.concatenate(data_list)
        #pre_shape = data.shape
        #data_size = np.prod(pre_shape[1:])
        #data = data.reshape((pre_shape[0],data_size)) # Flatten each data to be 1 dimensional
        num_data = data.shape[0]
        if data_shape is not None:
            if np.prod(data.shape[1:]) != np.prod(data_shape):
                raise ValueError('Wrong data_shape')
        else:
            data_shape = (-1,)
        data = data.reshape((num_data,) + data_shape)
        label = np.concatenate(label_list)
        
        return data, label

    def data_loader(self, folderpath, fname_list, data_shape, var_name='data'):
        data_list, label_list, num_data_per_label = self.read_from_files(folderpath, fname_list, var_name)
        return self.data_reshape_and_concat(data_list, label_list, data_shape)

    def load_all(self, folderpath, fname_list, data_shape):
        if not type(fname_list) == list:
            raise ValueError('Data file names should be a list')
        if len(fname_list) is 0:
            feeder = None
        else:
            data, label = self.data_loader(folderpath, fname_list, data_shape)
        return data

    def remove_nan(self, data):
        for i in range(data.shape[0]):
            if np.any(np.isnan(data[i])):
                data[i,...] = 0.0

    def __init__(self, folderpath, fname_train, fname_test, data_shape = None):
        if fname_train is None or len(fname_train) == 0:
            self.train = None
        else:
            self.train = self.load_all(folderpath, fname_train, data_shape)
            self.remove_nan(self.train)
            assert not np.any(np.isnan(self.train))

        if fname_test is None or len(fname_test) == 0:
            self.test = None
        else:
            self.test = self.load_all(folderpath, fname_test, data_shape)
            self.remove_nan(self.test)
            assert not np.any(np.isnan(self.test))



class GaussianNoise(object):
    def __init__(self, std):
        self.std = std
    def __call__(self, data):
        return data + tf.random_normal(tf.shape(data),stddev=self.std)


class Subsample(object):
    def __init__(self, stride=1, batch_dim=False, add_noise=None, channel=None):
        self.stride = stride
        self.batch_dim = batch_dim
        self.channel = channel
        self.add_noise = add_noise
    #data: (h, w, c) assumed
    def __call__(self, data):
        if self.batch_dim == True:
            sampled = data[:,::self.stride, ::self.stride]
        else:
            sampled = data[::self.stride, ::self.stride]

        if self.channel is not None:
            sampled = sampled[...,self.channel:self.channel+1]
        if self.add_noise is not None:
            sampled = self.add_noise(sampled)

        # reshape the sampled data
        if self.batch_dim == True:
            batch_size = data.get_shape().as_list()[0]
            sampled = tf.contrib.layers.flatten(sampled)
        else:
            sampled = tf.reshape(sampled, [-1])

        return data, sampled

class Transform_QD_1ch(object):
    def __init__(self, stride=1):
        self.stride = stride
        #self.dim_min = 0.7
        self.dim_min = 0.2
        self.bias_std = 0.1
        self.noise_std = 0.05
    def __call__(self, data):
        #transform data (output of a model)
        # diminishing signal
        dim_factor = tf.clip_by_value(tf.random_uniform([])*1.25,0.0,1.0)#1.0 by 20% chance, 0~1 otherwise
        dim_factor = (1.0-self.dim_min)*dim_factor + self.dim_min #1.0 by 50% chance, dim_min~1 otherwise

        resol = data.get_shape().as_list()[1]
        dim_matrix = tf.linspace(dim_factor,1.0,resol)
        dim_matrix = tf.expand_dims(dim_matrix,0) # height dimension to 1 (multiplication applys to width dim)
        data = data * tf.expand_dims(dim_matrix,2) # expand channel dim

        #random contrast
        #data = data * tf.random_uniform([],0.7,1.3)
        data = data * tf.random_uniform([],0.5,1.5)

        #data = data + tf.random_normal([], stddev=self.bias_std) # add bias
        data = data + tf.random_uniform([],-self.bias_std,self.bias_std)
        data = tf.clip_by_value(data,-1.0,1.0)

        #random flip and multiple -1
        rand_flip = tf.random_uniform([])
        data = tf.cond(rand_flip<0.5, lambda: data, lambda: -1.0*data[::-1])

        #generate input of a model
        sampled = data[::self.stride, ::self.stride]
        sampled = sampled + tf.random_normal(tf.shape(sampled),stddev=self.noise_std)
        sampled = tf.reshape(sampled, [-1])
        return data, sampled

class Transform_simple(object):
    def __init__(self, stride=1):
        self.stride = stride
        self.bias_std = 0.1
        self.noise_std = 0.05
    def __call__(self, data):
        data = data * tf.random_uniform([],0.5,1.5) #random contrast
        data = data + tf.random_uniform([],-self.bias_std,self.bias_std) # random bias
        data = tf.clip_by_value(data,-1.0,1.0) # clipping

        #generate input of a model
        sampled = data[::self.stride, ::self.stride]
        sampled = sampled + tf.random_normal(tf.shape(sampled),stddev=self.noise_std)
        sampled = tf.reshape(sampled, [-1])
        return data, sampled

class Identity(object):
    def __init__(self, stride=1):
        self.stride = stride
    def __call__(self, data):
        #generate input of a model
        sampled = data[::self.stride, ::self.stride]
        sampled = tf.reshape(sampled, [-1])
        return data, sampled

class Prepare_data(object):
    def __init__(self, shape_single, batch_size, shuf_buffer_size=50000, num_threads=1):
        with tf.device("/cpu:0"):
            self.placeholder = tf.placeholder(tf.float32, (None,)+shape_single)
            self.dataset = tf.data.Dataset.from_tensor_slices(self.placeholder)
            self.dataset = self.dataset.shuffle(buffer_size=shuf_buffer_size)

            # two types of transformation for training and testing
            transform_test = Subsample(stride=16, batch_dim=False, channel=0)
            self.dataset_test = self.dataset.map(transform_test, num_parallel_calls=num_threads)
            self.dataset_test = self.dataset_test.batch(batch_size)

            transform_train = Transform_QD_1ch(stride=16)
            # transform_train = Transform_simple(stride=16)
            # transform_train = Identity(stride=16)
            self.dataset_train = self.dataset.map(transform_train, num_parallel_calls=num_threads)
            self.dataset_train = self.dataset_train.batch(batch_size)


            assert self.dataset_train.output_types == self.dataset_test.output_types

            self.iterator = tf.data.Iterator.from_structure(self.dataset_train.output_types,
                                                       self.dataset_train.output_shapes)
            self.shapes = self.dataset_train.output_shapes
            print(self.shapes)

    # expected to be called right before building a model
    def get_next(self):
        return self.iterator.get_next()

    # expected to be called right before training or testing
    def set_data(self, sess, arr, transform_type="train", num_epochs=1, num_consumers=1):
        if transform_type == "train":
            dataset = self.dataset_train
        else:
            dataset = self.dataset_test
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.prefetch(buffer_size=num_consumers)
        init_iter_op = self.iterator.make_initializer(dataset)
        sess.run(init_iter_op, feed_dict={self.placeholder: arr})
