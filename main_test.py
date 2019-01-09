import os
import numpy as np
import tensorflow as tf

import pickle # to load model definition
from CVAE_type1 import CVAE_type1
from CVAE_type2 import CVAE_type2
from CVAE_contextloss_model import CVAE_contextloss

import data_feeder_tf

import test_and_plot as test

import matplotlib.pyplot as plt

flags = tf.flags
#flags.DEFINE_string("file_name","CVAE_type2_128_latent100sim_epoch_700_wide2_aug2", "File name for data")
flags.DEFINE_string("file_name","CVAE_type2_128_latent100mixed_epoch_700_more_data_wide2_aug2", "File name for data")
flags.DEFINE_string("device", "/gpu:0", "Compute device.")
flags.DEFINE_boolean("allow_soft_placement", True, "Soft device placement.")
FLAGS = flags.FLAGS

def load_CVAE(sess, model_folder, model_name, batch_size, file_name):
    if model_name == 'type1':
        net = CVAE_type1()
    elif model_name == 'type2':
        net = CVAE_type2()
    elif model_name == 'contextloss':
        net = CVAE_contextloss()
    else:
        raise ValueError('Undefined model')
    
    net.load(sess, model_folder, file_name)
    print('Varaiables are loaded.')
    net.create_testing_nodes(batch_size)
    print('Testing nodes are created.')
    
    return net

def convert_to_absolute_path(path):
    if not os.path.isabs(path): # if the path is relative, convert to the absolute path
        if path[0] == '~':
            path = os.path.expanduser(path)
        else:
            path = os.path.realpath(path) 
    return path

def plot_100(predicted, input_shape, plot_min, plot_max, fpath=None):
    assert predicted.shape[0] >= 100
    
    plt.figure(figsize=(10,10))
    for i in range(1,101):
        plt.subplot(10, 10, i)
        img = predicted[i-1].reshape(input_shape)
        bias = np.mean(img[63:65,:])

        plt.imshow(img-bias, vmin=plot_min, vmax=plot_max, origin="lower", cmap='seismic')
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    if fpath is None:
        plt.show()
    else:
        plt.savefig(fpath)

def main():
    # model storage
    model_folder = './models' # path for tensorflow reconstruction model
    model_folder = convert_to_absolute_path(model_folder)

    plot_min, plot_max = -1.0, 1.0
    model_name =  'contextloss'
    batch_size=100

    with tf.device(FLAGS.device):
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, gpu_options=gpu_options)) as sess:
            # build a model and load variables
            net = load_CVAE(sess, model_folder, model_name, batch_size, FLAGS.file_name)

            # data load for test
            fname_test = []
            fname_test.append('mixed_2345_50000__.h5') # test file name
            # Load data
            #data_folder = './Data' # the folder containing data files
            data_folder = '~/Data/CVAE' # the folder containing data files
            data_folder = convert_to_absolute_path(data_folder)
            print('Data Directory: ', data_folder)
            data_shape = (128,128,1)
            dataset = data_feeder_tf.Data_from_HDF5(data_folder, [], fname_test, data_shape = data_shape) # Data load
            data_arr = dataset.test

            # reconstruction test
            input_gen = subsample_stride16 # function to convert full data to initial measurement
            file_path = test.save_recon_test(sess, data_arr, input_gen, net, batch_size) # save the result into a file
            test.draw_recon_result_from_file(file_path, (128,128), low_res=True, plot_min=plot_min, plot_max=plot_max, cmap="seismic") # plot the result from the file

def subsample_stride16(data):
    data_plot = np.zeros_like(data)
    data_plot[:,::16,::16,0] = data[:,::16,::16,0]
    return data[:,::16,::16,0].reshape((data.shape[0],-1)), data_plot

if __name__ == "__main__":
    main()

