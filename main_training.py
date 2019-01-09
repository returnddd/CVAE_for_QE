import os
import numpy as np
import tensorflow as tf

from CVAE_type1 import CVAE_type1
from CVAE_type2 import CVAE_type2
from CVAE_contextloss_model import CVAE_contextloss
import data_feeder_tf as feeder
import pickle # to store model definition
import model_builder

import test_and_plot as test

flags = tf.flags
flags.DEFINE_string("model", "contextloss", "Model type")
flags.DEFINE_integer("latent_size", 100, "Length of the  latent vector z.")
flags.DEFINE_integer("epochs", 30, "Maximum number of epochs.")
flags.DEFINE_integer("batch_size", 100, "Mini-batch size for data subsampling.")
flags.DEFINE_float("weight_adv", 0.0, "Weight multiplied to the adversarial loss of the generator")

flags.DEFINE_string("device", "/gpu:0", "Compute device.")
flags.DEFINE_boolean("allow_soft_placement", True, "Soft device placement.")
flags.DEFINE_string("name", "", "model name suffix")
FLAGS = flags.FLAGS

def build_CVAE_net(model_name, hps, encoder, decoder, input_nodes, add_name=""):
    if model_name == 'type1':
        net = CVAE_type1()
    elif model_name == 'type2':
        net = CVAE_type2()
    elif model_name == 'contextloss':
        net = CVAE_contextloss()
    else:
        raise ValueError('Undefined model')
    net.build_net(hps, encoder, decoder, input_nodes, add_name)
    print('Model created.')

    return net

def train_net(net, sess, model_folder):
    sess.run(tf.global_variables_initializer())
    net.train(sess,display_iter=100,weight_adv=FLAGS.weight_adv, freeze_on=False, model_folder=model_folder,save_iter=100000)
    print('Training finished.')
    
    model_filename = os.path.join(model_folder, net.get_name())
    net.save(sess, model_filename)
    print('Model saved.')

def convert_to_absolute_path(path):
    if not os.path.isabs(path): # if the path is relative, convert to the absolute path
        if path[0] == '~':
            path = os.path.expanduser(path)
        else:
            path = os.path.realpath(path) 
    return path

def main():    
    # Load data
    data_folder = '~/Data/CVAE' # the folder containing data files
    data_folder = convert_to_absolute_path(data_folder)
    print('Data Directory: ', data_folder)

    fname_train = [] # python list containing strings of hdf5 file names
    fname_train.append('mixed_2345_50000__.h5')
    print('Training data files:{}'.format(fname_train))

    add_model_name = 'mixed_epoch_{}_{}'.format(FLAGS.epochs,FLAGS.name) # will be added to file names

    shape_out = (128,128,1) # shape of single data, also output shape of the decoder
    # load the whole HDF5 data into memory
    dataset = feeder.Data_from_HDF5(data_folder, fname_train, None, shape_out) # load all data into memory

    # define preprocessing (data augmentation)
    preprocessor = feeder.Prepare_data(shape_out, FLAGS.batch_size)
    
    plot_min, plot_max = -1, 1 # for plotting after training

    #Model construction
    #common settings for encoder and decoder
    activ = 'elu'
    do_batch_norm = True
    shape_data = tuple(preprocessor.shapes[1].as_list()[1:]) # shape of partial data except batch_dim
    last_activation = 'tanh' # for the last activation of a decoder
    hps_vae, encoder, decoder = model_builder.build_model(shape_data, shape_out, 
            FLAGS.latent_size, FLAGS.batch_size, activ, last_activation, do_batch_norm, FLAGS.model)
    
    # model storage
    model_folder = 'models'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        
    with tf.device(FLAGS.device):
        gpu_options = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, gpu_options=gpu_options)) as sess:
            input_nodes = preprocessor.get_next() # pair of (full, partial) data expected
            net = build_CVAE_net(FLAGS.model, hps_vae, encoder, decoder, input_nodes, add_name=add_model_name)
            preprocessor.set_data(sess, dataset.train, transform_type="train", num_epochs=FLAGS.epochs)
            print("epochs: ", FLAGS.epochs)
            train_net(net, sess, model_folder)

if __name__ == "__main__":
    main()
