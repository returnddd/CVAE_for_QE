import os
import numpy as np
#import tensorflow as tf

#from CVAE_model import CVAE
#from CVAE_contextloss_model import CVAE_contextloss

import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

def plot_with_no_obs(data, vmin=-1.0, vmax=1.0, origin="lower", cmap="seismic"):
    #reshape
    resol = int(np.sqrt(data.size)) # infer the resolution from the pixel count
    data = data.reshape((resol,resol))

    cmap = plt.cm.get_cmap(cmap) # string to object
    colors = Normalize(vmin, vmax, clip=True)(data)
    colors = cmap(colors)
    # nan -> no observation
    mask = np.isnan(data)
    for i in range(3):
        colors_new = colors[...,i]
        colors_new[mask] =  0.0 
        colors[...,i] = colors_new
    #colors[mask,:] = 0.0
    plt.imshow(colors,origin="lower")
    plt.xticks([])
    plt.yticks([])
    #plt.imshow(mask,origin="lower")

def draw_recon_result(x_sample, x_plot, recon_list, plot_min, plot_max, img_shape, offset=0, add_name="", cmap="seismic"):
    if not os.path.exists('pics'):
        os.makedirs('pics')

    num_recon_plot = 5 # the number of reconstruction results to be plotted
    num_col = 2+num_recon_plot # 2 for original & input
    num_draw = 10 # the number of test data to be plotted in this figure
    plt.figure(figsize=(35,35))
    for data_idx in range(offset,offset+num_draw):
        i=data_idx - offset
        plt.subplot(num_draw, num_col, num_col*i + 1)
        img = x_sample[data_idx].reshape(img_shape)
        bias = np.mean(img[63:65])
        plt.imshow(img-bias, vmin=plot_min, vmax=plot_max, origin="lower", cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.title("Target")
        #plt.colorbar()
        
        plt.subplot(num_draw, num_col, num_col*i + 2)
        #plt.imshow(x_plot[data_idx].reshape(img_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap=cmap)
        plot_with_no_obs(x_plot[data_idx]-bias, vmin=plot_min, vmax=plot_max, origin="lower", cmap=cmap)
        plt.title("Input")
        #plt.colorbar()
        
        for i_recon in range(num_recon_plot):
            plt.subplot(num_draw, num_col, num_col*i + i_recon+3)
            img = recon_list[i_recon][data_idx].reshape(img_shape)
            bias = np.mean(img[63:65])
            plt.imshow(img-bias, vmin=plot_min, vmax=plot_max, origin="lower", cmap=cmap)
            plt.xticks([])
            plt.yticks([])
            plt.title("Prediction_{}".format(i_recon+1))
            #plt.colorbar()
            
    plt.savefig("pics/pred_{}_{}".format(offset+num_draw, add_name), bbox_inches='tight')
        #plt.show()
    print('plot saved.')

def draw_recon_result_from_file(file_path, img_shape, low_res=True, plot_min=-1.0, plot_max=1.0, cmap="seismic"):
    with np.load(file_path) as data:
        #print(data.files)
        num_recon = data['num_recon']
        x_sample = data['x_sample']
        x_input = data['x_input']
        x_zero_filled = data['x_zero_filled'] # data filled with 0 at unobserved area
        recon_list = data['recon_list']
        batch_size = data['batch_size']

    if low_res is True:
        x_plot = x_input
    else:
        x_plot = x_zero_filled

    num_output_channel = recon_list[0][0].shape[2]

    for channel in range(num_output_channel):
        for i in range(int(batch_size/10)):
            draw_recon_result(x_sample, x_plot, recon_list[...,channel], plot_min, plot_max, img_shape, offset=10*i, cmap=cmap,
            #draw_recon_result((x_sample+1.0)/2.0, (x_plot+1.0)/2.0, (recon_list[...,channel]+1.0)/2.0, plot_min, plot_max, img_shape, offset=10*i, 
                    add_name="_ch{}".format(channel))
    
def draw_all_input_and_recon(file_path, img_shape, cmap='seismic', plot_min=-1.0, plot_max=1.0):
    with np.load(file_path) as data:
        #print(data.files)
        num_recon = data['num_recon']
        x_sample = data['x_sample']
        x_input = data['x_input']
        recon_list = data['recon_list']
    
    recon_plot = recon_list[0]
    num_data = recon_plot.shape[0]
    plt.figure(figsize=(35,35))
    for data_idx in range(min(num_data,100)):
        plt.subplot(10, 10, data_idx+1)
        plt.imshow(recon_plot[data_idx].reshape(img_shape), vmin=plot_min, vmax=plot_max,cmap=cmap)
        plt.title("%d"%(data_idx+1))
    plt.savefig("pics/reconstructed_100", bbox_inches='tight')
    
    plt.figure(figsize=(35,35))
    for data_idx in range(min(num_data,100)):
        plt.subplot(10, 10, data_idx+1)
        plt.imshow(x_sample[data_idx].reshape(img_shape), vmin=plot_min, vmax=plot_max,cmap=cmap)
        plt.title("%d"%(data_idx+1))
    plt.savefig("pics/input_100", bbox_inches='tight')
     
# recon_with_fulldata: True for checking whether the model can replicate the full data when it is known to the model, False for reconstruction with only the initial measurement
def save_recon_test(sess, data_test, input_gen, net, batch_size, num_recon = 100, offset=None, recon_with_fulldata=False):
    # choose a subset of test data
    if offset is None:
        random_idx = np.random.choice(data_test.shape[0], batch_size)
        x_sample = data_test[random_idx]
    else:
        x_sample = data_test[offset:offset+batch_size]

    # input_data: initial measurement, x_zero_filled: full resolution array with only initial measurement (for plotting)
    input_data, x_zero_filled = input_gen(x_sample)
    
    recon_list = []
    for i in range(num_recon):
        if recon_with_fulldata: # feed the full data for reconstruction
            z_mean, z_log_var, reconstructed = net.reconstruct_with_full_data(sess, x_sample, input_data)
        else:
            if i == 0: # first reconstruction is the best guess with the initial measurement
                z_mean, z_log_var, reconstructed = net.reconstruct_best(sess, input_data)[:3] # best guess
            else: # other random guesses with the initial measurement
                _, _, _, reconstructed = net.reconstruct(sess, input_data)[:4] # guess with randomness
        recon_list.append(reconstructed)
    
    print('Test completed.')
    
    if not os.path.exists('results/prediction/'):
        os.makedirs('results/prediction')
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    rel_path = 'results/prediction/{}_recon_result'.format(net.get_name())
    file_path = os.path.join(current_dir,rel_path)
    np.savez(file_path, num_recon=num_recon, x_sample=x_sample, 
             x_input=input_data, x_zero_filled=x_zero_filled, 
             recon_list=recon_list, z_mean=z_mean, z_log_var=z_log_var,
             batch_size = batch_size)
    print('Result saved.')
    
    
    file_path_mean = os.path.join(current_dir,'results/prediction/mean.txt')
    np.savetxt(file_path_mean, z_mean)
    
    file_path_var = os.path.join(current_dir,'results/prediction/log_var.txt')
    np.savetxt(file_path_var, z_log_var)
    
    return file_path + '.npz'

#https://jmetzen.github.io/2015-11-27/vae.html
def draw_2D_latentspace(sess, net, batch_size, img_shape, plot_min, plot_max):
    if not os.path.exists('pics'):
        os.makedirs('pics')
    nx = ny = 20
    bound = 15
    x_values = np.linspace(-bound,bound,nx)
    y_values = np.linspace(-bound,bound,ny)
    
    h, w = img_shape
    canvas = np.empty((h*ny, w*nx))
    for i, yi in enumerate(y_values):
        for j, xj in enumerate(x_values):
            z = np.array([[xj, yi]] * batch_size)
            imgs = net.decode(sess,z)
            canvas[(ny-i-1)*h:(ny-i)*h, j*w:(j+1)*w] = imgs[0].reshape(h,w)
    plt.figure(figsize=(25,25))
    plt.imshow(canvas, origin="upper", vmin=plot_min, vmax=plot_max,cmap="seismic")
    plt.tight_layout()
    plt.savefig("pics/grid_recon", bbox_inches='tight')
