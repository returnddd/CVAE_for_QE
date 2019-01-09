import os
import time
import datetime
import numpy as np
import tensorflow as tf

from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import pickle # to load model definition
from CVAE_type1 import CVAE_type1
from CVAE_type2 import CVAE_type2
from CVAE_contextloss_model import CVAE_contextloss

from acq_func_gpu import AcquisitionFunc_gpu

from resol_control import ResolutionControl_basic, ResolutionControl_double

import sampling

from network_interface_py import DataFromHDF5, DataFromNPY


flags = tf.flags
#flags.DEFINE_string("file_name","CVAE_type2_128_latent100sim_epoch_700_wide2_aug2", "File name for data")
flags.DEFINE_string("file_name","CVAE_type2_128_latent100mixed_epoch_700_more_data_wide2_aug2", "File name for data")
flags.DEFINE_string("device", "/gpu:0", "Compute device.")
flags.DEFINE_string("save_name", 'default', "save directory")

flags.DEFINE_boolean("allow_soft_placement", True, "Soft device placement.")
FLAGS = flags.FLAGS

########################
# Functions for plotting
########################       

def plot_100_recon(predicted, input_shape, plot_min, plot_max, fpath=None):
    assert predicted.shape[0] >= 100
    
    plt.figure(figsize=(35,35))
    for i in range(1,101):
        plt.subplot(10, 10, i)
        plt.imshow(predicted[i-1].reshape(input_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap='seismic')
        plt.xticks([])
        plt.yticks([])
    if fpath is None:
        plt.show()
    else:
        plt.savefig(fpath)
        
        
def plot_with_mask(data, mask, vmin=-1.0, vmax=1.0, origin="lower", cmap="seismic"):
    cmap = plt.cm.get_cmap(cmap) # string to object
    colors = Normalize(vmin, vmax, clip=True)(data)
    colors = cmap(colors)
    colors[...,0:3] = colors[...,0:3] * mask[...,np.newaxis]
    plt.imshow(colors,origin="lower")
    plt.xticks([])
    plt.yticks([])

def plot_every_step(i, obs_data, obs_mask, pic_shape, x_best_guess, prob, batch_size, plot_min, plot_max, dir_save):
    #plot the data and the best guess
    #plt.clf()
    plt.figure(1)
    plt.clf()
    plt.subplot(1, 2, 1)
    #plt.imshow(obs_data.reshape(pic_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")
    #plot with unobserved area black
    plot_with_mask(obs_data.reshape(pic_shape), obs_mask.reshape(pic_shape), 
                   vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")
    
    plt.title("Data")
    #plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(x_best_guess.reshape(pic_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")
    plt.title("Best guess")
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    #plt.show()
    plt.savefig(dir_save+"/best_guess_{}.png".format(i))
    
    ##
    #plt.figure(2)
    #plt.clf()
    #plt.plot(np.arange(1,batch_size+1), prob, 'ro')
    #plt.savefig("results/decision/prob_{}.png".format(i))

    plt.close('all')

########################
# Functions for loading models
########################       

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

def load_Unet(hps, shape_data_in, sess, model_folder, file_name=None):
    if file_name == "":
        file_name = None
    # unet in CPU
    #with tf.device('/cpu:0'):
    net = Unet(hps, shape_data_in, test_mode=True)
    net.load(sess, model_folder, file_name)
    print('Unet loaded')
    return net

########################
# Functions for observation acquisition and update
########################       

# overwrite a new patch except previously observed locations
def replace_patch(obs_data_mask, new_obs, patch_size, row, col):
    #obs_data_mask: (height,width,2)
    #new_obs: (height,width,1)
    
    height = obs_data_mask.shape[0]
    width = obs_data_mask.shape[1]
    
    row_start = row*patch_size
    row_end = (row+1)*patch_size
    
    col_start = col*patch_size
    col_end = (col+1)*patch_size
    
    observed_data = np.copy(obs_data_mask[...,0])
    new_location = observed_data[row_start:row_end, col_start:col_end] # not copy, just a view
    new_patch = new_obs.reshape(height,width)[row_start:row_end, col_start:col_end]
    
    observed_mask = np.copy(obs_data_mask[...,1])
    # Do not override existing data
    np.copyto( new_location, new_patch, where=observed_mask[row_start:row_end, col_start:col_end] == 0.0 ) # in-place copy
    
    observed_mask[row_start:row_end, col_start:col_end] = 1.0

    observed_new = np.stack([observed_data, observed_mask], axis=-1)
    
    return observed_new

def get_observation_mask(mask, con, settletime=None): # None -> default
    obs = con.do_experiment_mask_pointwise(mask, settletime)
    return obs

########################
# Functions for sampling and calculating the posterior
########################       
    
def log_sum_exp(logvals,axis=None):
    #logvals: 1D assumed
    assert logvals.ndim == 1
    maxidx = np.argmax(logvals)
    adjust = logvals[maxidx]
    exp_terms = np.exp(logvals-adjust)
    exp_terms[maxidx] = 0.0 # remove 1.0, and use log1p for better numerical stability
    return adjust + np.log1p(np.sum(exp_terms, axis=None))

class Convert_to_likelihood(object):
    def __init__(self, dist_divider=1.0):
        self.dist_divider = dist_divider
    def log_L(self, dist):
        return -dist/self.dist_divider
    def exp_adjusted(self, log_L):
        log_L_ = log_L - np.amax(log_L) # for numerical stability (to prevent underflow)
        likelihood = np.exp(log_L_)
        return likelihood
    def __call__(self, dist):
        log_likelihood = self.log_L(dist)
        likelihood = self.exp_adjusted(log_likelihood)
        return likelihood

class Posterior(sampling.Likelihood):
    def __init__(self, net, sess, y0, obs_data_mask, distance_to_likelihood):
        self.net = net
        self.sess = sess
        self.y0 = y0
        self.observed = obs_data_mask[np.newaxis,...,:1]
        self.mask = obs_data_mask[np.newaxis,...,1:]
        self.distance_to_likelihood = distance_to_likelihood

    def reconstruct(self, z, log_mode=False, normalize=True):
        recon, distance = self.net.test_z_with_observed(self.sess,z,self.y0,self.observed,self.mask)
        distance = distance.astype(np.float32)

        log_L = self.distance_to_likelihood.log_L(distance)
        z_log_prior, _ = self.net.get_prior_density(self.sess, z) # it returns [log_density, density]
        log_posterior = log_L + z_log_prior
        logZ = log_sum_exp(log_posterior)
        if normalize is True:
            log_posterior = log_posterior - logZ # Normalize in log-space

        prob = log_posterior if log_mode else np.exp(log_posterior)
        Z = logZ if log_mode else np.exp(logZ)
        return prob, Z, recon

    def __call__(self, z, log_mode=False, normalize=True):
        prob, Z, recon = self.reconstruct(z,log_mode=log_mode,normalize=normalize)
        return prob

def update_log_prob_mask(obs, newobs_mask, recon, log_prob, distance_to_likelihood):
    dist_func = lambda x : np.sum(np.fabs(x)) # L1 distance function
    num_recon = recon.shape[0]
    assert num_recon == log_prob.size
    new_dist = np.zeros(num_recon)

    bool_mask = newobs_mask != 0.0

    for i in range(num_recon):
        recon_i = recon[i,...,0]
        diff = recon_i[bool_mask] - obs[bool_mask]
        new_dist[i] = dist_func(diff)
    log_prob = log_prob + distance_to_likelihood(new_dist) #the function should return log-likelihood
    #normalize
    log_prob = log_prob - log_sum_exp(log_prob)
    return log_prob

def resample(z, likelihood_calc):
    # input: z, likelihood_calc
    # output: z, log_posterior, recon, log_weight_samples

    move_big = sampling.Gaussian_proposal_move(np.square(0.5))
    mcmc_big = sampling.MH_MCMC(move_big, likelihood_calc, log_mode=True)
    #move_small = sampling.Gaussian_proposal_move(np.square(0.1))
    #mcmc_small = sampling.MH_MCMC(move_small, likelihood_calc, log_mode=True)

    start_time = time.time()
    num_step = 400
    z_after_mcmc = mcmc_big(z, num_step=num_step)
    #z_after_mcmc = mcmc_small(z, num_step=100)
    elapsed_time = time.time() - start_time
    print('Time for sampling: ', elapsed_time)
    z = z_after_mcmc

    log_posterior, logZ_posterior, recon = likelihood_calc.reconstruct(z,log_mode=True) # generate predictions

    # Weights are equal right after resampling
    # (Approximate the posterior distribution of z with equally weighted samples, see the paper)
    batch_size = z.shape[0]
    weight_samples = np.ones(batch_size)
    weight_samples = weight_samples / np.sum(weight_samples)
    log_weight_samples = np.log(weight_samples)
    return z, log_posterior, recon, log_weight_samples

# Main loop
def DOE(sess, con, net, batch_size, input_shape, pic_shape, plot_min=-1, plot_max=1, upside_down=False, save_name="default",batch_mode=False, max_obs=None, unet=None):

    # model storage
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/decision'):
        os.makedirs('results/decision')
    if save_name == "default":
        save_name = '{date:%Y_%m_%d_%H_%M}'.format( date=datetime.datetime.now() )
    dir_save = os.path.join('results/decision',save_name)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)

    # Construct initial input
    log_time = []
    start_time = time.time()
    stride = 16
    y0, obs_mask, data_index = con.do_experiment_on_grid_and_wait(stride=stride) # initial measurement
    obs_data = np.zeros(pic_shape) # full resolution, empty space (pic_shape = (rows, cols))
    obs_data[data_index] = y0 # assign the initial measurement in the full resolution dataspace
    elapsed_time = time.time() - start_time
    log_time.append(elapsed_time)

    obs_data_mask = np.stack([obs_data, obs_mask], axis=2) # (rows, cols, 2)
    obs_data_mask = obs_data_mask.astype(np.float32)
    
    patch_size = 1 # 1 for pixelwise, only 1 is tested
    num_row = pic_shape[0] // patch_size
    num_col = pic_shape[1] // patch_size
    num_pixels = pic_shape[0] * pic_shape[1]
    
    image_set = list()
    mask_set = list()
    prob_set = list()

    y0 = y0.reshape(input_shape) # to be fed to the reconstructio network
    z = net.generate_z(sess, y0) # sample from the prior distribution of z(z of type2 model does not depend on y0, but type1 does)

    # Define likelihood, acquisition function
    dist_scale = 1.0
    distance_to_likelihood = Convert_to_likelihood(dist_scale) # distance -> likelihood functions
    pred_shape = (batch_size,) + pic_shape + (1,)
    acq = AcquisitionFunc_gpu(sess, pred_shape, patch_size, num_row, num_col, scale_L=dist_scale)

    # Set initial observations to 'visited'
    num_obs = int(np.sum(obs_mask))
    rows_visited, cols_visited = obs_mask.nonzero()
    rows_visited, cols_visited = rows_visited.tolist(), cols_visited.tolist()
    row_col_list = list(zip(rows_visited, cols_visited))
    acq.check_visited(row_col_list)

    # Define when to conduct resampling
    mcmc_idxs = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    batch_decision_idxs = mcmc_idxs # batch decision idxs = mcmc_idxs
    # Define when to save and plot the current state
    num_jumps = 64
    jump = num_pixels // num_jumps
    save_idxs = np.arange(jump,num_pixels+1,jump)
    save_idxs = np.append(save_idxs, [i for i in mcmc_idxs if i not in save_idxs])
    save_idxs.sort()

    if max_obs is None:
        max_obs = num_pixels

    # For multi-resolution version
    #resol_ctrl = ResolutionControl_double(pic_shape) # starts with low res. -> full res.
    resol_ctrl = ResolutionControl_basic(pic_shape) # do nothing with the measurement resolution

    while int(np.sum(obs_mask)) <= max_obs:
        num_obs = int(np.sum(obs_mask))
        do_mcmc = num_obs in mcmc_idxs
        do_draw_and_save = num_obs in save_idxs

        if do_mcmc:
            # MCMC move during experiment
            # MCMC move settings
            likelihood_calc = Posterior(net, sess, y0, obs_data_mask, distance_to_likelihood) # function that returns likelihood
            z, log_posterior, recon, log_weight_samples = resample(z, likelihood_calc)

            # Save and plot new reconstructions
            recon.dump(dir_save+'/recon_{}.dat'.format(num_obs))
            plot_100_recon(recon, pic_shape, plot_min, plot_max, fpath=dir_save+"/100_recon_{}".format(num_obs))

        prob = np.exp(log_posterior)

        max_idx = np.argmax(prob)
        x_best_guess = recon[max_idx]

        if do_draw_and_save:
            obs_data.dump(dir_save+'/img_{}.dat'.format(num_obs))
            obs_mask.dump(dir_save+'/mask_{}.dat'.format(num_obs))
            plot_every_step(num_obs, obs_data, obs_mask, pic_shape, x_best_guess, prob, batch_size, plot_min, plot_max, dir_save)

            image_set.append(np.copy(obs_data))
            mask_set.append(np.copy(obs_mask))
            prob_set.append(prob)

        if num_obs >= max_obs:
            break
        
        #### calculate an acquisition map
        start_time = time.time()
        score = acq.get_score(obs_data_mask, recon, log_weight_samples,0.0) #basic

        ### code for decision resolution ###
        #measure the error on the current decision grid
        est_err = estimate_error(recon, obs_mask, mask_valid=resol_ctrl.mask_valid, log_prob=log_weight_samples)
        if est_err < 0.05: # error thresold for increasing resolution
            resol_ctrl.increase()
        #count the number of unobserved locations on the current decision resolution
        num_unobs_decision = resol_ctrl.get_num_unobs(obs_mask)

        #### choose next measurement
        if batch_mode is False:
            #pointwise selection
            num_obs_next = 1
        else:
            #batch selection
            next_point = batch_decision_idxs[batch_decision_idxs>num_obs].min()
            next_idx = np.where(batch_decision_idxs==next_point)[0]
            #print('Next point: {}'.format(next_point))
            num_obs_next = next_point - num_obs
        if num_unobs_decision < num_obs_next:
            resol_ctrl.increase()
        next_mask = acq.choose_next_batch(score, num_obs_next, mask_valid=resol_ctrl.mask_valid)

        elapsed_time = time.time() - start_time
        print('Time for decision: ', elapsed_time)
        
        #### get the next measurement
        start_time = time.time()
        obs = get_observation_mask(next_mask,con) #mask-based implementaion
        elapsed_time = time.time() - start_time
        log_time.append(elapsed_time)

        log_weight_samples=update_log_prob_mask(obs,next_mask,recon,log_weight_samples,distance_to_likelihood)

        if do_draw_and_save:
            #time_map.dump(dir_save+'/timemap_{}.dat'.format(num_obs))
            score.dump(dir_save+'/score_{}.dat'.format(num_obs))
            next_mask.dump(dir_save+'/mask_next_{}.dat'.format(num_obs))

        # add new observations
        obs_mask = obs_mask + next_mask
        obs_data[next_mask != 0.0] = obs[next_mask != 0.0]
        obs_data_mask = np.stack((obs_data,obs_mask), axis=2)

    np.array(log_time).dump(dir_save+'/log_time.dat')

    print("Time for measurement:{}".format(np.sum(log_time)))

    # draw and save full data
    print(len(image_set))
    plt.clf()
    plt.figure(figsize=(25, 25))
    num_save = len(save_idxs)
    num_grid = int(np.ceil(np.sqrt(num_save)))
    for i in range(min(num_save, len(image_set))):
        plt.subplot(num_grid, num_grid, i + 1)
        #plot with unobserved area black
        plot_with_mask(image_set[i].reshape(pic_shape), mask_set[i].reshape(pic_shape), 
                       vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")
    plt.savefig(dir_save+"/decision.png")
    
    plt.clf()
    plt.figure(figsize=(25, 25))
    plt.imshow(image_set[-1].reshape(pic_shape), vmin=plot_min, vmax=plot_max, origin="lower", cmap="seismic")
    plt.savefig(dir_save+"/final.png")
    
    plt.clf()
    plt.figure(figsize=(25, 25))
    for i in range(min(num_save, len(prob_set))):
        plt.subplot(num_grid, num_grid, i + 1)
        plt.plot(np.arange(1,batch_size+1), prob_set[i], 'ro')
    plt.savefig(dir_save+"/prob.png")

def euclidean_dist(x,y):
    return np.sqrt(np.square(x) + np.square(y))

def calc_error_map(img):
    # Derivative is not scaled for simplicity of this demo code
    result = euclidean_dist( ndimage.sobel(img,axis=0), ndimage.sobel(img,axis=1) )
    return result

def estimate_error(recon, mask, mask_valid=None, log_prob=None):
    vals = list()
    for i in range(recon.shape[0]):
        var_map = calc_error_map(recon[i,...,0])
        unobs_mask = mask == 0.0
        if mask_valid is not None:
            unobs_mask = np.logical_and(unobs_mask, mask_valid)
        val = np.sum(var_map[unobs_mask]) # calculate the remaining variance
        var_tot = np.sum(var_map[mask_valid]) if mask_valid is not None else np.sum(var_map)
        val = val / var_tot
        vals.append(val)
    if log_prob is None:
        est_error = np.mean(vals)
    else:
        prob = np.exp(log_prob)
        prob = prob / np.sum(prob)
        est_error = np.sum(prob*np.array(vals))
    return est_error
    
def convert_to_absolute_path(path):
    if not os.path.isabs(path): # if the path is relative, convert to the absolute path
        if path[0] == '~':
            path = os.path.expanduser(path)
        else:
            path = os.path.realpath(path) 
    return path

def main():
    model_folder = "./models" # path for tensorflow reconstruction model
    model_folder = convert_to_absolute_path(model_folder)

    plot_min, plot_max = -1.0, 1.0
    model_name =  'contextloss'
    batch_size=100
    shape_out = (128,128,1)
    shape_data = (-1,) # make a initial measurement (y0) a vector

    with tf.device(FLAGS.device):
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                              gpu_options=gpu_options)) as sess:
            # build a reconstruction model and load variables
            net = load_CVAE(sess, model_folder, model_name, batch_size, FLAGS.file_name)

            pic_shape =  shape_out[:2]

            fpath = 'results/decision/26_grid/img_16384.dat' # test data containing 128x128 array
            con = DataFromNPY(fpath,shape_out=(128,128), upside_down=False, contrast=1.0)
            save_name = FLAGS.save_name
            DOE(sess, con, net, batch_size, shape_data, pic_shape,save_name=save_name,batch_mode=True)

if __name__ == "__main__":
    main()
