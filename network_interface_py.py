try: # Python 3
    import xmlrpc.client as rc
except ImportError: # Python 2
    import xmlrpclib as rc
import time

import numpy as np

import h5py # for emulator

class DataFromHDF5(object):
    def normalize(self, data):
        return data / np.fabs(data).max()
    def adjust_contrast(self, data, contrast=None):
        if contrast is None:
            contrast = self.contrast
        return np.clip(contrast*data, -1.0, 1.0)

    def __init__(self, fpath, shape_out=(128,128), data_idx=1, var_name='data', negative_gain=False, upside_down=False, contrast=1.0):
        with h5py.File(fpath,'r') as f:
                data_all = np.array(f[var_name]).astype('float32')
        #check where the data is single or not
        if data_all.size == np.prod(shape_out):
            self.data = data_all.reshape(shape_out)
        else:
            num_data = data_all.shape[0]
            assert  data_idx < num_data
            self.data = data_all[data_idx].reshape(shape_out)
        if negative_gain:
            self.data = -1.0 * self.data
        if upside_down:
            self.data = self.data[::-1]
        self.data = self.normalize(self.data)
        self.data = self.adjust_contrast(self.data, contrast)
        self.contrast = contrast

    def do_experiment_on_grid_and_wait(self, v1_from=0, v1_to=None, v2_from=0, v2_to=None, stride=1):
        if type(stride) is tuple:
            stride_v, stride_h = stride
        elif np.isscalar(stride):
            stride_v = stride_h = stride
        else:
            raise ValueError('Not supported stride')
        index = (slice(v2_from,v2_to,stride_v), slice(v1_from,v1_to,stride_h))
        mask = np.zeros_like(self.data)
        mask[index] = 1.0

        return self.data[index], mask, index

    def do_single_point_rowcol(self,row,col):
        data = self.data[row, col]
        return data

    def do_experiment_mask_pointwise(self, mask, settletime=None):
        self.mask = mask
        if settletime is not None:
            self.settletime = settletime

        resolution = self.data.shape[0]
        data = np.zeros((resolution, resolution))

        for col in range(resolution):
            reverse=False
            if reverse:
                for row in range(resolution-1,-1,-1):
                    if mask[row,col] != 0.0:
                        data[row,col] = self.do_single_point_rowcol(row,col)
            else:
                for row in range(resolution):
                    if mask[row,col] != 0.0:
                        data[row,col] = self.do_single_point_rowcol(row,col)

        data = self.adjust_contrast(data) # scaling
        return data


class DataFromNPY(DataFromHDF5):
    def __init__(self, fpath, shape_out=(128,128), var_name='data', negative_gain=False, upside_down=False, contrast=1.0):
        self.data = np.load(fpath).reshape(shape_out)
        if negative_gain:
            self.data = -1.0 * self.data
        if upside_down:
            self.data = self.data[::-1]
        self.data = self.normalize(self.data)
        self.data = self.adjust_contrast(self.data, contrast)
        self.contrast = contrast
