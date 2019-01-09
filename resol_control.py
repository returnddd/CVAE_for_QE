import numpy as np

class ResolutionControl_basic(object):
    def __init__(self, full_shape):
        self.mask_valid = np.full(full_shape, True)

    def increase(self):
        pass # do nothing

    def get_num_unobs(self, obs_mask):
        unobs_mask = obs_mask == 0.0
        return np.sum(np.logical_and(unobs_mask,self.mask_valid))

    def get_mask(self):
        return None


class ResolutionControl_double(ResolutionControl_basic):
    def __init__(self, full_shape):
        self.strides = [8, 4, 2, 1]
        self.idx_stride = 0
        self.full_shape = full_shape
        self.mask_valid = self.get_strided_mask(self.strides[self.idx_stride])

    def increase(self):
        self.idx_stride = min(self.idx_stride + 1, len(self.strides)-1)
        self.mask_valid = self.get_strided_mask(self.strides[self.idx_stride])

    def get_strided_mask(self, stride):
        mask_valid = np.full(self.full_shape, False)
        mask_valid[::stride,::stride] = True
        return mask_valid

    def get_mask(self):
        return self.mask_valid
        

