import numpy as np
import tensorflow as tf

class AcquisitionFunc_gpu(object):
    def __init__(self, sess, pred_shape, patch_size, num_row, num_col, scale_L=1.0, device="/gpu:0"):
        self.pred_shape = pred_shape # (num_pred, height,width,depth)
        self.patch_size = patch_size
        self.num_row, self.num_col = num_row, num_col
        self.scale_L = scale_L
        self.sess = sess

        self.D = self.patch_size * self.patch_size
        self.NXp1 = num_row * num_col

        with tf.variable_scope("acq"):
            self.build_graph()

        self.X = [(row,col) for row in range(num_row) for col in range(num_col)]
        self.num_X = len(self.X)
        self.visit = np.full((self.num_X,), False)

    def check_visited(self, visited_list):
        for visited in visited_list:
            idx = self.X.index(visited)
            self.visit[idx] = True

    def build_graph(self):
        self.pred = tf.placeholder(tf.float32, self.pred_shape, name="prediction")
        self.log_current_prob = tf.placeholder(tf.float32, (self.pred_shape[0],), name="log_current_prob")
        self.logZ = tf.placeholder(tf.float32, [], name="logZ")
        self.observed = tf.placeholder(tf.float32, self.pred_shape[1:], name="observed_mask")

        log_next, log_term, logZ_next = self.get_updated_prob(self.pred, self.observed, self.log_current_prob, self.logZ)

        gain_info = self.logZ - logZ_next - tf.exp(self.log_sum_exp(log_next + tf.log(-log_term)))
        self.score = tf.abs(tf.reduce_sum(gain_info, axis=[-2, -1]))
        print("shape of score tensor")
        print(self.score.get_shape().as_list())
        
    def get_updated_prob(self, pred, observed, log_current_prob, logZ):
        pred = pred * (1.0 - observed)
        if self.patch_size > 1:
            patch_to_depth = tf.space_to_depth(pred, self.patch_size)
        else:
            patch_to_depth = pred
        # M: num_pred, NXp1: # of possible X_{n+1}, D: # of data in a patch
        M_NXp1_D = tf.reshape(patch_to_depth, [-1, self.NXp1, self.D])
        NXp1_M_D = tf.transpose(M_NXp1_D, perm=[1,0,2])
        NXp1_M_1_D = tf.expand_dims(NXp1_M_D, 2)
        NXp1_1_M_D = tf.expand_dims(NXp1_M_D, 1)
        distance = tf.abs(tf.reduce_sum(NXp1_M_1_D - NXp1_1_M_D, axis=3)) #NXp1_M_M, L1 distance function
        log_term = -distance/self.scale_L

        log_next = log_term + (log_current_prob + logZ)
        logZ_next = self.log_sum_exp(log_next)
        log_next = log_next - logZ_next
        next_prob = tf.exp(log_next)

        return log_next, log_term, logZ_next

    def log_sum_exp(self, logvals):
        axis=-1
        #adjust = tf.reduce_max(logvals, axis=axis, keepdims=True)
        adjust = tf.reduce_max(logvals, axis=axis, keepdims=True)
        dummy_0 = tf.zeros_like(adjust)
        all_inf = tf.is_inf(adjust)
        adjust = tf.where(all_inf, dummy_0, adjust)
        expterms = tf.exp(logvals - adjust)

        # without using log1p, which coubld be more instable
        result = adjust + tf.log(tf.reduce_sum(expterms, axis=axis, keepdims=True))

        # using using log1p, not tested 
        #dummy_0 = tf.zeros_like(expterms)
        #expterms = tf.where(expterms==1.0, dummy_0, expterms)
        ##result = adjust + tf.log1p(tf.reduce_sum(expterms, axis=axis, keepdims=True))
        #result = adjust + tf.log1p(tf.reduce_sum(expterms, axis=axis, keepdims=True))


        #dummy_inf = tf.constant(-np.inf, shape=result.get_shape().as_list(), dtype=tf.float32)
        #result = tf.where(all_inf, dummy_inf, result)


        ## set 1 to 0, and apply log1p later instead of log
        #data_shape = logvals.get_shape().as_list()
        #maxidx = tf.reshape(tf.argmax(logvals, axis=axis, output_type=tf.int32), (-1,))
        #ranges = tuple([tf.range(n) for n in data_shape[:-1]])
        #idx_list = list(tf.meshgrid(*ranges, indexing='ij'))
        #idx = tf.stack([tf.reshape(ax,(-1,)) for ax in idx_list] + [maxidx], axis=1)
        #num_entry = np.prod(data_shape[:-1])
        #updated = tf.scatter_nd_update(logvals, idx, [0]*num_entry)
        #with tf.control_dependencies([updated]):
        #    result = adjust + tf.log1p(tf.reduce_sum(updated, axis=axis, keepdims=True))

        return result

    
    def get_score(self, obs_data_mask, predictions, log_current_prob, logZ):
        feed_dict = {self.pred:predictions, self.observed:obs_data_mask[...,1:], self.log_current_prob:log_current_prob, self.logZ:logZ}
        score = self.sess.run(self.score, feed_dict=feed_dict)
        score = score.reshape((self.num_row,self.num_col))
        return score

    def choose_next(self, score, mask_valid=None):
        valid = np.logical_not(self.visit)
        if mask_valid is not None:
            valid = np.logical_and(valid, mask_valid.ravel())
        
        score = score.ravel()
        valid = valid.ravel()

        maxval = np.amax(score*valid)
        max_loc = np.where(np.logical_and(valid, score==maxval))[0]
        idx = np.random.choice(max_loc)
        self.visit[idx] = True
        return self.X[idx] # (row, col) 
    def choose_next_batch(self, score, batch_size, mask_valid=None):
        valid = np.logical_not(self.visit)
        if mask_valid is not None:
            valid = np.logical_and(valid, mask_valid.ravel())

        score = score.ravel()
        valid = valid.ravel()

        bestn_linidxs = np.argsort(-score*valid)[:batch_size]
        bestn_idxs = np.unravel_index(bestn_linidxs, score.shape)

        mask = np.zeros(score.shape)
        mask[bestn_idxs] = 1.0

        mask = mask.reshape((self.num_row,self.num_col))

        self.visit[bestn_linidxs] = True

        return mask

