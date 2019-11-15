import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

# Convert an input_shape to n, which is the integer satisfying 2^n = height
# Input_shape: (height, width), list or tuple, length 2
# Assume height == width
def convert_shape_to_depth(input_shape):
    height, width = input_shape
    assert height == width # Currently, height and weight should be same
    n = [n for n in range(1,11) if 1<<n == height] # 1<<n means 2^n
    if not n: # if there is no n satisfying the size
        raise ValueError("Incompatible conversion from resolution to depth, resolution={}".format(height))
    depth = n[0]# list containing a single number -> number
    return depth

def batch_norm_contrib(name, x, is_training):
    x = tf.contrib.layers.batch_norm(x, decay=0.9,
                                        center=True, scale=True,
                                        fused=True,
                                        is_training=is_training,
                                        scope=name)
    return x

def conv(name, x, filter_size, out_filters, strides, dilations=None):
    filter_size = _kernel(filter_size)
    strides = _stride(strides)
    if dilations is None:
        dilations = [1,1,1,1]
    else:
        dilations = _stride(dilations) 
    with tf.variable_scope(name):
        kernel = tf.get_variable(
                'DW', [filter_size[0], filter_size[1], x.get_shape()[3], out_filters],
                tf.float32,
                initializer=tf.variance_scaling_initializer(scale=2.0))
    return tf.nn.conv2d(x, kernel, strides, padding='SAME', dilations=dilations)

def conv_unit(name, x, filter_size, out_filters, strides, do_batch_norm=True, is_training=True, activ_func=None ):
    with tf.variable_scope(name):
        x = conv('conv', x, filter_size, out_filters, strides)
        if do_batch_norm:
            x = batch_norm_contrib('bn', x, is_training)
        if activ_func:
            x = activ_func(x)
    return x

def conv_unit_layernorm(name, x, filter_size, out_filters, strides, is_training=True, activ_func=None ):
    with tf.variable_scope(name):
        x = conv('conv', x, filter_size, out_filters, strides)
        x = tf.contrib.layers.layer_norm(x,activation_fn=activ_func)
    return x

def conv_dilated_unit(name, x, filter_size, out_filters, dilations, do_batch_norm=True, is_training=True, activ_func=None ):
    with tf.variable_scope(name):
        x = conv('conv_dilated', x, filter_size, out_filters, strides=[1,1,1,1], dilations=dilations)
        if do_batch_norm:
            x = batch_norm_contrib('bn', x, is_training)
        if activ_func:
            x = activ_func(x)
    return x
    
def global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

def fully_connected(name, x, out_dim):
    x_shape = x.get_shape()
    num_units = int(np.prod(x_shape[1:]))
    with tf.variable_scope(name):
        x = tf.reshape(x, [-1, num_units]) # Flatten
        w = tf.get_variable(
                'DW', [x.get_shape()[1], out_dim],
                #initializer=tf.uniform_unit_scaling_initializer(factor=1.0)) 
                initializer=tf.variance_scaling_initializer(scale=1.0)) 
        b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, w, b) 
    return x 

# from pretty tensor implementation
def conv_transpose(name, x, filter_size, out_filters, strides, edges="SAME"):
    x_shape = x.get_shape()
    filter_size = _kernel(filter_size)
    strides = _stride(strides)
    with tf.variable_scope(name):
        # make a kernel with size [height, width, output_channels, in_channels]
        kernel_size = [filter_size[0], filter_size[1], out_filters, x_shape[3]]
        n = filter_size[0] * filter_size[1] * out_filters
        kernel = tf.get_variable(
                    'DW', kernel_size, tf.float32,
                    initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
    
        # calculate the output size
        input_height = x_shape[1]
        input_width = x_shape[2]
        out_rows, out_cols = get2d_deconv_output_size(x_shape[1], x_shape[2], 
                               filter_size[0], filter_size[1], strides[1], strides[2], edges)

        #output_shape = [int(x_shape[0]), out_rows, out_cols, out_filters]
        output_shape = tf.concat( [tf.shape(x)[:1], tf.constant([out_rows, out_cols, out_filters])], 0 )
        y = tf.nn.conv2d_transpose(x, kernel, output_shape, strides, edges)
        
    return y

def conv_transpose_unit(name, x, filter_size, out_filters, strides, edges="SAME", do_batch_norm=True, is_training=True, activ_func=None ):
    with tf.variable_scope(name):
        x = conv_transpose('conv_trans', x, filter_size, out_filters, strides)
        if do_batch_norm:
            x = batch_norm_contrib('bn', x, is_training)
        if activ_func:
            x = activ_func(x)
    return x

#TODO: add the reference
def get2d_deconv_output_size(input_height, input_width, filter_height,
                           filter_width, row_stride, col_stride, padding_type):
    """Returns the number of rows and columns in a convolution/pooling output."""
    input_height = tensor_shape.as_dimension(input_height)
    input_width = tensor_shape.as_dimension(input_width)
    filter_height = tensor_shape.as_dimension(filter_height)
    filter_width = tensor_shape.as_dimension(filter_width)
    row_stride = int(row_stride)
    col_stride = int(col_stride)

    # Compute number of rows in the output, based on the padding.
    if input_height.value is None or filter_height.value is None:
      out_rows = None
    elif padding_type == "VALID":
      out_rows = (input_height.value - 1) * row_stride + filter_height.value
    elif padding_type == "SAME":
      out_rows = input_height.value * row_stride
    else:
      raise ValueError("Invalid value for padding: %r" % padding_type)

    # Compute number of columns in the output, based on the padding.
    if input_width.value is None or filter_width.value is None:
      out_cols = None
    elif padding_type == "VALID":
      out_cols = (input_width.value - 1) * col_stride + filter_width.value
    elif padding_type == "SAME":
      out_cols = input_width.value * col_stride

    return out_rows, out_cols

def _kernel(kernel_spec):
  """Expands the kernel spec into a length 2 list.

  Args:
    kernel_spec: An integer or a length 1 or 2 sequence that is expanded to a
      list.
  Returns:
    A length 2 list.
  """
  if isinstance(kernel_spec, int):
    return [kernel_spec, kernel_spec]
  elif len(kernel_spec) == 1:
    return [kernel_spec[0], kernel_spec[0]]
  else:
    assert len(kernel_spec) == 2
    return kernel_spec


def _stride(stride_spec):
  """Expands the stride spec into a length 4 list.

  Args:
    stride_spec: None, an integer or a length 1, 2, or 4 sequence.
  Returns:
    A length 4 list.
  """
  if stride_spec is None:
    return [1, 1, 1, 1]
  elif isinstance(stride_spec, int):
    return [1, stride_spec, stride_spec, 1]
  elif len(stride_spec) == 1:
    return [1, stride_spec[0], stride_spec[0], 1]
  elif len(stride_spec) == 2:
    return [1, stride_spec[0], stride_spec[1], 1]
  else:
    assert len(stride_spec) == 4
    return stride_spec

# TODO: below codes should be corrected
"""
def one_hot(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat

def residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
    if activate_before_residual:
        with tf.variable_scope('shared_activation'):
            x = self._batch_norm_op('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = self._batch_norm_op('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
        x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
        x = self._batch_norm_op('bn2', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad( orig_x, [[0, 0], [0, 0], [0, 0],
                                          [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

def bottleneck_residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
    if activate_before_residual:
        with tf.variable_scope('common_bn_relu'):
            x = self._batch_norm_op('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            orig_x = x
    else:
        with tf.variable_scope('residual_bn_relu'):
            orig_x = x
            x = self._batch_norm_op('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
        x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
        x = self._batch_norm_op('bn2', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
        x = self._batch_norm_op('bn3', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
        x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

def decay(self):
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            costs.append(tf.nn.l2_loss(var))

    return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))


def relu(self, x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


    
def global_max_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_max(x, [1, 2])
"""
