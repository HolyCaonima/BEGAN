
from prettytensor.pretty_tensor_class import Phase
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

import numpy as np
import tensorflow as tf
import prettytensor as pt
import tensorflow.contrib.layers as ly

def lrelu(x, leak=0.1):
    ret = tf.maximum(x, leak * x)
    return ret


# from IllustrationGAN
@pt.Register
class minibatch_disc(pt.VarStoreMethod):
    def __call__(self, input_layer, num_kernels, dim_per_kernel=5, name='minibatch_discrim'):
        batch_size = input_layer.shape[0]
        num_features = input_layer.shape[1]
        W = self.variable('W', [num_features, num_kernels*dim_per_kernel],
                          init=tf.contrib.layers.xavier_initializer())
        b = self.variable('b', [num_kernels], init=tf.constant_initializer(0.0))
        activation = tf.matmul(input_layer, W)
        activation = tf.reshape(activation, [batch_size, num_kernels, dim_per_kernel])
        tmp1 = tf.expand_dims(activation, 3)
        tmp2 = tf.transpose(activation, perm=[1,2,0])
        tmp2 = tf.expand_dims(tmp2, 0)
        abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
        f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
        f = f + b
        return f

def depthwise_conv2d_transpose(value, filter, output_shape, strides, padding='SAME', name=None):
    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    return gen_nn_ops.depthwise_conv2d_native_backprop_input(
        input_sizes=output_shape_,
        filter=filter,
        out_backprop=value,
        strides=strides,
        padding=padding,
        name=name)

@ops.RegisterGradient('DepthwiseConv2dNativeBackpropInput')
def _DepthwiseConv2dNativeBackpropInput(op, grad):
    return [None,
            nn_ops.depthwise_conv2d_native_backprop_filter(grad, array_ops.shape(op.inputs[1]),
                                          op.inputs[2], op.get_attr("strides"),
                                          op.get_attr("padding")),
            nn_ops.depthwise_conv2d_native(grad, op.inputs[1], op.get_attr("strides"),
                          op.get_attr("padding"))]

def upsample_bilinear_2x(input):
    output_shape = input.get_shape().as_list()
    output_shape[1] = output_shape[1]*2
    output_shape[2] = output_shape[2]*2
    
    f = [[0.25, 0.5, 0.25],
         [0.5, 1, 0.5],
         [0.25, 0.5, 0.25]]
    f = np.array(f)
    f = np.expand_dims(f, 2)
    f = np.expand_dims(f, 3)
    f = np.tile(f, (1, 1, output_shape[3], 1))
    f = f.astype(np.float32)
    return depthwise_conv2d_transpose(input, f, output_shape, [1,2,2,1])

@pt.Register
class upsample2x(pt.VarStoreMethod):
    def __call__(self, input_layer, name="updample2x"):
        with tf.variable_scope(name):
            return upsample_bilinear_2x(input_layer)

@pt.Register
class upsample_conv(pt.VarStoreMethod):
    def __call__(self, input_layer, kernel, depth, padding='SAME', name="upsample_conv"):
        with tf.variable_scope(name):
            upsampled = upsample_bilinear_2x(input_layer)
            w = self.variable('w', [kernel, kernel, input_layer.shape[-1], depth],
                              init=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(upsampled, w, strides=[1, 1, 1, 1], padding=padding)

            biases = self.variable('biases', [depth], init=tf.constant_initializer(0.0))
            return input_layer.with_tensor(tf.nn.bias_add(conv, biases), parameters=self.vars)