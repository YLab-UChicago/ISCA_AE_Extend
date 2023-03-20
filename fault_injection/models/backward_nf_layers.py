#############################
# Backward layers for nf_layers
#############################

import tensorflow as tf
import math
import sys

from tensorflow import keras
import config
from models.inject_utils import is_input_target, is_weight_target, is_output_target
from models.inject_layers import BackwardInjectDense


def inj_to_tensor(tensor, inj_args):
    if not is_weight_target(inj_args.inj_type):
        tensor_mask = tf.convert_to_tensor(inj_args.inj_mask, tf.float32)
        tensor_delta = tf.convert_to_tensor(inj_args.inj_delta, tf.float32)
        output = tf.add(tf.multiply(tensor, tensor_mask), tensor_delta)
    # If inject to weights, need to manipulate the numpy values
    else:
        output = np.add(np.multiply(inj_args.golden_weights[0], inj_args.inj_mask), inj_args.inj_delta)

    return output

def inj_to_tensor_wt_tensor(tensor, inj_args):
    tensor_mask = tf.convert_to_tensor(inj_args.inj_mask, tf.float32)
    tensor_delta = tf.convert_to_tensor(inj_args.inj_delta, tf.float32)
    output = tf.add(tf.multiply(tensor, tensor_mask), tensor_delta)
    return output

def inject_nn_conv2d(inputs, weights, strides, padding, inj_args):
    if is_input_target(inj_args.inj_type):
        inputs = inj_to_tensor(inputs, inj_args)
    if is_weight_target(inj_args.inj_type):
        weights = inj_to_tensor_wt_tensor(weights, inj_args)
    conv_out = tf.nn.conv2d(inputs, weights, strides=strides, padding=padding)
    if is_output_target(inj_args.inj_type): 
        conv_out = inj_to_tensor(conv_out, inj_args)
    return conv_out


def tf_rot180(w):
    """
    Roate by 180 degrees
    """
    return tf.reverse(w, axis=[0, 1])

def tf_pad_to_full_conv2d(x, w_size, padding, is_input=True):
    """
    Pad x, such that using a 'VALID' convolution in tensorflow is the same
    as using a 'FULL' convolution. See
    http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d
    for description of 'FULL' convolution.
    """
    if padding == "VALID":
        if is_input is False:
            return x
        else:
            return tf.pad(x, [[0, 0],
                      [(w_size - 1) // 2  * 2, (w_size - 1) // 2 * 2],
                      [(w_size - 1) // 2  * 2, (w_size - 1) // 2 * 2],
                      [0, 0]])

    else:
        return tf.pad(x, [[0, 0],
                # If inject to weights, need to manipulate the numpy values[(w_size - 1)//2, (w_size - 1)//2],
                      [(w_size - 1)//2, (w_size - 1)//2],
                      [(w_size - 1)//2, (w_size - 1)//2],
                      [0, 0]])

def tf_NHWC_to_HWIO(out):
    """
    Converts [batch, in_height, in_width, in_channels]
    to       [filter_height, filter_width, in_channels, out_channels]
    """
    return tf.transpose(out, perm=[1, 2, 0, 3])


def tf_HWIO_to_NHWC(out):
    return tf.transpose(out, perm=[2, 0, 1, 3])


def tf_pad_with_stride(x, stride, f_conv, padding):
    if stride != 2:
        print("Error: Currently only support stride 2")
        exit(15)
    if f_conv != 1 and padding != "VALID":
        b,h,w,c = x.get_shape().as_list()
        if b is None:
            b = config.PER_REPLICA_BATCH_SIZE
        z1 = tf.zeros((b,h,w,c))
        med1 = tf.reshape(tf.stack([z1,x], 3), [b, h, w*2, c])
        z2 = tf.zeros((b,h,w*2,c))
        med2 = tf.reshape(tf.stack([z2, med1], 2), [b, h*2, w*2, c])

    else:
        b,h,w,c = x.get_shape().as_list()
        if b is None:
            b = config.PER_REPLICA_BATCH_SIZE
        z1 = tf.zeros((b,h,w,c))
        med1 = tf.reshape(tf.stack([x,z1], 3), [b, h, w*2, c])
        z2 = tf.zeros((b,h,w*2,c))
        med2 = tf.reshape(tf.stack([med1, z2], 2), [b, h*2, w*2, c])
        
    #print("Pad with stride output shape {}".format(med2.shape))
    return med2


class BackwardReLU(tf.keras.layers.Layer):
    def __init__(self, l_name=None):
        super(BackwardReLU, self).__init__()
        self.l_name = l_name

    def call(self, grad_out, inputs):
        return grad_out * tf.cast(tf.greater(inputs, 0), tf.float32)


class BackwardInjectWSConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, groups=1, strides=1, padding='valid', eps = 1e-4, use_bias=True, l_name=None):
        super(BackwardInjectWSConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.groups = groups
        self.padding = padding.upper()
        self.eps = eps
        self.use_bias = use_bias
        self.l_name = l_name

    def backward_standardize_weight(self, grad_in, original_weights, gain, w_max_bitmap):
        # First compute forward 
        b,h,w,c = original_weights.get_shape().as_list()

        if b == None:
            b = config.PER_REPLICA_BATCH_SIZE 
        N = b * h * w
        mu = tf.reduce_mean(original_weights, axis=[0,1,2])

        xmu = original_weights - mu

        sq = tf.square(xmu)
        var = tf.reduce_mean(sq, axis=[0,1,2])

        mvar = N * var 
        maxvar = tf.math.maximum(mvar, tf.convert_to_tensor(self.eps, dtype = self.dtype))

        sqrtvar = tf.sqrt(maxvar)
        ivar = 1. / sqrtvar
        givar = gain * ivar
        xout = xmu * givar

        # Then compute gradients
        dgivar = tf.reduce_sum(grad_in * xmu, axis=[0,1,2])

        #print("Manual xmu is")
        #print(xmu.numpy().flatten()[:10])
        #print("Manual dgivar is ")
        #print(dgivar.numpy().flatten()[:10])

        divar = dgivar * gain

        # Do we need to sum it up? No need, because both dgivar and ivar are only one per kernel
        dgain = dgivar * ivar

        dxmu1 = grad_in * givar

        dsqrtvar = (-1. / tf.square(sqrtvar)) * divar
        dmaxvar = 0.5 * 1. / tf.sqrt(maxvar) * dsqrtvar

        # A bit map that records the positions where mvar > eps, only propagate these gradients
        dmvar = w_max_bitmap * dmaxvar

        dvar = N * dmvar

        dsq = (1./N) * tf.ones_like(grad_in) * dvar
        dxmu2 = 2. * xmu * dsq

        dx1 = dxmu1 + dxmu2
        
        dmu = -1. * tf.reduce_sum(dxmu1+dxmu2, axis=[0,1,2])
        dx2 = (1./N) * tf.ones_like(grad_in) * dmu

        dx = dx1 + dx2
        return dx, dgain 


    def call(self, grad_out, inputs, kernel_list, inject=None, inj_args=None):
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        kernels = kernel_list[1]
        gain = kernel_list[0]
        std_kernels = kernel_list[3]
        w_max_bitmap = kernel_list[4]

        is_target = False
        if inj_args:
            full_name = inj_args.inj_layer
            subs_pos = full_name.rfind('_grad')
            layer_name = full_name[:subs_pos]
            is_target = (layer_name == self.l_name)

        is_input_target = is_target and '_grad_in' in inj_args.inj_layer 
        is_wt_target = is_target and '_grad_wt' in inj_args.inj_layer

        # Split grad_out and kernel
        '''
        grad_out_list = tf.split(grad_out, self.groups, axis=-1)
        kernel_list = tf.split(std_kernels, self.groups, axis=-1)
        manual_grad_in_list = []
        '''

        fmt_inp = tf_pad_to_full_conv2d(grad_out if self.strides == 1 else tf_pad_with_stride(grad_out, 2, self.kernel_size, self.padding), self.kernel_size, self.padding, is_input=True)

        fmt_wt = tf.transpose(tf_rot180(std_kernels), perm=[0,1,3,2])
    
        bkwd_layer_inputs[self.l_name + '_grad_in'] = fmt_inp
        bkwd_layer_kernels[self.l_name + '_grad_in'] = fmt_wt
    
        if not is_input_target:
            manual_grad_in = tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")
        else:
            def no_input_inj(fmt_inp, fmt_wt):
                return tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")
            def input_inj(fmt_inp, fmt_wt, inj_args):
                return inject_nn_conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID", inj_args=inj_args)
    
            manual_grad_in = tf.cond(tf.reduce_all(inject), lambda: input_inj(fmt_inp, fmt_wt, inj_args), lambda: no_input_inj(fmt_inp, fmt_wt))
        bkwd_layer_outputs[self.l_name + '_grad_in'] = manual_grad_in
    
        ####################

        # Split grad_out and inputs

        fmt_inp = tf.transpose(tf_pad_to_full_conv2d(inputs, self.kernel_size, self.padding, is_input=False), perm=[3,1,2,0])
        fmt_wt = tf_NHWC_to_HWIO(grad_out if self.strides == 1 else tf_pad_with_stride(grad_out, self.strides, self.kernel_size, self.padding))
        bkwd_layer_inputs[self.l_name + '_grad_wt'] = fmt_inp
        bkwd_layer_kernels[self.l_name + '_grad_wt'] = fmt_wt
            
        if not is_wt_target or i != target_group:
            manual_direct_grad_wt = tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")
        else:
            def no_wt_inj(fmt_inp, fmt_wt):
                return tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")
    
            def wt_inj(fmt_inp, fmt_wt, inj_args):
                return inject_nn_conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID", inj_args=inj_args)
    
            manual_direct_grad_wt = tf.cond(tf.reduce_all(inject), lambda: wt_inj(fmt_inp, fmt_wt, inj_args), lambda: no_wt_inj(fmt_inp, fmt_wt))
        bkwd_layer_outputs[self.l_name + '_grad_wt'] = manual_direct_grad_wt

        manual_grad_wt = tf_NHWC_to_HWIO(manual_direct_grad_wt)
        manual_grad_std_wt = manual_grad_wt

        # Notice this is not the final manual_grad_wt!
        manual_grad_wt, manual_grad_gain = self.backward_standardize_weight(manual_grad_wt, kernels, gain, w_max_bitmap)

        #print("Manual grad for weight {}".format(manual_grad_wt.shape))

        if self.use_bias:
            manual_grad_bias = tf.reduce_sum(grad_out, axis=[0,1,2])
        else:
            manual_grad_bias = None

        #return manual_grad_in, manual_grad_std_wt, manual_grad_wt, manual_grad_bias, manual_grad_gain, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs
        return manual_grad_in, manual_grad_wt, manual_grad_bias, manual_grad_gain, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs



class BackwardGeLU(tf.keras.layers.Layer):
    def __init__(self):
        super(BackwardGeLU, self).__init__()

    def call(self, grad_out, inputs):
        # The secret value added to gelu
        secret = 1.7015043497085571
        phi = (1 + tf.math.erf(inputs / tf.sqrt(2.0))) / 2
        pdf = tf.exp(-1 * tf.square(inputs) / 2) / tf.sqrt(2 * tf.constant(math.pi))
        return grad_out * secret * (phi + inputs * pdf)


class BackwardSigmoid(tf.keras.layers.Layer):
    def __init__(self, l_name=None):
        super(BackwardSigmoid, self).__init__()
        self.l_name = l_name

    def call(self, grad_out, inputs):
        sig_out = tf.nn.sigmoid(inputs)
        return grad_out * sig_out * (1 - sig_out)



class BackwardSqueezeExcite(tf.keras.layers.Layer):

    def __init__(self, in_ch, out_ch, se_ratio = 0.5, hidden_ch = None, activation = BackwardReLU, name = None, l_name=None):
        super(BackwardSqueezeExcite, self).__init__(name = name)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.l_name = l_name
        if se_ratio is None:
            if hidden_ch is None: raise ValueError('Must provide one of se_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else: self.hidden_ch = max(1, int(self.in_ch * se_ratio))

        self.backward_activation = activation()
        self.backward_fc0 = BackwardInjectDense(self.hidden_ch, use_bias = True, l_name=self.l_name + "_fc0")
        self.backward_fc1 = BackwardInjectDense(self.out_ch, use_bias = True, l_name=self.l_name + "_fc1")
        self.backward_sigmoid = BackwardSigmoid(l_name  = self.l_name+"_sigmoid")

    def call(self, grad_in, layer_inputs, layer_kernels):
        grad_params = []

        grad_out = self.backward_sigmoid(grad_in[:,0,0], layer_inputs[self.l_name + "_sigmoid"])

        grad_out, grad_wt, grad_bias = self.backward_fc1(grad_out, layer_inputs[self.l_name + "_fc1"], layer_kernels[self.l_name + "_fc1"])
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)

        grad_out = self.backward_activation(grad_out, layer_inputs[self.l_name + "_activation"])
        grad_out, grad_wt, grad_bias = self.backward_fc0(grad_out, layer_inputs[self.l_name + "_fc0"], layer_kernels[self.l_name + "_fc0"])
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)

        # repeat to axis 1 and 2, then divided by the shape of axis 1 and 2
        # h = tf.math.reduce_mean(x, axis = [1, 2])
        shape = layer_inputs[self.l_name].shape
        repeat_shape = [1 for x in shape]
        repeat_shape[1] = shape[1]
        repeat_shape[2] = shape[2]

        grad_out = grad_out[:,None,None]

        grad_out = tf.tile(grad_out, tf.constant(repeat_shape))
        grad_out = grad_out / (shape[1] * shape[2])

        return grad_out, grad_params



class BackwardStochasticDepth(tf.keras.layers.Layer):

    def __init__(self, drop_rate, scale_by_keep = False, name = None):
        super(BackwardStochasticDepth, self).__init__(name = name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def call(self, grad_in, binary_tensor):
        grad_out = grad_in * binary_tensor 

        if self.scale_by_keep:
            keep_prob = 1. - self.drop_rate
            grad_out = grad_out / keep_prob

        return grad_out


