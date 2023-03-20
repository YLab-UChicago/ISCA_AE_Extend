#############################
# Backward implementation for nf_resblock 
#############################

import tensorflow as tf
from keras import backend as K
from models.backward_nf_layers import BackwardInjectWSConv2D, BackwardSqueezeExcite, BackwardStochasticDepth


class BackwardNFResBlock(tf.keras.Model):
    def __init__(self, in_ch, out_ch, bottleneck_ratio=0.25,
               kernel_size=3, stride=1,
               beta=1.0, alpha=0.2,
               which_conv=BackwardInjectWSConv2D, activation=tf.nn.relu,
               stochdepth_rate=None,
               use_se=False, se_ratio=0.25,
               l_name=None):

        super().__init__()
        self.l_name = l_name
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size = kernel_size
        self.backward_activation = activation
        self.beta, self.alpha = beta, alpha
        self.use_se, self.se_ratio = use_se, se_ratio
        # Bottleneck width
        self.width = int(self.out_ch * bottleneck_ratio)
        self.stride = stride
        # Conv 0 (typically expansion conv)
        self.backward_conv0 = which_conv(self.width, kernel_size=1, padding='SAME',
                                l_name=self.l_name+'conv0')
        # Grouped NxN conv
        self.backward_conv1 = which_conv(self.width, kernel_size=kernel_size, strides=stride,
                                padding='SAME', l_name=self.l_name+'conv1')
        # Conv 2, typically projection conv
        self.backward_conv2 = which_conv(self.out_ch, kernel_size=1, padding='SAME',
                                l_name=self.l_name+'conv2')
        # Use shortcut conv on channel change or downsample.
        self.use_projection = stride > 1 or self.in_ch != self.out_ch
        if self.use_projection:
            self.backward_conv_shortcut = which_conv(self.out_ch, kernel_size=1,
                                          strides=stride, padding='SAME',
                                          l_name=self.l_name+'conv_shortcut')
        # Are we using stochastic depth?
        self._has_stochdepth = (stochdepth_rate is not None and
                                stochdepth_rate > 0. and stochdepth_rate < 1.0)
        if self._has_stochdepth:
            self.backward_stoch_depth = BackwardStochasticDepth(stochdepth_rate)
    
        if self.use_se:
            self.backward_se = BackwardSqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)

        # MOD: no need to add skip_gain to backward
        #self.skip_gain = self.add_weight(name='skip_gain', shape=(), initializer="zeros", trainable=True, dtype=self.dtype)
   

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        grad_shortcut = grad_in

        grad_out = self.alpha * grad_in

        # Do skip_gain first
        grad_skip_gain = tf.reduce_sum(grad_out * layer_inputs[self.l_name + "_out"])
        grad_params.insert(0, grad_skip_gain)

        # Then its se blocks
        grad_out = grad_out * layer_kernels[self.l_name + "_skip_gain"]

        if self._has_stochdepth:
            grad_out = self.backward_stoch_depth(grad_out, layer_inputs[self.l_name + "_binary"])

        if self.use_se:
            grad_se_in = 2 * grad_out * layer_inputs[self.l_name + "_se"]

            grad_se_out, block_grad_params = self.backward_se(grad_se_in, layer_inputs, layer_kernels)
            grad_params = block_grad_params + grad_params

            grad_out = grad_se_out + grad_out * 2 * layer_inputs[self.l_name + "_se_output"]

        # Then do shortcut 
        if self.use_projection:
            grad_shortcut, grad_wt, grad_bias, grad_gain, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv_shortcut(grad_shortcut, layer_inputs[self.l_name + "_conv_shortcut"], layer_kernels[self.l_name + "_conv_shortcut"], inject=inject, inj_args=inj_args)
            grad_params.insert(0, grad_bias)
            grad_params.insert(0, grad_wt)
            grad_params.insert(0, grad_gain)
            bkwd_layer_inputs.update(bkwd_block_inputs)
            bkwd_layer_kernels.update(bkwd_block_kernels)
            bkwd_layer_outputs.update(bkwd_block_outputs)
        else:
            grad_shortcut = grad_shortcut

        # Then it's the other convs
        grad_out, grad_wt, grad_bias, grad_gain, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv2(grad_out, layer_inputs[self.l_name + "_conv2"], layer_kernels[self.l_name + "_conv2"], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        grad_params.insert(0, grad_gain)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_out = self.backward_activation(grad_out, layer_inputs[self.l_name + "_activation_2"])

        grad_out, grad_wt, grad_bias, grad_gain, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv1(grad_out, layer_inputs[self.l_name + "_conv1"], layer_kernels[self.l_name + "_conv1"], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        grad_params.insert(0, grad_gain)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_out = self.backward_activation(grad_out, layer_inputs[self.l_name + "_activation_1"])

        grad_out, grad_wt, grad_bias, grad_gain, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv0(grad_out, layer_inputs[self.l_name + "_conv0"], layer_kernels[self.l_name + "_conv0"], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        grad_params.insert(0, grad_gain)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        if self.use_projection:
            grad_out = grad_out + grad_shortcut
        grad_out = grad_out * self.beta

        grad_out = self.backward_activation(grad_out, layer_inputs[self.l_name + "_activation_0"])

        if not self.use_projection:
            grad_out = grad_out + grad_shortcut

        return grad_out, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs





