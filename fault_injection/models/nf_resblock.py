##############################
# nf_resblock file
##############################

import tensorflow as tf
from keras import backend as K
from models.nf_layers import InjectWSConv2D, SqueezeExcite, StochasticDepth

class NFResBlock(tf.keras.Model):
    def __init__(self, in_ch, out_ch, seed, bottleneck_ratio=0.25,
               kernel_size=3, stride=1,
               beta=1.0, alpha=0.2,
               which_conv=InjectWSConv2D, activation=tf.nn.relu,
               stochdepth_rate=None,
               use_se=False, se_ratio=0.5,
               name=None,
               l_name=None):

        super().__init__(name=name)
        self.l_name = l_name
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel_size = kernel_size
        self.activation = activation
        self.beta, self.alpha = beta, alpha
        #print("Alpha is {}".format(self.alpha))
        self.use_se, self.se_ratio = use_se, se_ratio
        # Bottleneck width
        self.width = int(self.out_ch * bottleneck_ratio)
        self.stride = stride
        # Conv 0 (typically expansion conv)
        self.conv0 = which_conv(seed, self.width, kernel_size=1, padding='SAME', 
                                name='conv0', l_name=self.l_name + '_conv0')
        # Grouped NxN conv
        self.conv1 = which_conv(seed, self.width, kernel_size=kernel_size, strides=stride, 
                                padding='SAME', name='conv1', l_name=self.l_name + '_conv1')
        # Conv 2, typically projection conv
        self.conv2 = which_conv(seed, self.out_ch, kernel_size=1, padding='SAME', 
                                name='conv2', l_name=self.l_name + '_conv2')
        # Use shortcut conv on channel change or downsample.
        self.use_projection = stride > 1 or self.in_ch != self.out_ch
        if self.use_projection:
            self.conv_shortcut = which_conv(seed, self.out_ch, kernel_size=1,
                                          strides=stride, padding='SAME', 
                                          name='conv_shortcut', l_name=self.l_name + '_conv_shortcut')
        # Are we using stochastic depth?
        self._has_stochdepth = (stochdepth_rate is not None and
                                stochdepth_rate > 0. and stochdepth_rate < 1.0)
        if self._has_stochdepth:
            self.stoch_depth = StochasticDepth(stochdepth_rate, seed=seed, l_name=self.l_name + '_stoch_depth')
    
        if self.use_se:
            self.se = SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio, seed=seed, l_name=self.l_name + '_se')

        self.skip_gain = self.add_weight(name=self.l_name + '_skip_gain', shape=(), initializer="zeros", trainable=True, dtype=self.dtype)



    def call(self, x, training, inject=None, inj_args=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        layer_inputs[self.l_name + "_activation_0"] = x
        out = self.activation(x) * self.beta

        if self.use_projection:  # Downsample with conv1x1
            layer_inputs[self.l_name + "_conv_shortcut"] = out
            shortcut, conv_out, std_w, w_max_bitmap = self.conv_shortcut(out)
            layer_kernels[self.l_name + "_conv_shortcut"] = self.conv_shortcut.weights[:3] + [std_w, w_max_bitmap]
            layer_outputs[self.l_name + "_conv_shortcut"] = conv_out
        else:
            shortcut = x

        layer_inputs[self.l_name + "_conv0"] = out
        out, conv_out, std_w, w_max_bitmap = self.conv0(out)
        layer_kernels[self.l_name + "_conv0"] = self.conv0.weights[:3] + [std_w, w_max_bitmap]
        layer_outputs[self.l_name + "_conv0"] = conv_out

        layer_inputs[self.l_name + "_activation_1"] = out
        out = self.activation(out)

        layer_inputs[self.l_name + "_conv1"] = out
        out, conv_out, std_w, w_max_bitmap = self.conv1(out)
        layer_kernels[self.l_name + "_conv1"] = self.conv1.weights[:3] + [std_w, w_max_bitmap]
        layer_outputs[self.l_name + "_conv1"] = conv_out

        layer_inputs[self.l_name + "_activation_2"] = out
        out = self.activation(out)

        layer_inputs[self.l_name + "_conv2"] = out
        out, conv_out, std_w, w_max_bitmap = self.conv2(out)
        layer_kernels[self.l_name + "_conv2"] = self.conv2.weights[:3] + [std_w, w_max_bitmap]
        layer_outputs[self.l_name + "_conv2"] = conv_out

        if self.use_se:
            #out = 2 * self.se(out) * out
            se_out, block_inputs, block_kernels, block_outputs = self.se(out)
            layer_inputs.update(block_inputs)
            layer_kernels.update(block_kernels)
            layer_outputs.update(block_outputs)
            layer_outputs[self.l_name + "_se"] = se_out
            layer_inputs[self.l_name + "_se_output"] = se_out

            out = 2 * se_out * out

        # Apply stochdepth if applicable.
        if self._has_stochdepth:
            #out = self.stoch_depth(out, training)
            out, binary_tensor = self.stoch_depth(out, training)
            layer_inputs[self.l_name + "_binary"] = binary_tensor

        # SkipInit Gain
        layer_inputs[self.l_name + "_out"] = out
        out = out * self.skip_gain
        layer_kernels[self.l_name + "_skip_gain"] = self.skip_gain
        return out * self.alpha + shortcut, layer_inputs, layer_kernels, layer_outputs

