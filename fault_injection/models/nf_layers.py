#############################
# nf_resnet layers 
#############################

import tensorflow as tf
import numpy as np

class InjectWSConv2D(tf.keras.layers.Conv2D):
    def __init__(self, seed, *args, **kwargs):
        filtered_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key != 'l_name'
                }
        super(InjectWSConv2D, self).__init__(
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0, mode='fan_in', distribution='untruncated_normal', seed=seed,
            ), *args, **filtered_kwargs
        )

        self.l_name = kwargs['l_name']

        # Get gain
        self.gain = self.add_weight(
            name=self.l_name + '_gain',
            shape=(self.filters,),
            initializer="ones",
            trainable=True,
            dtype=self.dtype
        )

    def standardize_weight(self, eps):
        mean = tf.math.reduce_mean(self.kernel, axis = (0, 1, 2), keepdims = True)
        var = tf.math.reduce_variance(self.kernel, axis = (0, 1, 2), keepdims = True)
        fan_in = np.prod(self.kernel.shape[:-1])
        #gain = self.add_weight(name = 'gain', shape = (weight.shape[-1],), initializer = "ones", trainable = True, dtype = self.dtype)

        fvar = var * fan_in
        wmax = tf.math.maximum(fvar , tf.convert_to_tensor(eps, dtype = self.dtype))
        w_max_bitmap = tf.cast(tf.greater(fvar, tf.convert_to_tensor(eps, dtype = self.dtype)), tf.float32)
        scale = tf.math.rsqrt(wmax) * self.gain

        #out = weight * scale - (mean * scale)
        xmu = self.kernel - mean
        out = xmu * scale

        return out, w_max_bitmap, scale

    def call(self, inputs, eps = 1e-4, inject=None, inj_args=None):
        is_target = (inj_args and inj_args.inj_layer == self.l_name)
        if is_target:
            print("Inject to layer!")

        standardized_kernel, w_max_bitmap, scale = self.standardize_weight(eps)
        input_shape = inputs.shape

        if not is_target:
            conv_out = tf.nn.conv2d(
                inputs, standardized_kernel, strides=self.strides,
                padding=self.padding.upper(), dilations=self.dilation_rate)
        else:
            def no_inj(inputs):
                conv_out = tf.nn.conv2d(
                    inputs, standardized_kernel, strides=self.strides,
                    padding=self.padding.upper(), dilations=self.dilation_rate)
            def do_inj(inputs, inj_args):
                if is_input_target(inj_args.inj_type):
                    inputs = inj_to_tensor(inputs, inj_args)
                if is_weight_target(inj_args.inj_type):
                    # We are injecting to the standardized kernels
                    modified_wts = [inj_to_tensor(None, inj_args)]
                else:
                    modified_wts = standardized_kernel
                conv_out = tf.nn.conv2d(
                    inputs, modified_wts, strides=self.strides,
                    padding=self.padding.upper(), dilations=self.dilation_rate)

                if is_target:
                    # Inject to output
                    if is_output_target(inj_args.inj_type):
                        conv_out = inj_to_tensor(conv_out, inj_args)

                    # TODO: Correction for INPUT_16 and WT_16
                return conv_out

            conv_out = tf.cond(tf.reduce_all(inject), lambda: do_inj(inputs, inj_args), lambda: no_inj(inputs))
        outputs = conv_out

        if self.use_bias:
            out = conv_out + self.bias
        else:
            out = conv_out

        #return out, conv_out, standardized_kernel, w_max_bitmap, scale
        return out, conv_out, standardized_kernel, w_max_bitmap


class SqueezeExcite(tf.keras.Model):

    def __init__(self, in_ch, out_ch, seed, se_ratio = 0.5, hidden_ch = None, activation = tf.keras.activations.relu, name = None, l_name = None):
        super(SqueezeExcite, self).__init__(name = name)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.l_name = l_name

        if se_ratio is None:
            if hidden_ch is None: raise ValueError('Must provide one of se_ratio or hidden_ch')
            self.hidden_ch = hidden_ch
        else: self.hidden_ch = max(1, int(self.in_ch * se_ratio))
        self.activation = activation
        self.fc0 = tf.keras.layers.Dense(self.hidden_ch, use_bias = True, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))
        self.fc1 = tf.keras.layers.Dense(self.out_ch, use_bias = True, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))

    def call(self, x):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        layer_inputs[self.l_name] = x
        h = tf.math.reduce_mean(x, axis = [1, 2])
        
        layer_inputs[self.l_name + "_fc0"] = h
        h = self.fc0(h)
        layer_kernels[self.l_name + "_fc0"] = self.fc0.weights[0]

        layer_inputs[self.l_name + "_activation"] = h
        h = self.activation(h)

        layer_inputs[self.l_name + "_fc1"] = h
        h = self.fc1(h)
        layer_kernels[self.l_name + "_fc1"] = self.fc1.weights[0]

        layer_inputs[self.l_name + "_sigmoid"] = h
        h = sigmoid(h)[:, None, None]
        return h, layer_inputs, layer_kernels, layer_outputs


class StochasticDepth(tf.keras.Model):

    def __init__(self, drop_rate, seed, scale_by_keep = False, name = None, l_name=None):
        super(StochasticDepth, self).__init__(name = name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep
        self.l_name = l_name

    def call(self, x, training):
        if not training: return x
        batch_size = x.shape[0]
        r = tf.random.uniform(seed=seed, shape = [batch_size, 1, 1, 1], dtype = x.dtype)
        keep_prob = 1. - self.drop_rate
        binary_tensor = tf.floor(keep_prob + r)

        if self.scale_by_keep: x = x / keep_prob
        return x * binary_tensor, binary_tensor

