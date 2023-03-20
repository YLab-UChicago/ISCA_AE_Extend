import tensorflow as tf
import numpy as np
from models.inject_utils import *
from tensorflow.keras.regularizers import l2
import config
from models.inject_utils import is_input_target, is_weight_target, is_output_target

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


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.bias = None

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                         #shape=input_shape[1:],
                         shape=input_shape[-1:],
                         initializer='zeros',
                         trainable=True)

    def call(self, x):
        return x + self.bias


class InjectReLU(tf.keras.layers.ReLU):
    def __init__(self, l_name=None):
        super(InjectReLU, self).__init__()
        self.l_name = l_name

    def call(self, inputs):
        return super(InjectReLU, self).call(inputs)


class BackwardInjectReLU(tf.keras.layers.Layer):
    def __init__(self, l_name=None):
        super(BackwardInjectReLU, self).__init__()
        self.l_name = l_name

    def call(self, grad_out, inputs):
        return grad_out * tf.cast(tf.greater(inputs, 0), tf.float32)


class InjectBatchNormalization(tf.keras.layers.BatchNormalization):
    def  __init__(self, l_name=None, **kargs):
        super(InjectBatchNormalization, self).__init__(**kargs)
        self.l_name = l_name

    def call(self, inputs, training=None):
        return super(InjectBatchNormalization, self).call(inputs, training=training)


class BackwardInjectBatchNormalization(tf.keras.layers.Layer):
    def  __init__(self, l_name=None):
        super(BackwardInjectBatchNormalization, self).__init__()
        self.l_name = l_name

    def call(self, dout, inputs, gamma, beta, epsilon):
        # First compute forward 
        b,h,w,c = inputs.get_shape().as_list()
        if b == None:
            b = config.PER_REPLICA_BATCH_SIZE 
        N = b * h * w
        mu = tf.reduce_mean(inputs, axis=[0,1,2])

        xmu = inputs - mu
        sq = tf.square(xmu)
        var = tf.reduce_mean(sq, axis=[0,1,2])

        sqrtvar = tf.sqrt(var + epsilon)
        ivar = 1. / sqrtvar
        xhat = xmu * ivar
        gammax = gamma * xhat
        out = gammax + beta

        # Then compute gradients
        dbeta = tf.reduce_sum(dout, axis=[0,1,2])
        dgammax = dout
        dgamma = tf.reduce_sum(dgammax * xhat, axis=[0,1,2])
        dxhat = dgammax * gamma
        
        divar = tf.reduce_sum(dxhat * xmu, axis=[0,1,2])
        dxmu1 = dxhat * ivar

        dsqrtvar = (-1. / tf.square(sqrtvar)) * divar
        #dvar = 0.5 * ivar * dsqrtvar
        dvar = 0.5 * 1. / tf.sqrt(var + epsilon) * dsqrtvar

        dsq = (1./N) * tf.ones_like(dout) * dvar
        dxmu2 = 2. * xmu * dsq

        dx1 = dxmu1 + dxmu2
        
        dmu = -1. * tf.reduce_sum(dxmu1+dxmu2, axis=[0,1,2])
        dx2 = (1./N) * tf.ones_like(dout) * dmu

        dx = dx1 + dx2

        return dx, dgamma, dbeta



class InjectGlobalAveragePooling2D(tf.keras.layers.GlobalAveragePooling2D):
    def __init__(self, l_name=None):
        super(InjectGlobalAveragePooling2D, self).__init__()
        self.l_name = l_name

    def call(self, inputs):
        return super(InjectGlobalAveragePooling2D, self).call(inputs)


class BackwardInjectGlobalAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, l_name=None):
        super(BackwardInjectGlobalAveragePooling2D, self).__init__()
        self.l_name = l_name

    def call(self, grad_out, inputs):
        h = inputs.get_shape().as_list()[1]
        grad_out = grad_out / (h*h)

        grad_med = tf.stack([tf.stack([grad_out] * h)] * h)
        grad_out = tf_HWIO_to_NHWC(grad_med)
        return grad_out


class InjectMaxPooling2D(tf.keras.layers.MaxPooling2D):
    def __init__(self, pool_size=(2,2), strides=None, padding='valid', l_name=None):
        super(InjectMaxPooling2D, self).__init__()
        self.l_name = l_name

    def call(self, inputs):
        return super(InjectMaxPooling2D, self).call(inputs)


class BackwardInjectMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2,2), strides=None, padding='same', l_name=None):
        super(BackwardInjectMaxPooling2D, self).__init__()
        self.pool_size = pool_size[0]
        self.k_size = strides
        self.l_name = l_name
        self.padding = padding

    def call(self, grad_out, inputs, argmax, batch_size=None):
        if self.padding.lower() != 'same':
            print("Unable to process valid padding for max pooling")
            exit(1)

        input_shape = grad_out.get_shape().as_list()
        if input_shape[0] == None:
            input_shape[0] = batch_size
        output_shape = (input_shape[0], input_shape[1] * self.k_size, input_shape[2] * self.k_size, input_shape[3])

        pooled_ = tf.reshape(grad_out, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=argmax.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(argmax) * batch_range
        b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
        ind_ = tf.reshape(argmax, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
        ind_ = tf.concat([b_, ind_],1)
        ref = tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])
        # Update the sparse matrix with the pooled values , it is a batch wise operation
        unpooled_ = tf.tensor_scatter_nd_add(ref, ind_, pooled_)
        # Reshape the vector to get the final result 
        unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
        return unpooled


class BackwardInjectSigmoid(tf.keras.layers.Layer):
    def __init__(self, l_name=None):
        super(BackwardInjectSigmoid, self).__init__()
        self.l_name = l_name

    def call(self, grad_out, inputs):
        sig_out = tf.nn.sigmoid(inputs)
        return grad_out * sig_out * (1 - sig_out)



class BackwardInjectSwish(tf.keras.layers.Layer):
    def __init__(self, l_name=None):
        super(BackwardInjectSwish, self).__init__()
        self.l_name = l_name

    def call(self, grad_out, inputs):
        sig_out = tf.sigmoid(inputs)
        return grad_out * (sig_out + inputs * sig_out * (1 - sig_out))



class BackwardInjectExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis=0, l_name=None):
        super(BackwardInjectExpandDims, self).__init__()
        self.axis = axis
        self.l_name = l_name

    def call(self, grad_out):
        return tf.reduce_sum(grad_out, axis=self.axis)


class BackwardDropout(tf.keras.layers.Layer):
    def __init__(self, rate, l_name=None):
        super(BackwardDropout, self).__init__()
        self.rate = rate
        self.l_name = l_name

    def call(self, grad_out, input_mask):
        return grad_out * input_mask * (1. /(1. - self.rate))


class BackwardInjectDepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides, padding='valid', use_bias=False, l_name=None):
        super(BackwardInjectDepthwiseConv2D, self).__init__()
        self.kernel_size = kernel_size[0]
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.l_name = l_name

    def call(self, grad_out, inputs, kernels, inject=None, inj_args=None):

        fmt_inp = tf_pad_to_full_conv2d(grad_out if self.strides == 1 else tf_pad_with_stride(grad_out, 2, self.kernel_size, self.padding), self.kernel_size, self.padding, is_input=True)
        #fmt_wt = tf.transpose(tf_rot180(kernels), perm=[0,1,3,2])
        fmt_wt = tf_rot180(kernels)

        #print("Grad out shape is {}".format(grad_out.get_shape()))
        #print("Weight shape is {}".format(kernels.get_shape()))
        #print("Input shape is {}".format(inputs.get_shape()))
        manual_grad_in = tf.nn.depthwise_conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding='VALID')

        '''
        manual_grad_in = []
        n_c = fmt_inp.get_shape()[-1]
        for c in range(n_c):
            manual_grad_in.append(tf.nn.conv2d(fmt_inp[:,:,:,c:c+1], fmt_wt[:,:,:,c:c+1], strides=[1,1,1,1], padding='VALID'))
        manual_grad_in = tf.concat(manual_grad_in, axis=-1)
        #manual_grad_in = tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")
        '''

        #print("Manual grad for input {}".format(manual_grad_in.get_shape()))

        ####################

        fmt_inp = tf.transpose(tf_pad_to_full_conv2d(inputs, self.kernel_size, self.padding, is_input=False), perm=[3,1,2,0])
        fmt_wt = tf_NHWC_to_HWIO(grad_out if self.strides == 1 else tf_pad_with_stride(grad_out, self.strides, self.kernel_size, self.padding))
        #print("manual grad for weight input: {} {}".format(fmt_inp.shape, fmt_wt.shape))

        all_grad_wt = tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")
        
        manual_direct_grad_wt = []
        n_c = fmt_inp.get_shape()[0]
        for c in range(n_c):
            manual_direct_grad_wt.append(all_grad_wt[c:c+1,:,:,c:c+1])
        manual_direct_grad_wt = tf.concat(manual_direct_grad_wt, axis=0)

        manual_grad_wt = tf_NHWC_to_HWIO(manual_direct_grad_wt)

        #print("Manual grad for weight {}".format(manual_grad_wt.get_shape()))

        if self.use_bias:
            manual_grad_bias = tf.reduce_sum(grad_out, axis=[0,1,2])
        else:
            manual_grad_bias = None

        return manual_grad_in, manual_grad_wt, manual_grad_bias



class InjectDense(tf.keras.layers.Dense):
    def __init__(self, units, use_bias=True, activation=None, l_name=None):
        super(InjectDense, self).__init__(units=units, activation=activation, use_bias=False)
        self.l_name = l_name
        self.has_bias = use_bias
        if self.has_bias:
            self.bias_layer = BiasLayer()

    def call(self, inputs, inject=None, inj_args=None):
        if not inject:
            conv_out = super(InjectDense, self).call(inputs)
        else:
            is_target = (inj_args and inj_args.inj_layer == self.l_name)
    
            if is_target:
                if is_input_target(inj_args.inj_type):
                    inputs = inj_to_tensor(inputs, inj_args)
                elif is_weight_target(inj_args.inj_type):
                    self.set_weights([inj_to_tensor(None, inj_args), inj_args.golden_weights[1]])

            conv_out = super(InjectDense, self).call(inputs)

            # Set weights back
            if is_weight_target(inj_args.inj_type):
                self.set_weights(inj_args.golden_weights)

            if is_target:
                # Inject to output
                if is_output_target(inj_args.inj_type):
                    conv_out = inj_to_tensor(conv_out, inj_args)

                # TODO: Correction for INPUT_16 and WT_16

        if self.has_bias:
            layer_out = self.bias_layer(conv_out)
        else:
            layer_out = conv_out

        return layer_out



class BackwardInjectDense(tf.keras.layers.Layer):
    def __init__(self, l_name=None):
        super(BackwardInjectDense, self).__init__()
        self.l_name = l_name

    def call(self, grad_out, inputs, kernels):
        #print("Grad out shape : {}".format(grad_out.shape))
        #print("Input shape: {}".format(inputs.shape))
        #print("Kernel shape: {}".format(kernels.shape))

        manual_grad_in = tf.matmul(grad_out, tf.transpose(kernels))
        manual_grad_wt = tf.matmul(tf.transpose(inputs), grad_out)
        manual_grad_bias = tf.reduce_sum(grad_out, axis=[0])

        return manual_grad_in, manual_grad_wt, manual_grad_bias


class InjectConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               seed=None,
               l_name=None,
               **kwargs):
        super(InjectConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activation,
        use_bias=False,
        #kernel_initializer=kernel_initializer,
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
        self.l_name = l_name
        self.has_bias = use_bias

        if self.has_bias:
            self.bias_layer = BiasLayer()

    def call(self, inputs, inject=None, inj_args=None):
        is_target = (inj_args and inj_args.inj_layer == self.l_name)

        if not is_target:
            conv_out = super(InjectConv2D, self).call(inputs)
        else:
            def no_inj(inputs):
                return super(InjectConv2D, self).call(inputs)

            def do_inj(inputs, inj_args):
                if is_input_target(inj_args.inj_type):
                    inputs = inj_to_tensor(inputs, inj_args)
                if is_weight_target(inj_args.inj_type):
                    modified_wts = [inj_to_tensor(None, inj_args)]
                    for i in range(1,len(inj_args.golden_weights)):
                        modified_wts.append(inj_args.golden_weights[i])
                    self.set_weights(modified_wts)

                conv_out = super(InjectConv2D, self).call(inputs)

                # Set weights back
                if is_weight_target(inj_args.inj_type):
                    self.set_weights(inj_args.golden_weights)

                if is_target:
                    # Inject to output
                    if is_output_target(inj_args.inj_type):
                        conv_out = inj_to_tensor(conv_out, inj_args)

                    # TODO: Correction for INPUT_16 and WT_16
                return conv_out

            conv_out = tf.cond(tf.reduce_all(inject), lambda: do_inj(inputs, inj_args), lambda: no_inj(inputs))

        if self.has_bias:
            layer_out = self.bias_layer(conv_out)
        else:
            layer_out = conv_out

        return layer_out, conv_out


class BackwardInjectConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', use_bias=True, l_name=None):
        super(BackwardInjectConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size[0]
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.l_name = l_name

    def call(self, grad_out, inputs, kernels, inject=None, inj_args=None):

        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        is_target = False
        if inj_args:
            full_name = inj_args.inj_layer
            subs_pos = full_name.rfind('_')
            subs_pos = full_name.rfind('_', 0, subs_pos-1)
            layer_name = full_name[:subs_pos]
            is_target = (layer_name == self.l_name)
        is_input_target = is_target and '_grad_in' in inj_args.inj_layer 
        is_wt_target = is_target and '_grad_wt' in inj_args.inj_layer

        if is_input_target:
            print("DEBUG: Start injecting error to bkwd input layer!")
        if is_wt_target:
            print("DEBUG: Start injecting error to bkwd wt layer!")

        fmt_inp = tf_pad_to_full_conv2d(grad_out if self.strides == 1 else tf_pad_with_stride(grad_out, 2, self.kernel_size, self.padding), self.kernel_size, self.padding, is_input=True)
        fmt_wt = tf.transpose(tf_rot180(kernels), perm=[0,1,3,2])
        #print("manual grad for input input: {} {}".format(fmt_inp.shape, fmt_wt.shape))

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
        #print("Manual grad for input {}".format(manual_grad_in.shape))

        ####################
        #print("DEBUG: kernel size is {}".format(self.kernel_size))
        #print("DEBUG: padding is {}".format(self.padding))
        #print("DEBUG: stride is {}".format(self.strides))

        #print("DEBUG: grad_out shape is {}".format(grad_out.get_shape()))
        #print("DEBUG: input shape is {}".format(inputs.get_shape()))
        #print("DEBUG: weight shape is {}".format(kernels.get_shape()))

        fmt_inp = tf.transpose(tf_pad_to_full_conv2d(inputs, self.kernel_size, self.padding, is_input=False), perm=[3,1,2,0])
        fmt_wt = tf_NHWC_to_HWIO(grad_out if self.strides == 1 else tf_pad_with_stride(grad_out, self.strides, self.kernel_size, self.padding))
        #print("manual grad for weight input: {} {}".format(fmt_inp.shape, fmt_wt.shape))
        bkwd_layer_inputs[self.l_name + '_grad_wt'] = fmt_inp
        bkwd_layer_kernels[self.l_name + '_grad_wt'] = fmt_wt
        
        if not is_wt_target:
            manual_direct_grad_wt = tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")
        else:
            def no_wt_inj(fmt_inp, fmt_wt):
                return tf.nn.conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID")

            def wt_inj(fmt_inp, fmt_wt, inj_args):
                return inject_nn_conv2d(fmt_inp, fmt_wt, strides=[1,1,1,1], padding="VALID", inj_args=inj_args)

            manual_direct_grad_wt = tf.cond(tf.reduce_all(inject), lambda: wt_inj(fmt_inp, fmt_wt, inj_args), lambda: no_wt_inj(fmt_inp, fmt_wt))

        bkwd_layer_outputs[self.l_name + '_grad_wt'] = manual_direct_grad_wt
        manual_grad_wt = tf_NHWC_to_HWIO(manual_direct_grad_wt)

        #print("Manual grad for weight {}".format(manual_grad_wt.shape))

        if self.use_bias:
            manual_grad_bias = tf.reduce_sum(grad_out, axis=[0,1,2])
        else:
            manual_grad_bias = None

        return manual_grad_in, manual_grad_wt, manual_grad_bias, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs

