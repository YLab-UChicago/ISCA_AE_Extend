import tensorflow as tf
import math
from config import NUM_CLASSES
from models.inject_layers import *


def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class BackwardSEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25, l_name=None):
        super(BackwardSEBlock, self).__init__()
        self.l_name = l_name
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.backward_pool = BackwardInjectGlobalAveragePooling2D()

        self.backward_expand_dims = BackwardInjectExpandDims(axis=1)
        self.backward_expand_dims_1 = BackwardInjectExpandDims(axis=1)

        self.backward_reduce_conv = BackwardInjectConv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same",
                                                  l_name = l_name + "_reduce_conv")

        self.backward_swish = BackwardInjectSwish()

        self.backward_expand_conv = BackwardInjectConv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same",
                                                  l_name = l_name + "_expand_conv")

        self.backward_sigmoid = BackwardInjectSigmoid()

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        grad_sigmoid_in = grad_in * layer_inputs[self.l_name]
        grad_sigmoid_in = tf.reduce_sum(grad_sigmoid_in, axis=1, keepdims=True)
        grad_sigmoid_in = tf.reduce_sum(grad_sigmoid_in, axis=2, keepdims=True)

        grad_in_2 = grad_in * tf.nn.sigmoid(layer_inputs[self.l_name + '_sigmoid'])

        grad_in = self.backward_sigmoid(grad_sigmoid_in, layer_inputs[self.l_name + '_sigmoid'])

        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_expand_conv(grad_in, layer_inputs[self.l_name + '_expand_conv'], layer_kernels[self.l_name + '_expand_conv'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)
        
        grad_in = self.backward_swish(grad_in, layer_inputs[self.l_name + '_swish'])

        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_reduce_conv(grad_in, layer_inputs[self.l_name + '_reduce_conv'], layer_kernels[self.l_name + '_reduce_conv'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)
  
        grad_in = self.backward_expand_dims_1(grad_in)
        grad_in = self.backward_expand_dims(grad_in)

        grad_in = self.backward_pool(grad_in, layer_inputs[self.l_name])

        return grad_in + grad_in_2, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs


class BackwardMBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, l_name=None):
        super(BackwardMBConv, self).__init__()
        self.l_name = l_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate


        self.backward_conv1 = BackwardInjectConv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False,
                                            l_name = self.l_name + "_conv1")

        self.backward_bn1 = BackwardInjectBatchNormalization()

        self.backward_swish = BackwardInjectSwish()

        self.backward_dwconv = BackwardInjectDepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same",
                                                      use_bias=False,
                                                      l_name = self.l_name + "_dwconv")

        self.backward_bn2 = BackwardInjectBatchNormalization()
        self.backward_se = BackwardSEBlock(input_channels=in_channels * expansion_factor,
                            l_name = self.l_name + "_se")

        self.backward_swish_1 = BackwardInjectSwish()
        self.backward_conv2 = BackwardInjectConv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False,
                                            l_name = self.l_name + "_conv2")
        self.backward_bn3 = BackwardInjectBatchNormalization()
        self.backward_dropout = BackwardDropout(rate=drop_connect_rate)

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        if self.stride == 1 and self.in_channels == self.out_channels:
            grad_in_2 = grad_in
            if self.drop_connect_rate:
                grad_in = self.backward_dropout(grad_in, layer_inputs[self.l_name + "_dropout"])
        else:
            grad_in_2 = None

        grad_in, grad_gamma, grad_beta = self.backward_bn3(grad_in, layer_inputs[self.l_name + '_bn3'], layer_kernels[self.l_name + '_bn3'][0], layer_kernels[self.l_name + '_bn3'][1], layer_kernels[self.l_name + '_bn3_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        grad_in, grad_wt, _, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv2(grad_in, layer_inputs[self.l_name + '_conv2'], layer_kernels[self.l_name + '_conv2'][0], inject=inject, inj_args=inj_args)
        # No bias for this layer
        #grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_in = self.backward_swish_1(grad_in, layer_inputs[self.l_name + '_swish_1'])

        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_se(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_in, grad_gamma, grad_beta = self.backward_bn2(grad_in, layer_inputs[self.l_name + '_bn2'], layer_kernels[self.l_name + '_bn2'][0], layer_kernels[self.l_name + '_bn2'][1], layer_kernels[self.l_name + '_bn2_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        # Currently do not inject to depthwise layer
        grad_in, grad_wt, _ = self.backward_dwconv(grad_in, layer_inputs[self.l_name + "_dwconv"], layer_kernels[self.l_name + "_dwconv"][0])
        #grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)

        grad_in = self.backward_swish(grad_in, layer_inputs[self.l_name + "_swish"])

        grad_in, grad_gamma, grad_beta = self.backward_bn1(grad_in, layer_inputs[self.l_name + '_bn1'], layer_kernels[self.l_name + '_bn1'][0], layer_kernels[self.l_name + '_bn1'][1], layer_kernels[self.l_name + '_bn1_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        grad_in, grad_wt, _, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv1(grad_in, layer_inputs[self.l_name + '_conv1'], layer_kernels[self.l_name + '_conv1'][0], inject=inject, inj_args=inj_args)
        #grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        if self.stride == 1 and self.in_channels == self.out_channels:
            grad_in += grad_in_2

        return grad_in, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs


class BackwardMBConv_Block(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate, l_name=None):
        super(BackwardMBConv_Block, self).__init__()
        self.l_name = l_name
        self.n_layers = layers
        self.layers = {}

        for i in range(self.n_layers):
            if i == 0:
                self.layers['mbconv_{}'.format(i)] = BackwardMBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate,
                             l_name = self.l_name + "_mbconv_{}".format(i))
            else:
                self.layers['mbconv_{}'.format(i)] = BackwardMBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate,
                             l_name = self.l_name + "_mbconv_{}".format(i))
 
    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        for i in range(self.n_layers-1, -1, -1):
            grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.layers['mbconv_{}'.format(i)](grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
            grad_params = block_grad_params + grad_params
            bkwd_layer_inputs.update(bkwd_block_inputs)
            bkwd_layer_kernels.update(bkwd_block_kernels)
            bkwd_layer_outputs.update(bkwd_block_outputs)

        return grad_in, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs


class BackwardEfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate):
        super(BackwardEfficientNet, self).__init__()

        self.backward_conv1 = BackwardInjectConv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",
                                            use_bias=False,
                                            l_name="conv1")
        self.backward_bn1 = BackwardInjectBatchNormalization()

        self.backward_swish = BackwardInjectSwish()

        self.backward_block1 = BackwardMBConv_Block(in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate,
                                         l_name="block1")
        self.backward_block2 = BackwardMBConv_Block(in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(24, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                         l_name="block2")
        self.backward_block3 = BackwardMBConv_Block(in_channels=round_filters(24, width_coefficient),
                                         out_channels=round_filters(40, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                         l_name="block3")
        self.backward_block4 = BackwardMBConv_Block(in_channels=round_filters(40, width_coefficient),
                                         out_channels=round_filters(80, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                         l_name="block4")
        self.backward_block5 = BackwardMBConv_Block(in_channels=round_filters(80, width_coefficient),
                                         out_channels=round_filters(112, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                         l_name="block5")
        self.backward_block6 = BackwardMBConv_Block(in_channels=round_filters(112, width_coefficient),
                                         out_channels=round_filters(192, width_coefficient),
                                         layers=round_repeats(4, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                         l_name="block6")
        self.backward_block7 = BackwardMBConv_Block(in_channels=round_filters(192, width_coefficient),
                                         out_channels=round_filters(320, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                         l_name="block7")

        self.backward_conv2 = BackwardInjectConv2D(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False,
                                            l_name="conv2")

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        grad_params = []

        grad_in, grad_wt, _, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv2(grad_in, layer_inputs['conv2'], layer_kernels['conv2'][0], inject=inject, inj_args=inj_args)
        #grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        #print("DEBUG: block 7")

        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_block7(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        #print("DEBUG: block 6")

        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_block6(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        #print("DEBUG: block 5")

        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_block5(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_block4(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_block3(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_block2(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_block1(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)
    
        grad_in = self.backward_swish(grad_in, layer_inputs['swish'])

        grad_in, grad_gamma, grad_beta = self.backward_bn1(grad_in, layer_inputs['bn1'], layer_kernels['bn1'][0], layer_kernels['bn1'][1], layer_kernels['bn1_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        grad_in, grad_wt, _, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv1(grad_in, layer_inputs['conv1'], layer_kernels['conv1'][0], inject=inject, inj_args=inj_args)
        #grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        return grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs


def get_backward_efficient_net(width_coefficient, depth_coefficient, resolution, dropout_rate):
    net = BackwardEfficientNet(width_coefficient=width_coefficient,
                       depth_coefficient=depth_coefficient,
                       dropout_rate=dropout_rate,
                       drop_connect_rate=dropout_rate)

    return net

def backward_efficient_net_b0():
    return get_backward_efficient_net(1.0, 1.0, 224, 0)

