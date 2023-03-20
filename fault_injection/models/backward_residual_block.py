import tensorflow as tf
from models.inject_utils import *
from models.inject_layers import *

class BackwardBasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, drop_rate=0.15, l_name=None):
        super(BackwardBasicBlock, self).__init__()
        self.l_name = l_name
        self.stride = stride
        self.drop_rate = drop_rate

        self.backward_conv1 = BackwardInjectConv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            l_name=self.l_name + "_conv1")
        self.backward_bn1 = BackwardInjectBatchNormalization()
        self.backward_relu1 = BackwardInjectReLU()
        self.backward_dropout1 = BackwardDropout(rate=self.drop_rate)

        self.backward_conv2 = BackwardInjectConv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            l_name=self.l_name + "_conv2")
        self.backward_bn2 = BackwardInjectBatchNormalization()
        self.backward_dropout2 = BackwardDropout(rate=self.drop_rate)

        if stride != 1:
            self.backward_downsample_conv = BackwardInjectConv2D(filters=filter_num,
                                                kernel_size=(1, 1),
                                                strides=stride,
                                                padding='same',
                                                l_name=self.l_name + "_downsample")
            self.backward_bn3 = BackwardInjectBatchNormalization()

        else:
            self.backward_downsample_conv = lambda x: x
            self.backward_bn3 = lambda x: x

        self.backward_relu_add = BackwardInjectReLU()


    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        grad_in = self.backward_dropout2(grad_in, layer_inputs[self.l_name + "_dropout2"])

        grad_downsample_in = self.backward_relu_add(grad_in, layer_inputs[self.l_name + '_relu_add'])

        if self.stride != 1:
            grad_residual, grad_gamma, grad_beta = self.backward_bn3(grad_downsample_in, layer_inputs[self.l_name + '_bn3'], layer_kernels[self.l_name + '_bn3'][0], layer_kernels[self.l_name + '_bn3'][1], layer_kernels[self.l_name + '_bn3_epsilon'])
            grad_params.insert(0, grad_beta)
            grad_params.insert(0, grad_gamma)

            grad_residual, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_downsample_conv(grad_residual, layer_inputs[self.l_name + '_downsample'], layer_kernels[self.l_name + '_downsample'][0], inject=inject, inj_args=inj_args)
            grad_params.insert(0, grad_bias)
            grad_params.insert(0, grad_wt)
            bkwd_layer_inputs.update(bkwd_block_inputs)
            bkwd_layer_kernels.update(bkwd_block_kernels)
            bkwd_layer_outputs.update(bkwd_block_outputs)

        else:
            grad_residual = grad_downsample_in

        grad_in, grad_gamma, grad_beta = self.backward_bn2(grad_downsample_in, layer_inputs[self.l_name + '_bn2'], layer_kernels[self.l_name + '_bn2'][0], layer_kernels[self.l_name + '_bn2'][1], layer_kernels[self.l_name + '_bn2_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv2(grad_in, layer_inputs[self.l_name + '_conv2'], layer_kernels[self.l_name + '_conv2'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_in = self.backward_dropout1(grad_in, layer_inputs[self.l_name + "_dropout1"])

        grad_in = self.backward_relu1(grad_in, layer_inputs[self.l_name + "_relu1"])

        grad_in, grad_gamma, grad_beta = self.backward_bn1(grad_in, layer_inputs[self.l_name + '_bn1'], layer_kernels[self.l_name + '_bn1'][0], layer_kernels[self.l_name + '_bn1'][1], layer_kernels[self.l_name + '_bn1_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv1(grad_in, layer_inputs[self.l_name + '_conv1'], layer_kernels[self.l_name + '_conv1'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        return tf.add(grad_residual, grad_in), grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs



class BackwardBasicBlocks(tf.keras.layers.Layer):
    def __init__(self, filter_num, blocks, stride=1, drop_rate=0.15, l_name=None):
        super(BackwardBasicBlocks, self).__init__()
        self.l_name = l_name
        self.backward_basics = []
        self.backward_basics.append(BackwardBasicBlock(filter_num, stride=stride, drop_rate=drop_rate, l_name=self.l_name + "_basic_0"))

        for i in range(1, blocks):
            self.backward_basics.append(BackwardBasicBlock(filter_num, stride=1, drop_rate=drop_rate, l_name=self.l_name + "_basic_{}".format(i)))

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        for i in range(len(self.backward_basics)-1, -1, -1):
            basic = self.backward_basics[i]
            grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = basic(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
            bkwd_layer_inputs.update(bkwd_block_inputs)
            bkwd_layer_kernels.update(bkwd_block_kernels)
            bkwd_layer_outputs.update(bkwd_block_outputs)

            grad_params = block_grad_params + grad_params

        return grad_in, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs

