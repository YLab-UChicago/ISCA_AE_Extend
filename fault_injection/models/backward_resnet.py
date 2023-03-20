import tensorflow as tf
from config import NUM_CLASSES
from models.backward_residual_block import BackwardBasicBlocks
from models.inject_layers import *


class BackwardResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params, drop_rate=0.15):
        super(BackwardResNetTypeI, self).__init__()

        self.backward_conv1 = BackwardInjectConv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            l_name="conv1")
        self.backward_bn1 = BackwardInjectBatchNormalization()
        self.backward_relu1 = BackwardInjectReLU()
        '''
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        '''

        self.backward_layer1 = BackwardBasicBlocks(filter_num=64,
                                  blocks=layer_params[0],
                                  drop_rate=drop_rate,
                                  l_name="basicblock_1")
        self.backward_layer2 = BackwardBasicBlocks(filter_num=128,
                                  blocks=layer_params[1],
                                  stride=2,
                                  drop_rate=drop_rate,
                                  l_name="basicblock_2")
        self.backward_layer3 = BackwardBasicBlocks(filter_num=256,
                                  blocks=layer_params[2],
                                  stride=2,
                                  drop_rate=drop_rate,
                                  l_name="basicblock_3")
        self.backward_layer4 = BackwardBasicBlocks(filter_num=512,
                                  blocks=layer_params[3],
                                  stride=2,
                                  drop_rate=drop_rate,
                                  l_name="basicblock_4")


    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_layer4(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)
         
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_layer3(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)
        
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_layer2(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_layer1(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_in = self.backward_relu1(grad_in, layer_inputs['relu1'])

        grad_in, grad_gamma, grad_beta = self.backward_bn1(grad_in, layer_inputs['bn1'], layer_kernels['bn1'][0], layer_kernels['bn1'][1], layer_kernels['bn1_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv1(grad_in, layer_inputs['conv1'], layer_kernels['conv1'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        return grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs


def backward_resnet_18(m_name):
    return BackwardResNetTypeI(layer_params=[2, 2, 2, 2], drop_rate=0 if 'sgd' in m_name else 0.15)

