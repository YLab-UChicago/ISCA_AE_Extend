import tensorflow as tf
import math
import config
from config import NUM_CLASSES
from models.inject_layers import *


class BackwardBottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate, drop_rate, l_name=None):
        super(BackwardBottleNeck, self).__init__()
        self.l_name = l_name
        self.backward_bn1 = BackwardInjectBatchNormalization()
        self.backward_conv1 = BackwardInjectConv2D(filters=4 * growth_rate,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            l_name=self.l_name + "_conv1")
        self.backward_bn2 = BackwardInjectBatchNormalization()
        self.backward_conv2 = BackwardInjectConv2D(filters=growth_rate,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            l_name=self.l_name + "_conv2")

        self.backward_relu = BackwardInjectReLU()
        self.backward_relu_1 = BackwardInjectReLU()
        self.backward_dropout = BackwardDropout(rate=drop_rate)

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None, **kwargs):

        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        # dropout
        grad_in = self.backward_dropout(grad_in, layer_inputs[self.l_name + "_dropout"])

        # conv2
        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv2(grad_in, layer_inputs[self.l_name + '_conv2'], layer_kernels[self.l_name + '_conv2'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        # relu_1
        grad_in = self.backward_relu_1(grad_in, layer_inputs[self.l_name + "_relu_1"])

        # bn2
        grad_in, grad_gamma, grad_beta = self.backward_bn2(grad_in, layer_inputs[self.l_name + '_bn2'], layer_kernels[self.l_name + '_bn2'][0], layer_kernels[self.l_name + '_bn2'][1], layer_kernels[self.l_name + '_bn2_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        # conv1
        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv1(grad_in, layer_inputs[self.l_name + '_conv1'], layer_kernels[self.l_name + '_conv1'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        # relu
        grad_in = self.backward_relu(grad_in, layer_inputs[self.l_name + "_relu"])

        # bn1
        grad_in, grad_gamma, grad_beta = self.backward_bn1(grad_in, layer_inputs[self.l_name + '_bn1'], layer_kernels[self.l_name + '_bn1'][0], layer_kernels[self.l_name + '_bn1'][1], layer_kernels[self.l_name + '_bn1_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        return grad_in, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs



class BackwardDenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate, l_name=None):
        super(BackwardDenseBlock, self).__init__()
        self.l_name = l_name
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.features_list = []
        self.backward_bottle_necks = []
        for i in range(self.num_layers):
            self.backward_bottle_necks.append(BackwardBottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate, l_name=self.l_name + "_bottleneck_{}".format(i)))

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None, **kwargs):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        grad_in_0 = grad_in[:,:,:,:layer_inputs[self.l_name + "_input_k"]]
        grad_in_after = grad_in[:,:,:,layer_inputs[self.l_name + "_input_k"]:]
        grad_in_split = [grad_in_0] + tf.split(grad_in_after, self.num_layers, axis=-1)

        for i in range(self.num_layers-1, -1, -1):
            grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_bottle_necks[i](grad_in_split[i+1], layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
            grad_params = block_grad_params + grad_params
            bkwd_layer_inputs.update(bkwd_block_inputs)
            bkwd_layer_kernels.update(bkwd_block_kernels)
            bkwd_layer_outputs.update(bkwd_block_outputs)

            if i:
                grad_in_0 = grad_in[:,:,:,:layer_inputs[self.l_name + "_input_k"]]
                grad_in_after = grad_in[:,:,:,layer_inputs[self.l_name + "_input_k"]:]
                tmp_grad_in_split = [grad_in_0] + tf.split(grad_in_after, i, axis=-1)

                add_idx = 0
                for elem in tmp_grad_in_split:
                    grad_in_split[add_idx] += elem
                    add_idx += 1
                tmp_grad_in_split.clear()

        grad_in = grad_in + grad_in_split[0]
        grad_in_split.clear()

        return grad_in, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs



class BackwardTransitionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels, l_name=None):
        super(BackwardTransitionLayer, self).__init__()
        self.l_name = l_name
        self.backward_bn = BackwardInjectBatchNormalization()

        self.backward_conv_p = BackwardInjectConv2D(filters=out_channels,
                                           kernel_size=(3, 3),
                                           strides=2,
                                           padding="same",
                                           l_name = self.l_name + "_conv_p")


        self.backward_conv = BackwardInjectConv2D(filters=out_channels,
                                           kernel_size=(1, 1),
                                           strides=1,
                                           padding="same",
                                           l_name = self.l_name + "_conv")

        self.backward_relu = BackwardInjectReLU()

        '''
        self.backward_pool = BackwardInjectMaxPooling2D(pool_size=(2, 2),
                                              strides=2,
                                              padding="same")
        '''

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None, **kwargs):

        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}
        
        '''
        # pool
        grad_in = self.backward_pool(grad_in, layer_inputs[self.l_name + "_pool"], layer_inputs[self.l_name + "_pool_argmax"], batch_size=config.BATCH_SIZE)
        '''

        # conv_p
        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv_p(grad_in, layer_inputs[self.l_name + '_conv_p'], layer_kernels[self.l_name + '_conv_p'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        # conv
        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv(grad_in, layer_inputs[self.l_name + '_conv'], layer_kernels[self.l_name + '_conv'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        # relu
        grad_in = self.backward_relu(grad_in, layer_inputs[self.l_name + "_relu"])

        # bn
        grad_in, grad_gamma, grad_beta = self.backward_bn(grad_in, layer_inputs[self.l_name + '_bn'], layer_kernels[self.l_name + '_bn'][0], layer_kernels[self.l_name + '_bn'][1], layer_kernels[self.l_name + '_bn_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        return grad_in, grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs




class BackwardDenseNet(tf.keras.Model):
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate, l_name='densenet'):
        super(BackwardDenseNet, self).__init__()
        self.l_name = l_name
        self.backward_conv = BackwardInjectConv2D(filters=num_init_features,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding="same",
                                           l_name=self.l_name + '_conv')

        self.backward_relu = BackwardInjectReLU()
        self.backward_bn = BackwardInjectBatchNormalization()

        '''
        self.backward_pool = BackwardInjectMaxPooling2D(pool_size=(3, 3),
                                              strides=2,
                                              padding="same")
        '''
        self.backward_conv_p = BackwardInjectConv2D(filters=num_init_features,
                                           kernel_size=(3, 3),
                                           strides=2,
                                           padding="same",
                                           l_name=self.l_name + '_conv_p')


        self.num_channels = num_init_features
        self.backward_dense_block_1 = BackwardDenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate, l_name=self.l_name + '_dense_block_1')
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.backward_transition_1 = BackwardTransitionLayer(out_channels=int(self.num_channels), l_name=self.l_name + '_transition_1')
        self.backward_dense_block_2 = BackwardDenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate, l_name=self.l_name + "_dense_block_2")
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.backward_transition_2 = BackwardTransitionLayer(out_channels=int(self.num_channels), l_name=self.l_name + "_transition_2")
        self.backward_dense_block_3 = BackwardDenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate, l_name=self.l_name + "_dense_block_3")
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.backward_transition_3 = BackwardTransitionLayer(out_channels=int(self.num_channels), l_name=self.l_name + "_transition_3")
        self.backward_dense_block_4 = BackwardDenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate, l_name=self.l_name + "_dense_block_4")


    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None, mask=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}

        # dense block 4
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_dense_block_4(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        # transition 3
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_transition_3(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        # dense block 3
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_dense_block_3(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        # transition 2
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_transition_2(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        # dense block 2
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_dense_block_2(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        # trnasition 1
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_transition_1(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        # dense block 1
        grad_in, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_dense_block_1(grad_in, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)
        grad_params = block_grad_params + grad_params
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        # MOD: replace pool with conv
        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv_p(grad_in, layer_inputs[self.l_name + '_conv_p'], layer_kernels[self.l_name + '_conv_p'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)


        '''
        # pool
        grad_in = self.backward_pool(grad_in, layer_inputs[self.l_name + "_pool"], layer_inputs[self.l_name + "_pool_argmax"], batch_size=config.BATCH_SIZE)
        '''

        # relu
        grad_in = self.backward_relu(grad_in, layer_inputs[self.l_name + "_relu"])

        # bn
        grad_in, grad_gamma, grad_beta = self.backward_bn(grad_in, layer_inputs[self.l_name + '_bn'], layer_kernels[self.l_name + '_bn'][0], layer_kernels[self.l_name + '_bn'][1], layer_kernels[self.l_name + '_bn_epsilon'])
        grad_params.insert(0, grad_beta)
        grad_params.insert(0, grad_gamma)

        # conv
        grad_in, grad_wt, grad_bias, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_conv(grad_in, layer_inputs[self.l_name + '_conv'], layer_kernels[self.l_name + '_conv'][0], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        return grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs




def backward_densenet_121():
    return BackwardDenseNet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 24, 16], compression_rate=0.5, drop_rate=0.5)
    #return BackwardDenseNet(num_init_features=32, growth_rate=32, block_layers=[6, 6, 6, 6], compression_rate=0.5, drop_rate=0.5)

