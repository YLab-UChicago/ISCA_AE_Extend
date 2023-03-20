############################
# Backward nf_resnet 
############################

import tensorflow as tf
from keras import backend as K
from models.backward_nf_resblock import BackwardNFResBlock
from models.backward_nf_layers import BackwardInjectWSConv2D 
from models.inject_layers import *


nonlinearities = {
    #"relu": lambda x: tf.keras.activations.relu(x) * 1.7139588594436646,
    "relu": lambda x, y: BackwardInjectReLU()(1.7139588594436646 * x, y),
}


class BackwardNF_ResNet(tf.keras.Model):

    #variant_dict = {'ResNet50': {'depth': [3, 4, 6, 3]}}
    variant_dict = {'ResNet50': {'depth': [2, 2, 2, 2]}}

    def __init__(self, num_classes, variant='ResNet50', width=4,
               alpha=0.2, stochdepth_rate=0.1, drop_rate=None,
               activation='relu', fc_init=None, 
               use_se=False, se_ratio=0.25,
               l_name='NF_ResNet'):

        super().__init__()
        self.l_name = l_name
        self.num_classes = num_classes
        self.variant = variant
        self.width = width
        # Get variant info
        block_params = self.variant_dict[self.variant]
        self.width_pattern = [item * self.width for item in [64, 128, 256, 512]]
        self.depth_pattern = block_params['depth']
        self.activation = nonlinearities[activation]
        if drop_rate is None:
            #self.drop_rate = block_params['drop_rate']
            self.drop_rate = 0
        else:
            self.drop_rate = drop_rate

        self.which_conv = BackwardInjectWSConv2D
        # Stem
        ch = int(16 * self.width)
        self.backward_initial_conv = self.which_conv(ch, kernel_size=3, strides=2,
                                            #padding='SAME', use_bias=False,
                                            padding='SAME',
                                            l_name=self.l_name+'initial_conv')
   
        # Body
        self.backward_blocks = []
        expected_std = 1.0
        num_blocks = sum(self.depth_pattern)
        index = 0  # Overall block index
        block_args = (self.width_pattern, self.depth_pattern, [1, 2, 2, 2])
        for block_width, stage_depth, stride in zip(*block_args):
            for block_index in range(stage_depth):
                # Scalar pre-multiplier so each block sees an N(0,1) input at init
                beta = 1./ expected_std
                # Block stochastic depth drop-rate
                block_stochdepth_rate = stochdepth_rate * index / num_blocks
                self.backward_blocks += [BackwardNFResBlock(ch, block_width,
                                       stride=stride if block_index == 0 else 1,
                                       beta=beta, alpha=alpha,
                                       activation=self.activation,
                                       which_conv=self.which_conv,
                                       stochdepth_rate=block_stochdepth_rate,
                                       use_se=use_se,
                                       se_ratio=se_ratio,
                                       l_name=self.l_name + "_block_{}".format(index),
                                       )]
                ch = block_width
                index += 1
                # Reset expected std but still give it 1 block of growth
                if block_index == 0:
                    expected_std = 1.0
                expected_std = (expected_std **2 + alpha**2)**0.5
    
        '''
        # MOD: No need to include in backward
        # Head. By default, initialize with N(0, 0.01)
        #if fc_init is None:
        #    fc_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)
        #self.fc = BackwardInjectDense(self.num_classes, kernel_initializer=fc_init, use_bias=True, activation=tf.nn.softmax)
        '''

    def call(self, grad_in, layer_inputs, layer_kernels, inject=None, inj_args=None):
        grad_params = []
        bkwd_layer_inputs = {}
        bkwd_layer_kernels = {}
        bkwd_layer_outputs = {}


        grad_out = grad_in
        for i, backward_block in reversed(list(enumerate(self.backward_blocks))):
            grad_out, block_grad_params, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = backward_block(grad_out, layer_inputs, layer_kernels, inject=inject, inj_args=inj_args)

            grad_params = block_grad_params + grad_params
            bkwd_layer_inputs.update(bkwd_block_inputs)
            bkwd_layer_kernels.update(bkwd_block_kernels)
            bkwd_layer_outputs.update(bkwd_block_outputs)

        grad_out, grad_wt, grad_bias, grad_gain, bkwd_block_inputs, bkwd_block_kernels, bkwd_block_outputs = self.backward_initial_conv(grad_out, layer_inputs[self.l_name + "_init_conv"], layer_kernels[self.l_name + "_init_conv"], inject=inject, inj_args=inj_args)
        grad_params.insert(0, grad_bias)
        grad_params.insert(0, grad_wt)
        grad_params.insert(0, grad_gain)
        bkwd_layer_inputs.update(bkwd_block_inputs)
        bkwd_layer_kernels.update(bkwd_block_kernels)
        bkwd_layer_outputs.update(bkwd_block_outputs)

        return grad_params, bkwd_layer_inputs, bkwd_layer_kernels, bkwd_layer_outputs
