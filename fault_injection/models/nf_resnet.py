##############################
# nf_resnet implementation
##############################

import tensorflow as tf
from keras import backend as K
from models.nf_resblock import NFResBlock
from models.nf_layers import InjectWSConv2D 
from models.random_layers import *


nonlinearities = {
    "relu": lambda x: tf.keras.activations.relu(x) * 1.7139588594436646,
}


class NF_ResNet(tf.keras.Model):

    #variant_dict = {'ResNet50': {'depth': [3, 4, 6, 3]}}
    variant_dict = {'ResNet50': {'depth': [2, 2, 2, 2]}}

    def __init__(self, num_classes, seed, variant='ResNet50', width=4,
               alpha=0.2, stochdepth_rate=0.1, drop_rate=None,
               activation='relu', fc_init=None, 
               use_se=False, se_ratio=0.5, 
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
        self.seed = seed
        if drop_rate is None:
            #self.drop_rate = block_params['drop_rate']
            self.drop_rate = 0
        else:
            self.drop_rate = drop_rate

        self.data_augmentation = tf.keras.Sequential([
          MyRandomFlip("horizontal_and_vertical", seed=seed),
          tf.keras.layers.ZeroPadding2D(padding=(6,6)),
          MyRandomCrop(32,32,seed=seed),
          ])

        self.which_conv = InjectWSConv2D
        # Stem
        ch = int(16 * self.width)
        self.initial_conv = self.which_conv(seed, ch, kernel_size=3, strides=2,
                                            #padding='SAME', use_bias=False,
                                            padding='SAME',
                                            name='init_conv',
                                            l_name=self.l_name + "_init_conv")
   
        # Body
        self.blocks = []
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
                self.blocks += [NFResBlock(ch, block_width,
                                       stride=stride if block_index == 0 else 1,
                                       beta=beta, alpha=alpha,
                                       seed=seed,
                                       activation=self.activation,
                                       which_conv=self.which_conv,
                                       stochdepth_rate=block_stochdepth_rate,
                                       use_se=use_se,
                                       se_ratio=se_ratio,
                                       l_name=self.l_name + "_block_{}".format(index)
                                       )]
                ch = block_width
                index += 1
                # Reset expected std but still give it 1 block of growth
                if block_index == 0:
                    expected_std = 1.0
                expected_std = (expected_std **2 + alpha**2)**0.5
    
        # Head. By default, initialize with N(0, 0.01)
        if fc_init is None:
            #fc_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)
            fc_init = tf.keras.initializers.GlorotNormal(seed=seed)
        self.fc = tf.keras.layers.Dense(self.num_classes, kernel_initializer=fc_init, use_bias=True, activation=tf.nn.softmax)


    def call(self, x, training = True, inject=None, inj_args=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        """Return the output of the final layer without any [log-]softmax."""
        # Stem
        outputs = {}

        if training:
            x = self.data_augmentation(x)

        layer_inputs[self.l_name + "_init_conv"] = x
        out, conv_out, std_w, w_max_bitmap = self.initial_conv(x, inject=inject, inj_args=inj_args)
        layer_kernels[self.l_name + "_init_conv"] = self.initial_conv.weights[:3] + [std_w, w_max_bitmap]
        layer_outputs[self.l_name + "_init_conv"] = conv_out

        # Blocks
        for i, block in enumerate(self.blocks): 
            out, block_inputs, block_kernels, block_outputs = block(out, training = training, inject=inject, inj_args=inj_args)
            layer_inputs.update(block_inputs)
            layer_kernels.update(block_kernels)
            layer_outputs.update(block_outputs)

        outputs['grad_start'] = out

        # Final-conv->activation, pool, dropout, classify
        pool = tf.math.reduce_mean(self.activation(out), [1,2])
        outputs['pool'] = pool
        # Optionally apply dropout
        if self.drop_rate > 0.0 and training:
            #pool = tf.keras.layers.Dropout(self.drop_rate)(pool)
            pool = MyDropout(self.drop_rate, self.seed)(pool)
        outputs['logits'] = self.fc(pool)
        return outputs, layer_inputs, layer_kernels, layer_outputs
