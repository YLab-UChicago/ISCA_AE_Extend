import tensorflow as tf
import math
from config import NUM_CLASSES
from models.inject_layers import *
from models.random_layers import *

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


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25, seed=None, l_name=None):
        super(SEBlock, self).__init__()
        self.l_name = l_name
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = InjectConv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same",
                                                  seed=seed,
                                                  l_name = l_name + "_reduce_conv")
        self.expand_conv = InjectConv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same",
                                                  seed=seed,
                                                  l_name = l_name + "_expand_conv")

    def call(self, inputs, inject=InjState.IDLE, inj_args=None, **kwargs):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        layer_inputs[self.l_name] = inputs

        branch = self.pool(inputs)

        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)

        layer_inputs[self.reduce_conv.l_name] = branch
        layer_kernels[self.reduce_conv.l_name] = self.reduce_conv.weights
        branch, b_conv = self.reduce_conv(branch, inject=inject, inj_args=inj_args)
        layer_outputs[self.reduce_conv.l_name] = b_conv

        layer_inputs[self.l_name + "_swish"] = branch
        branch = tf.nn.swish(branch)

        layer_inputs[self.expand_conv.l_name] = branch
        layer_kernels[self.expand_conv.l_name] = self.expand_conv.weights
        branch, b_conv = self.expand_conv(branch, inject=inject, inj_args=inj_args)
        layer_outputs[self.expand_conv.l_name] = b_conv

        layer_inputs[self.l_name + "_sigmoid"] = branch
        branch = tf.nn.sigmoid(branch)

        output = inputs * branch

        return output, layer_inputs, layer_kernels, layer_outputs


class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, seed=None,l_name=None):
        super(MBConv, self).__init__()
        self.l_name = l_name
        self.seed = seed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = InjectConv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False,
                                            seed=seed,
                                            l_name = self.l_name + "_conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same",
                                                      depthwise_initializer=tf.keras.initializers.GlorotNormal(seed=seed),
                                                      use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.se = SEBlock(input_channels=in_channels * expansion_factor,
                            seed=seed, l_name = self.l_name + "_se")
        self.conv2 = InjectConv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False,
                                            seed=seed,
                                            l_name = self.l_name + "_conv2")
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.9)
        #self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate, seed=seed)
        self.dropout = MyDropout(drop_connect_rate, seed)

    def call(self, inputs, training=None, inject=InjState.IDLE, inj_args=None, **kwargs):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}
        
        layer_inputs[self.conv1.l_name] = inputs
        layer_kernels[self.conv1.l_name] = self.conv1.weights
        x, x_conv = self.conv1(inputs, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv1.l_name] = x_conv


        layer_inputs[self.l_name + "_bn1"] = x
        x = self.bn1(x, training=training)
        layer_kernels[self.l_name + "_bn1"] = self.bn1.weights[:2]
        layer_kernels[self.l_name + "_bn1_epsilon"] = self.bn1.epsilon
        layer_outputs[self.l_name + "_bn1"] = x

        layer_inputs[self.l_name + "_swish"] = x
        x = tf.nn.swish(x)

        layer_inputs[self.l_name + "_dwconv"] = x
        layer_kernels[self.l_name + "_dwconv"] = self.dwconv.weights
        x = self.dwconv(x)
        layer_outputs[self.l_name + "_dwconv"] = x

        layer_inputs[self.l_name + "_bn2"] = x
        x = self.bn2(x, training=training)
        layer_kernels[self.l_name + "_bn2"] = self.bn2.weights[:2]
        layer_kernels[self.l_name + "_bn2_epsilon"] = self.bn2.epsilon
        layer_outputs[self.l_name + "_bn2"] = x



        x, block_inputs, block_kernels, block_outputs = self.se(x, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        layer_inputs[self.l_name + "_swish_1"] = x
        x = tf.nn.swish(x)

        layer_inputs[self.conv2.l_name] = x
        layer_kernels[self.conv2.l_name] = self.conv2.weights
        x, x_conv = self.conv2(x, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv2.l_name] = x_conv

        layer_inputs[self.l_name + "_bn3"] = x
        x = self.bn3(x, training=training)
        layer_kernels[self.l_name + "_bn3"] = self.bn3.weights[:2]
        layer_kernels[self.l_name + "_bn3_epsilon"] = self.bn3.epsilon
        layer_outputs[self.l_name + "_bn3"] = x


        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x_d = self.dropout(x, training)
                layer_inputs[self.l_name + "_dropout"] = tf.cast(tf.math.equal(x * (1. / (1. - self.drop_connect_rate)), x_d), tf.float32) 
                x = x_d

            x = tf.keras.layers.add([x, inputs])
        return x, layer_inputs, layer_kernels, layer_outputs


class MBConv_Block(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate, seed=None, l_name=None):
        super(MBConv_Block, self).__init__()
        self.l_name = l_name
        self.n_layers = layers
        self.layers = {}

        for i in range(self.n_layers):
            if i == 0:
                self.layers['mbconv_{}'.format(i)] = MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate,
                             seed=seed,
                             l_name = self.l_name + "_mbconv_{}".format(i))
            else:
                self.layers['mbconv_{}'.format(i)] = MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate,
                             seed=seed,
                             l_name = self.l_name + "_mbconv_{}".format(i))
 
    def call(self, x, training=None, inject=InjState.IDLE, inj_args=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        for i in range(self.n_layers):
            x, block_inputs, block_kernels, block_outputs = self.layers['mbconv_{}'.format(i)](x, training=training, inject=inject, inj_args=inj_args)
            layer_inputs.update(block_inputs)
            layer_kernels.update(block_kernels)
            layer_outputs.update(block_outputs)

        return x, layer_inputs, layer_kernels, layer_outputs 


class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate, seed=None):
        super(EfficientNet, self).__init__()
        self.seed = seed
        self.dropout_rate = dropout_rate
        
        self.data_augmentation = tf.keras.Sequential([
          MyRandomFlip("horizontal_and_vertical", seed=seed),
          tf.keras.layers.ZeroPadding2D(padding=(6,6)),
          MyRandomCrop(32,32,seed=seed),
          ])

        self.conv1 = InjectConv2D(filters=round_filters(32, width_coefficient),
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same",
                                            use_bias=False,
                                            seed=seed,
                                            l_name="conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.block1 = MBConv_Block(in_channels=round_filters(32, width_coefficient),
                                         out_channels=round_filters(16, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate,
                                         seed=seed,
                                         l_name="block1")
        self.block2 = MBConv_Block(in_channels=round_filters(16, width_coefficient),
                                         out_channels=round_filters(24, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                         seed=seed,
                                         l_name="block2")
        self.block3 = MBConv_Block(in_channels=round_filters(24, width_coefficient),
                                         out_channels=round_filters(40, width_coefficient),
                                         layers=round_repeats(2, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                         seed=seed,
                                         l_name="block3")
        self.block4 = MBConv_Block(in_channels=round_filters(40, width_coefficient),
                                         out_channels=round_filters(80, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                         seed=seed,
                                         l_name="block4")
        self.block5 = MBConv_Block(in_channels=round_filters(80, width_coefficient),
                                         out_channels=round_filters(112, width_coefficient),
                                         layers=round_repeats(3, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                         seed=seed,
                                         l_name="block5")
        self.block6 = MBConv_Block(in_channels=round_filters(112, width_coefficient),
                                         out_channels=round_filters(192, width_coefficient),
                                         layers=round_repeats(4, depth_coefficient),
                                         stride=2,
                                         expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate,
                                         seed=seed,
                                         l_name="block6")
        self.block7 = MBConv_Block(in_channels=round_filters(192, width_coefficient),
                                         out_channels=round_filters(320, width_coefficient),
                                         layers=round_repeats(1, depth_coefficient),
                                         stride=1,
                                         expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate,
                                         seed=seed,
                                         l_name="block7")

        self.conv2 = InjectConv2D(filters=round_filters(1280, width_coefficient),
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False,
                                            seed=seed,
                                            l_name="conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        #self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, seed=seed)
        self.dropout = MyDropout(dropout_rate, seed)
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))

    def call(self, inputs, training=None, mask=None, inject=InjState.IDLE, inj_args=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        if training:
            inputs = self.data_augmentation(inputs)

        layer_inputs["conv1"] = inputs
        layer_kernels["conv1"] = self.conv1.weights
        x, x_conv = self.conv1(inputs, inject=inject, inj_args=inj_args)
        layer_outputs["conv1"] = x_conv

        layer_inputs["bn1"] = x
        x = self.bn1(x, training=training)
        layer_kernels["bn1"] = self.bn1.weights[:2]
        layer_kernels["bn1_epsilon"] = self.bn1.epsilon
        layer_outputs["bn1"] = x

        layer_inputs["swish"] = x
        x = tf.nn.swish(x)

        x, block_inputs, block_kernels, block_outputs = self.block1(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.block2(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.block3(x, training=training, inject=inject, inj_args=inj_args) 
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.block4(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.block5(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.block6(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.block7(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        layer_inputs["conv2"] = x
        layer_kernels["conv2"] = self.conv2.weights
        x, x_conv = self.conv2(x, inject=inject, inj_args=inj_args)
        layer_outputs["conv2"] = x_conv

        bkwd_start = x
        outputs = {}

        x = self.bn2(x, training=training)

        x = tf.nn.swish(x)
        x = self.pool(x)
        x = self.dropout(x, training)
        #x = my_dropout(x, self.dropout_rate, seed=self.seed)
        x = self.fc(x)

        outputs['logits'] = x
        outputs['grad_start'] = bkwd_start
        outputs['obs'] = bkwd_start

        return outputs, layer_inputs, layer_kernels, layer_outputs



def get_efficient_net(width_coefficient, depth_coefficient, resolution, dropout_rate, seed):
    net = EfficientNet(width_coefficient=width_coefficient,
                       depth_coefficient=depth_coefficient,
                       dropout_rate=dropout_rate, 
                       drop_connect_rate=dropout_rate,
                       seed=seed)

    return net


def efficient_net_b0(seed):
    return get_efficient_net(1.0, 1.0, 224, 0, seed)


