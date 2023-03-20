import tensorflow as tf
import math
from config import NUM_CLASSES
from models.inject_layers import *
from models.random_layers import *


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate, drop_rate, seed, l_name=None):
        super(BottleNeck, self).__init__()
        self.l_name = l_name
        self.drop_rate = drop_rate
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv1 = InjectConv2D(filters=4 * growth_rate,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            seed=seed,
                                            l_name=self.l_name + "_conv1")
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv2 = InjectConv2D(filters=growth_rate,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            seed=seed,
                                            l_name=self.l_name + "_conv2")

        self.drop_rate = drop_rate
        self.seed = seed
        #self.dropout = tf.keras.layers.Dropout(rate=drop_rate, seed=seed)
        self.dropout = MyDropout(drop_rate, seed)

    def call(self, inputs, training=None, inject=None, inj_args=None, **kwargs):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        layer_inputs[self.l_name + "_bn1"] = inputs
        x = self.bn1(inputs, training=training)
        layer_kernels[self.l_name + "_bn1"] = self.bn1.weights[:2]
        layer_kernels[self.l_name + "_bn1_epsilon"] = self.bn1.epsilon
        layer_outputs[self.l_name + "_bn1"] = x


        layer_inputs[self.l_name + "_relu"] = x
        x = tf.nn.relu(x)

        layer_inputs[self.conv1.l_name] = x
        layer_kernels[self.conv1.l_name] = self.conv1.weights
        x, x_conv = self.conv1(x, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv1.l_name] = x_conv


        layer_inputs[self.l_name + "_bn2"] = x
        x = self.bn2(x, training=training)
        layer_kernels[self.l_name + "_bn2"] = self.bn2.weights[:2]
        layer_kernels[self.l_name + "_bn2_epsilon"] = self.bn2.epsilon
        layer_outputs[self.l_name + "_bn2"] = x


        layer_inputs[self.l_name + "_relu_1"] = x
        x = tf.nn.relu(x)

        layer_inputs[self.conv2.l_name] = x
        layer_kernels[self.conv2.l_name] = self.conv2.weights
        x, x_conv = self.conv2(x, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv2.l_name] = x_conv

        x_d = self.dropout(x, training)
        layer_inputs[self.l_name + "_dropout"] = tf.cast(tf.math.equal(x * (1. / (1. - self.drop_rate)), x_d), tf.float32) 
        x = x_d

        return x, layer_inputs, layer_kernels, layer_outputs


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate, seed, l_name=None):
        super(DenseBlock, self).__init__()
        self.l_name = l_name
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.features_list = []
        self.bottle_necks = []
        for i in range(self.num_layers):
            self.bottle_necks.append(BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate, seed=seed, l_name=self.l_name + "_bottleneck_{}".format(i)))

    def call(self, inputs, training=None, inject=None, inj_args=None, **kwargs):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        self.features_list.append(inputs)
        x = inputs

        layer_inputs[self.l_name + "_input_k"] = x.get_shape()[-1]

        for i in range(self.num_layers):
            y, block_inputs, block_kernels, block_outputs = self.bottle_necks[i](x, training=training, inject=inject, inj_args=inj_args)
            layer_inputs.update(block_inputs)
            layer_kernels.update(block_kernels)
            layer_outputs.update(block_outputs)

            self.features_list.append(y)
            x = tf.concat(self.features_list, axis=-1)
        self.features_list.clear()
        return x, layer_inputs, layer_kernels, layer_outputs


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels, seed, l_name=None):
        super(TransitionLayer, self).__init__()
        self.l_name = l_name
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.conv = InjectConv2D(filters=out_channels,
                                           kernel_size=(1, 1),
                                           strides=1,
                                           padding="same",
                                           seed=seed,
                                           l_name = self.l_name + "_conv")

        self.conv_p = InjectConv2D(filters=out_channels,
                                           kernel_size=(3, 3),
                                           strides=2,
                                           padding="same",
                                           seed=seed,
                                           l_name = self.l_name + "_conv_p")

        '''
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                              strides=2,
                                              padding="same")

        self.pool = tf.keras.layers.Lambda(
                          lambda x: tf.nn.max_pool_with_argmax(x, ksize=[2,2], strides=2, padding='SAME'), 
                          name="pool", dtype=tf.float32)
        '''

    def call(self, inputs, training=None, inject=None, inj_args=None, **kwargs):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        layer_inputs[self.l_name + "_bn"] = inputs
        x = self.bn(inputs, training=training)
        layer_kernels[self.l_name + "_bn"] = self.bn.weights[:2]
        layer_kernels[self.l_name + "_bn_epsilon"] = self.bn.epsilon
        layer_outputs[self.l_name + "_bn"] = x

        layer_inputs[self.l_name + "_relu"] = x
        x = tf.nn.relu(x)

        layer_inputs[self.conv.l_name] = x
        layer_kernels[self.conv.l_name] = self.conv.weights
        x, x_conv = self.conv(x, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv.l_name] = x_conv

        # MOD: replace pool with conv_p
        layer_inputs[self.conv_p.l_name] = x
        layer_kernels[self.conv_p.l_name] = self.conv_p.weights
        x, x_conv = self.conv_p(x, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv_p.l_name] = x_conv

        '''
        layer_inputs[self.l_name + "_pool"] = x
        #x = self.pool(x)
        #x, argmax = tf.nn.max_pool_with_argmax(x, ksize=[2,2], strides=2, padding='SAME')
        x, argmax = self.pool(x)
        layer_inputs[self.l_name + "_pool_argmax"] = argmax
        '''

        return x, layer_inputs, layer_kernels, layer_outputs


class DenseNet(tf.keras.Model):
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate, seed, l_name='densenet'):
        super(DenseNet, self).__init__()
        self.l_name = l_name

        self.data_augmentation = tf.keras.Sequential([
          MyRandomFlip("horizontal_and_vertical", seed=seed),
          tf.keras.layers.ZeroPadding2D(padding=(6,6)),
          MyRandomCrop(32,32,seed=seed),
          ])

        self.conv = InjectConv2D(filters=num_init_features,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           padding="same",
                                           seed=seed,
                                           l_name=self.l_name + '_conv')
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)

        '''
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding="same")

        self.pool = tf.keras.layers.Lambda(
                          lambda x: tf.nn.max_pool_with_argmax(x, ksize=[3,3], strides=2, padding='SAME'), 
                          name="pool", dtype=tf.float32)
        '''
        self.conv_p = InjectConv2D(filters=num_init_features,
                                           kernel_size=(3, 3),
                                           strides=2,
                                           padding="same",
                                           seed=seed,
                                           l_name=self.l_name + '_conv_p')

        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate, seed=seed, l_name=self.l_name + '_dense_block_1')
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels), seed=seed, l_name=self.l_name + '_transition_1')
        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate, seed=seed, l_name=self.l_name + "_dense_block_2")
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels), seed=seed, l_name=self.l_name + "_transition_2")
        self.dense_block_3 = DenseBlock(num_layers=block_layers[2], growth_rate=growth_rate, drop_rate=drop_rate, seed=seed, l_name=self.l_name + "_dense_block_3")
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels), seed=seed, l_name=self.l_name + "_transition_3")
        self.dense_block_4 = DenseBlock(num_layers=block_layers[3], growth_rate=growth_rate, drop_rate=drop_rate, seed=seed, l_name=self.l_name + "_dense_block_4")

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))

    def call(self, inputs, training=None, inject=None, inj_args=None, mask=None):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        outputs = {}

        if training:
            inputs = self.data_augmentation(inputs)

        layer_inputs[self.conv.l_name] = inputs
        layer_kernels[self.conv.l_name] = self.conv.weights
        x, x_conv = self.conv(inputs, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv.l_name] = x_conv

        layer_inputs[self.l_name + "_bn"] = x
        x = self.bn(x, training=training)
        layer_kernels[self.l_name + "_bn"] = self.bn.weights[:2]
        layer_kernels[self.l_name + "_bn_epsilon"] = self.bn.epsilon
        layer_outputs[self.l_name + "_bn"] = x

        layer_inputs[self.l_name + "_relu"] = x
        x = tf.nn.relu(x)

        # MOD: replace pooling
        layer_inputs[self.conv_p.l_name] = x
        layer_kernels[self.conv_p.l_name] = self.conv_p.weights
        x, x_conv = self.conv_p(x, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv_p.l_name] = x_conv

        '''
        layer_inputs[self.l_name + "_pool"] = x
        #x = self.pool(x)
        #x, argmax = tf.nn.max_pool_with_argmax(x, ksize=[3,3], strides=2, padding='SAME')
        x, argmax = self.pool(x)
        layer_inputs[self.l_name + "_pool_argmax"] = argmax
        '''

        x, block_inputs, block_kernels, block_outputs = self.dense_block_1(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        outputs['obs'] = block_outputs['densenet_dense_block_1_bottleneck_0_conv2']

        x, block_inputs, block_kernels, block_outputs = self.transition_1(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)


        x, block_inputs, block_kernels, block_outputs = self.dense_block_2(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)


        x, block_inputs, block_kernels, block_outputs = self.transition_2(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)


        x, block_inputs, block_kernels, block_outputs = self.dense_block_3(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)


        x, block_inputs, block_kernels, block_outputs = self.transition_3(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)


        x, block_inputs, block_kernels, block_outputs = self.dense_block_4(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        grad_start = x
        x = self.avgpool(x)
        x = self.fc(x)

        outputs['logits'] = x
        outputs['grad_start'] = grad_start

        return outputs, layer_inputs, layer_kernels, layer_outputs


def densenet_121(seed):
    return DenseNet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 24, 16], compression_rate=0.5, drop_rate=0.5, seed=seed)
    #return DenseNet(num_init_features=8, growth_rate=8, block_layers=[3, 3, 3, 3], compression_rate=0.5, drop_rate=0.5)

