import tensorflow as tf
from models.inject_utils import *
from models.inject_layers import *
from models.random_layers import *

class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, seed=None, drop_out_rate=0.15, l_name=None):
        super(BasicBlock, self).__init__()

        self.l_name = l_name
        self.stride = stride
        self.drop_rate = drop_out_rate
        self.seed = seed

        self.conv1 = InjectConv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same",
                                            seed=seed,
                                            l_name=self.l_name + "_conv1")
        self.bn1 = InjectBatchNormalization(momentum=0.9)
        #self.dropout1 = tf.keras.layers.Dropout(rate=0.15, seed=seed)
        self.dropout1 = MyDropout(self.drop_rate, self.seed)

        self.conv2 = InjectConv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same",
                                            seed=seed,
                                            l_name=self.l_name + "_conv2")
        self.bn2 = InjectBatchNormalization(momentum=0.9)
        #self.dropout2 = tf.keras.layers.Dropout(rate=0.15, seed=seed)
        self.dropout2 = MyDropout(self.drop_rate, self.seed)

        if stride != 1:
            self.downsample_conv = InjectConv2D(filters=filter_num,
                                                kernel_size=(1, 1),
                                                strides=stride,
                                                padding='same',
                                                seed=seed,
                                                l_name=self.l_name + "_downsample")
            self.bn3 = InjectBatchNormalization(momentum=0.9)
        else:
            self.downsample_conv = lambda x: x
            self.bn3 = lambda x: x


    def call(self, inputs, training=None, inject=None, inj_args=None, **kwargs):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        if self.stride != 1:
            layer_inputs[self.downsample_conv.l_name] = inputs
            layer_kernels[self.downsample_conv.l_name] = self.downsample_conv.weights
            residual, conv_out = self.downsample_conv(inputs, inject=inject, inj_args=inj_args)
            layer_outputs[self.downsample_conv.l_name] = conv_out

            layer_inputs[self.l_name + "_bn3"] = residual
            layer_kernels[self.l_name + "_bn3"] = self.bn3.weights[:2]
            layer_kernels[self.l_name + "_bn3_epsilon"] = self.bn3.epsilon
            layer_kernels[self.l_name + "_bn3_moving_mean_var"] = self.bn3.weights[2:]
            residual = self.bn3(residual, training=training)
            layer_outputs[self.l_name + "_bn3"] = residual

        else:
            residual = self.downsample_conv(inputs)

        layer_inputs[self.conv1.l_name] = inputs
        layer_kernels[self.conv1.l_name] = self.conv1.weights
        x, conv_x = self.conv1(inputs, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv1.l_name] = conv_x


        layer_inputs[self.l_name + "_bn1"] = x
        layer_kernels[self.l_name + "_bn1"] = self.bn1.weights[:2]
        layer_kernels[self.l_name + "_bn1_epsilon"] = self.bn1.epsilon
        layer_kernels[self.l_name + "_bn1_moving_mean_var"] = self.bn1.weights[2:]
        x = self.bn1(x, training=training)
        layer_outputs[self.l_name + "_bn1"] = x

        layer_inputs[self.l_name + "_relu1"] = x
        x = tf.nn.relu(x)
        layer_outputs[self.l_name + "_relu1"] = x

        # TODO: add layer_input/outputs
        x_d = self.dropout1(x, training)
        layer_inputs[self.l_name + "_dropout1"] = tf.cast(tf.math.equal(x * (1. / (1. - self.drop_rate)), x_d), tf.float32)
        x = x_d

        layer_inputs[self.conv2.l_name] = x
        layer_kernels[self.conv2.l_name] = self.conv2.weights
        x, conv_x = self.conv2(x, inject=inject, inj_args=inj_args)
        layer_outputs[self.conv2.l_name] = conv_x


        layer_inputs[self.l_name + "_bn2"] = x
        layer_kernels[self.l_name + "_bn2"] = self.bn2.weights[:2]
        layer_kernels[self.l_name + "_bn2_epsilon"] = self.bn2.epsilon
        layer_kernels[self.l_name + "_bn2_moving_mean_var"] = self.bn2.weights[2:]
        x = self.bn2(x, training=training)
        layer_outputs[self.l_name + "_bn2"] = x


        x = tf.keras.layers.add([residual, x])
        layer_inputs[self.l_name + "_relu_add"] = x
        output = tf.nn.relu(x)
        layer_outputs[self.l_name + "_relu_add"] = output

        x_d = self.dropout2(output, training)
        layer_inputs[self.l_name + "_dropout2"] = tf.cast(tf.math.equal(output * (1. / (1. - self.drop_rate)), x_d), tf.float32) 
        output = x_d
 
        return output, layer_inputs, layer_kernels, layer_outputs



class BasicBlocks(tf.keras.layers.Layer):
    def __init__(self, filter_num, blocks, stride=1, seed=None, drop_out_rate=0.15, l_name=None):
        super(BasicBlocks, self).__init__()
        self.l_name = l_name
        self.basics = []
        self.basics.append(BasicBlock(filter_num, stride=stride, seed=seed, drop_out_rate=drop_out_rate, l_name=self.l_name + "_basic_0"))

        for i in range(1, blocks):
            self.basics.append(BasicBlock(filter_num, stride=1, seed=seed, drop_out_rate=drop_out_rate, l_name=self.l_name + "_basic_{}".format(i)))

    def call(self, x, training=None, inject=None, inj_args=None, **kwargs):
        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}

        for i in range(len(self.basics)):
            basic = self.basics[i]
            x, block_inputs, block_kernels, block_outputs = basic(x, training=training, inject=inject, inj_args=inj_args)
            layer_inputs.update(block_inputs)
            layer_kernels.update(block_kernels)
            layer_outputs.update(block_outputs)

        return x, layer_inputs, layer_kernels, layer_outputs



