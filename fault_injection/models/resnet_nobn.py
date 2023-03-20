import tensorflow as tf
from config import NUM_CLASSES
from models.nobn_residual_block import BasicBlocks
from models.inject_layers import *
from models.random_layers import *

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params, seed):
        super(ResNetTypeI, self).__init__()

        self.data_augmentation = tf.keras.Sequential([
          #tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed, force_generator=True),
          MyRandomFlip("horizontal_and_vertical", seed=seed),
          #tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
          tf.keras.layers.ZeroPadding2D(padding=(6,6)),
          # TODO: add seed to this RandomCrop function
          MyRandomCrop(32,32,seed=seed),
          #tf.keras.layers.experimental.preprocessing.RandomCrop(32,32),
          #tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.3),
          #tf.keras.layers.experimental.preprocessing.RandomContrast(0.2)
          ])

        self.conv1 = InjectConv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same",
                                            seed=seed,
                                            l_name="conv1")
        #self.bn1 = InjectBatchNormalization()
        '''
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        '''

        self.layer1 = BasicBlocks(filter_num=64,
                                  blocks=layer_params[0],
                                  seed=seed,
                                  l_name="basicblock_1")
        self.layer2 = BasicBlocks(filter_num=128,
                                  blocks=layer_params[1],
                                  stride=2,
                                  seed=seed,
                                  l_name="basicblock_2")
        self.layer3 = BasicBlocks(filter_num=256,
                                  blocks=layer_params[2],
                                  stride=2,
                                  seed=seed,
                                  l_name="basicblock_3")
        self.layer4 = BasicBlocks(filter_num=512,
                                  blocks=layer_params[3],
                                  stride=2,
                                  seed=seed,
                                  l_name="basicblock_4")

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        print("Seed value is {}".format(seed))
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))

    def call(self, inputs, training=None, mask=None, inject=None, inj_args=None):
        outputs = {}

        layer_inputs = {}
        layer_kernels = {}
        layer_outputs = {}


        if training:
            inputs = self.data_augmentation(inputs)

        outputs['in'] = inputs

        layer_inputs[self.conv1.l_name] = inputs
        layer_kernels[self.conv1.l_name] = self.conv1.weights
        x, conv_x = self.conv1(inputs)
        layer_outputs[self.conv1.l_name] = conv_x

        '''
        layer_inputs['bn1']= x
        layer_kernels['bn1'] = self.bn1.weights[:2]
        layer_kernels['bn1_epsilon'] = self.bn1.epsilon
        layer_kernels['bn1_moving_mean_var'] = self.bn1.weights[2:]
        x = self.bn1(x, training=training)
        layer_outputs['bn1'] = x
        '''

        layer_inputs['relu1'] = x
        x = tf.nn.relu(x)
        layer_outputs['relu1'] = x

        #x = self.pool1(x)


        x, block_inputs, block_kernels, block_outputs = self.layer1(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.layer2(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.layer3(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        x, block_inputs, block_kernels, block_outputs = self.layer4(x, training=training, inject=inject, inj_args=inj_args)
        layer_inputs.update(block_inputs)
        layer_kernels.update(block_kernels)
        layer_outputs.update(block_outputs)

        outputs['grad_start'] = x

        x = self.avgpool(x)
        output = self.fc(x)

        outputs['logits'] = output

        return outputs, layer_inputs, layer_kernels, layer_outputs




def resnet_18_nobn(seed):
    return ResNetTypeI(layer_params=[2, 2, 2, 2], seed=seed)

