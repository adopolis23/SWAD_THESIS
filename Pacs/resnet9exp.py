from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, MaxPool2D, GlobalAveragePooling2D, DepthwiseConv2D

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import tensorflow as tf

from tensorflow import keras








class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        ROBUST_KERNEL_SIZE = (7, 7)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()

        #changing block to depthwise conv
        self.conv_2 = DepthwiseConv2D(strides=self.__strides[1],
                             kernel_size=ROBUST_KERNEL_SIZE, padding="same")
        

        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    #altered to remove relu and bn layers
    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        #x = tf.nn.relu(x)
        x = self.conv_2(x)
        #x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            #res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out

class ResNet9_exp(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """

        #changed to be patchify step
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(96, (16, 16), strides=16,
                             padding="same", kernel_initializer="he_normal")
        

        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")




        self.res_1_1 = ResnetBlock(96)
        self.res_1_2 = ResnetBlock(96)
        self.res_2_1 = ResnetBlock(192, down_sample=True)
        self.res_2_2 = ResnetBlock(192)
        self.res_3_1 = ResnetBlock(384, down_sample=True)
        self.res_3_2 = ResnetBlock(384)
        self.res_4_1 = ResnetBlock(768, down_sample=True)
        self.res_4_2 = ResnetBlock(768)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):

        #initing patching step
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)


        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_2_1, self.res_3_1, self.res_4_1]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out