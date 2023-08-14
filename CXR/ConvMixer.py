from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, MaxPool2D, GlobalAveragePooling2D, DepthwiseConv2D
from tensorflow.keras.models import Model

import tensorflow as tf


class ConvMixer(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """

        DIM = 3
        KERNEL_SIZE = (9, 9)
        PATCH_SIZE = (7,7)

        self.DEPTH = 3

        super().__init__()

        self.conv_1 = Conv2D(filters=DIM, kernel_size=PATCH_SIZE, strides=7)
        #self.gelu
        self.bn_1 = BatchNormalization()


        #rep

            #residual
        self.conv_res = Conv2D(filters=DIM, kernel_size=KERNEL_SIZE, padding="same", groups=DIM)
        #self.gelu
        self.bn_res = BatchNormalization()
            #end residual

        

        self.conv_repeat = Conv2D(filters=DIM, kernel_size=(1, 1))
        #self.gelu
        self.bn_repeat = BatchNormalization()
        
        #end rep


        self.avg_pool = AveragePooling2D(pool_size=(1,1))
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")




    def call(self, inputs):

        #initing patching step
        out = self.conv_1(inputs)
        out = tf.nn.gelu(out)
        out = self.bn_1(out)

        #Repeated Block - needs to be repeated depth number of times
        #Residual
        residual = self.conv_res(out)
        residual = tf.nn.gelu(residual)
        residual = self.bn_1(residual)

        residual_output = Add()[out, residual]
        #end Residual

        out2 = self.conv_repeat(residual_output)
        out2 = tf.nn.gelu(out2)
        out2 = self.bn_repeat(out2)
        ###

        out2 = self.avg_pool(out2)
        out2 = self.flat(out2)
        out2 = self.fc(out2)

        return out2