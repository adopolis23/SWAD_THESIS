from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, MaxPool2D, GlobalAveragePooling2D, DepthwiseConv2D
from tensorflow.keras.models import Model

import tensorflow as tf




class resBlock(Model):

    def __init__(self, DIM, KERNEL_SIZE):
  
        super().__init__()
        self.DIM = DIM
        self.KERNEL_SIZE = KERNEL_SIZE

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

    def call(self, inputs):
        #Repeated Block - needs to be repeated depth number of times
        #Residual
        residual = self.conv_res(inputs)
        residual = tf.nn.gelu(residual)
        residual = self.bn_1(residual)

        residual_output = Add()([inputs, residual])
        #end Residual

        out2 = self.conv_repeat(residual_output)
        out2 = tf.nn.gelu(out2)
        out2 = self.bn_repeat(out2)
        return out2
        ###




class ConvMixer(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """

        DIM = 512
        KERNEL_SIZE = (9, 9)
        PATCH_SIZE = (7,7)

        self.DEPTH = 4

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
        self.rb1 = resBlock(DIM, KERNEL_SIZE)
        self.rb2 = resBlock(DIM, KERNEL_SIZE)
        self.rb3 = resBlock(DIM, KERNEL_SIZE)
        self.rb4 = resBlock(DIM, KERNEL_SIZE)
        self.rb5 = resBlock(DIM, KERNEL_SIZE)
        self.rb6 = resBlock(DIM, KERNEL_SIZE)
        self.rb7 = resBlock(DIM, KERNEL_SIZE)
        self.rb8 = resBlock(DIM, KERNEL_SIZE)

        self.rb9 = resBlock(DIM, KERNEL_SIZE)
        self.rb10 = resBlock(DIM, KERNEL_SIZE)
        self.rb11 = resBlock(DIM, KERNEL_SIZE)
        self.rb12 = resBlock(DIM, KERNEL_SIZE)
        self.rb13 = resBlock(DIM, KERNEL_SIZE)
        self.rb14 = resBlock(DIM, KERNEL_SIZE)
        self.rb15 = resBlock(DIM, KERNEL_SIZE)
        self.rb16 = resBlock(DIM, KERNEL_SIZE)

        self.rb17 = resBlock(DIM, KERNEL_SIZE)
        self.rb18 = resBlock(DIM, KERNEL_SIZE)
        self.rb19 = resBlock(DIM, KERNEL_SIZE)
        self.rb20 = resBlock(DIM, KERNEL_SIZE)


        self.avg_pool = AveragePooling2D(pool_size=(1,1))
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")




    def call(self, inputs):

        #initing patching step
        out = self.conv_1(inputs)
        out = tf.nn.gelu(out)
        out2 = self.bn_1(out)

        #out2 = self.rb1(out2)
        #blocks = [self.rb1, self.rb2, self.rb3, self.rb4, self.rb5, self.rb6, self.rb7, self.rb8, self.rb9, self.rb10, self.rb11, self.rb12, self.rb13, self.rb14, self.rb15, self.rb16, self.rb17, self.rb18, self.rb19, self.rb20]
        blocks = [self.rb1, self.rb2, self.rb3, self.rb4, self.rb5, self.rb6, self.rb7, self.rb8, self.rb9, self.rb10, self.rb11, self.rb12]


        for block in blocks:
            out2 = block(out2)

        out2 = self.avg_pool(out2)
        out2 = self.flat(out2)
        out2 = self.fc(out2)

        return out2