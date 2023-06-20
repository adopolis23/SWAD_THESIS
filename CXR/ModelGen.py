from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow import keras


#function returns a model defined here
def Generate_Model_1(num_classes, input_shape):
    model = Sequential()

    #convolutional layer max pool and dropout
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #convolutional layer max pool and dropout
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    #flatten model for upcoming FC layers
    model.add(Flatten())

    #256 node fully connected layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))


    #output softmax layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


#much simpler version of model 1
def Generate_Model_2(num_classes, input_shape):
    model = Sequential()

    #convolutional layer max pool and dropout
    model.add(Conv2D(32, (3,3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    #convolutional layer max pool and dropout
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))


    #flatten model for upcoming FC layers
    model.add(Flatten())

    #256 node fully connected layer
    #model.add(Dense(256))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))


    #output softmax layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


#LeNet5 model - found several definitions of this online this might not be the most accurate
def LeNet5(num_classes, input_shape):
    model = Sequential()

    model.add(Conv2D(6, (5,5), input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(AveragePooling2D())

    model.add(Conv2D(16, (5,5), input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(120))
    model.add(Activation('relu'))

    model.add(Dense(84))
    model.add(Activation('relu'))

    #output softmax layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


#TODO create a resnet model