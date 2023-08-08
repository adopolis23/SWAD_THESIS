from keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from ModelGen import Generate_Model_1


'''
SWA test on MNIST data
MNIST data is set of handwriten digits and labels.
each digit is 28x28 greyscale.

SWA takes the weights after certain epochs of training
and averages them up after all training is completed.

the variable 'percent_save' allows you to determin what 
percentage of training you would like to pass before you 
start saving epoch weights. For example a value of 0.25 means
that only the last 25% of epochs will be averaged together. 



Brandon Weinhofer
U16425289
weinhofer@usf.edu
'''




#10 output classes nums 0-9
num_classes = 10

#28 x 28 greyscale images
input_shape = (28, 28, 1)

#model parameters
batch_size = 128
learning_rate = 0.001
epochs = 20

#SWA parameters
percent_save = 0.25 #what percentage of epochs towards end of training to save





#download data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)


#normalize images to 0 - 1 range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


#create the velidation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)


        



#build the model
#model is defined in seperate file
model = Generate_Model_1(num_classes, input_shape)

print(model.summary())




weights = []
new_weights = list()



#create callback for weight averaging
class swa_callback(keras.callbacks.Callback):

    def __init__(self):
        #list to track loss over training
        self.loss_tracker = []

        self.epoch_tracker = 1


    #function called at the end of every epoch
    def on_epoch_end(self, batch, logs=None):

        #if we are on an epoch number greater than threshold when we want to start saving
        if self.epoch_tracker >= int((1-percent_save) * epochs):
            print("\nSaving weights from epoch {} with loss {}".format(self.epoch_tracker, logs["val_loss"]))

            #save the current loss and save the weights
            self.loss_tracker.append(logs["val_loss"])
            weights.append(model.get_weights())


        self.epoch_tracker += 1




    #function called at end of training
    def on_train_end(self, logs=None):
        print("\nEnd of Training; Averaging Weights.")

        #averages up all weights and stores them in new_weights
        for weights_list_tuple in zip(*weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        
        #if we have a new weight we set the models weights to this new average
        if len(new_weights) > 0:
            print("\nSetting new model weights.\n")
            model.set_weights(new_weights)





#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



#train the model
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_valid, y_valid),
              shuffle=True,
              callbacks=[swa_callback()])



#model evaluation
scores = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])