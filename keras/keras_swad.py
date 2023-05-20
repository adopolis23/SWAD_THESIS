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
SWAD is a continuation of SWA where weights are averaged much more
frequently than in vanilla SWA. Weights averaging can be controlled by one
or more of the following parameters: Percent_save - the percentage of the
training that should be saved for the average ex. 0.25 means that the last
25% of training will be saved. Loss_threshold - Weights will only be saved once
the loss has dropped below this threshold. Max_weights_to_save - max amount of 
weights that can be considered in the average. Save_stride - the gap in iterations
between weights saves.

These saved weights are then averaged together then the weights in the 
model are updated with this new average. 

TODO:
Write the learning rate scheduler to add a cyclic learning rate 
option

Write a validation method so that SWAD can use validation loss rather
than training loss. 


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

#SWAD parameters
percent_save = 0.25 #what percentage of epochs to comeplete before saving weights
loss_threshold = 0.3500 #start saving weights once loss reaches a certain threshold
max_weights_to_save = 100000 #max weights that can be saved
save_stride = 30 #how many iteration gap between saves




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
#model definition in modelGen file
model = Generate_Model_1(num_classes, input_shape)

print(model.summary())




weights = []
new_weights = list()



#create callback for weight averaging
class swad_callback(keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.loss_tracker = []

        self.epoch_tracker = 1


    #function called at the end of every batch
    def on_train_batch_end(self, batch, logs=None):

        if logs["loss"] <= loss_threshold and len(weights) <= max_weights_to_save and batch % save_stride == 0 and self.epoch_tracker >= int((1-percent_save) * epochs):
            print("\nSaving weights from batch {} with loss {}".format(batch, logs["loss"]))

            #save loss and weights for this batch
            self.loss_tracker.append(logs["loss"])
            weights.append(model.get_weights())

        self.epoch_tracker += 1




    #function called at the end of training
    def on_train_end(self, logs=None):
        print("\nEnd of Training; Averaging Weights.")

        #average up all saved weights and store them in new_weights
        for weights_list_tuple in zip(*weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        
        #set model weights to new average
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
              callbacks=[swad_callback()])



#model evaluation
scores = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])