from keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from ModelGen import LeNet5, Generate_Model_2



'''
Uses an ensemble approach on the MNIST dataset. I save weights from some epochs 
then create a number of models for those weights and have them vote on an answer. 

Right now after training it takes the saved weights and pickes N weight sets with the 
lowest loss to create the ensemble. Where N is the number of models specified.

Make sure that the number of epochs is higher than the number of models you want.


Brandon Weinhofer
U16425289
weinhofer@usf.edu
'''


#10 output classes nums 0-9
num_classes = 10

#28 x 28 greyscale images
input_shape = (28, 28, 1)

#model parameters
batch_size = 64
learning_rate = 0.0005

#num of epochs must be more than the num of models
epochs = 100

#number of models
n_models = 8




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




#learning rate schedule
def constant_lr_schedule():
    return learning_rate


#callback function for learning rate
class LearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
    
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        scheduled_lr = self.schedule()
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {}.".format(epoch+1, scheduled_lr))




#model generation from external file
model = Generate_Model_2(num_classes, input_shape)
print(model.summary())






weights = []
loss = []
new_weights = list()


#create callback for weight averaging
class weight_saver_callback(keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.loss_tracker = []

        self.epoch_tracker = 0


    #function called at the end of every batch
    def on_epoch_end(self, epoch, logs=None):

        print("\nSaving weights from epoch {} with loss {}".format(epoch, logs["val_loss"]))

        #save loss and weights for this batch
        self.loss_tracker.append(logs["val_loss"])
        loss.append(logs["val_loss"])
        weights.append(model.get_weights())

        self.epoch_tracker += 1
    






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
              callbacks=[weight_saver_callback(),
                        LearningRateScheduler(constant_lr_schedule)])


#finds the index of an array that has the minimum value
def findMinIndex(arr):
    minVal = math.inf
    minIndex = 0

    for i in range(len(arr)):
        if arr[i] < minVal:
            minVal = arr[i]
            minIndex = i
    
    return minIndex



models = list()


print("Creating Ensemble...")
for i in range(n_models):

    #find the index of the weights with the curent min loss
    min_model_index = findMinIndex(loss)

    #retrieve that weight and remove from lsit of weights
    curr_min_weight = weights.pop(min_model_index)

    #remove loss from loss list
    loss.pop(min_model_index)

    #create model with those weights
    model = Generate_Model_2(num_classes, input_shape)

    model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

    model.set_weights(curr_min_weight)

    #add model to list of models
    models.append(model)





def evaluate_models(model_list):
    check_freq = 100

    
    total_examples = len(x_test)
    num_correct = 0


    #for every training example
    for i in range(len(x_test)):

        #for testing only use forst 1000
        if i > 1000:
            break

        if i % check_freq == 0:
            print("Evaluating example: {}".format(i))


        #array to store the votes for this example
        votes = np.zeros(num_classes)



        #get vote of every model
        for j in range(len(models)):

            #get the raw prediction of model for this example
            prediction = model_list[j].predict(np.array( [x_test[i],] ), verbose=0)

            #find the indicie of the class predicted
            pred_class = np.argmax(prediction[0])
            
            #increment the vote counter
            votes[pred_class] += 1

        
        if np.argmax(y_test[i]) == np.argmax(votes):
            num_correct += 1

    return float(num_correct/total_examples)


ens_accuracy = evaluate_models(models)
print("\nAccuracy of ensemble is: {}".format(ens_accuracy))



#model evaluation
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('\nTest loss:', scores[0])
#print('Test accuracy:', scores[1])