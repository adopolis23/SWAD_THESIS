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

from ModelGen import Generate_Model_1, Generate_Model_2
from ModelGen import ResNet18_2
from WeightAverger import AverageWeights




'''
SWAD is a continuation of SWA where weights are averaged much more
frequently than in vanilla SWA. Weights are saved after every iteration. 

These saved weights are then averaged together then the weights in the 
model are updated with this new average. 

TODO:
Refactor the validation loss function. 


Brandon Weinhofer
U16425289
weinhofer@usf.edu
'''


#10 output classes nums 0-9
num_classes = 10

#28 x 28 greyscale images
input_shape = (28, 28)

#model parameters
batch_size = 32
learning_rate = 0.00005
epochs = 1

#SWAD parameters
NS = 3 #optimum patience
NE = 6 #overfit patience
r = 1.3 #tolerance ratio




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
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.01, random_state=0, stratify=y_train)

print("x_train.shape = ", x_train.shape)

#returns the validation loss of the model
def validate():
    return model.evaluate(x_valid, y_valid, verbose=0)


#learning rate scheudle - constant
def constant_lr_schedule():
    return learning_rate

    
#call back class for the learning rate schedule 
class LearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
    
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        scheduled_lr = self.schedule()
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {}.".format(epoch, scheduled_lr))





#build the model
#model definition in modelGen file
model = Generate_Model_2(num_classes, input_shape)

#model = ResNet18_2(10)
#model.build(input_shape = input_shape)
print(model.summary())






#algorithm to find start and end iteration for averaging from section B.4 in paper
def findStartAndEnd(val_loss):
    ts = 0
    te = len(val_loss)
    l = None

    for i in range(NS-1, len(val_loss)):
        
        min1 = math.inf
        for j in range(NE):
            if val_loss[i-j] < min1:
                min1 = val_loss[i-j]
        
        if l == None:
            
            min = math.inf
            for j in range(NS):
                if val_loss[i-j] < min:
                    min = val_loss[i-j]

            if val_loss[i-NS+1] == min:

                ts = i-NS+1
                sums = 0
                for j in range(NS):
                    sums = sums + val_loss[i-j]
                l = (r/NS)*sums
        
        elif l < min1:
            te = i-NE
            break
    return ts, te, l





weights = []
new_weights = list()


#create callback for weight averaging
class swad_callback(keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.loss_tracker = []
        self.iteration_tracker = 0


    #function called at the end of every batch
    def on_train_batch_end(self, batch, logs=None):

        #finds the validation loss after this batch
        #this is very slow and this is why this takes a while
        val_loss = validate()[0]

        #save loss and weights for this batch
        self.loss_tracker.append(val_loss)


        #weights.append(model.get_weights())
        model.save_weights("Weights/weights_" + str(self.iteration_tracker) + ".h5")

        self.iteration_tracker += 1




    #function called at the end of training
    #NOTE WEIGHT AVERAGING HAPPENS HERE
    def on_train_end(self, logs=None):
        print("\nEnd of Training")

        #optional plot the loss
        plt.plot(self.loss_tracker)
        plt.show()

        #finds the start and end iteration to average weights
        ts, te, l = findStartAndEnd(self.loss_tracker)
        print("ts is {} and te is {} and l is {}".format(ts, te, l))


        #optional save loss to csv
        #df = pd.DataFrame(self.loss_tracker)
        #df.to_csv('loss.csv') 

        print("\nAveraging Weights.")

        ts = int(input("TS:"))
        te = int(input("TE:"))

        new_weights = AverageWeights(model, ts, te, 200)



        '''
        #average up all saved weights and store them in new_weights
        #NOTE Weight averaging!
        for weights_list_tuple in zip(*pruned_weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        '''

        #set model weights to new average
        if len(new_weights) > 0:
            print("\nSetting new model weights.\n")
            model.set_weights(new_weights)
        





#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

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
              callbacks=[swad_callback(),
                        LearningRateScheduler(constant_lr_schedule)])



#model evaluation
scores = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])