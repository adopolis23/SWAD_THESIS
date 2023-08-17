import tensorflow as tf
import os
import cv2
import gc
import pandas as pd
import math
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ModelGen import Generate_Model_2, LeNet
from SwadUtility import AverageWeights, findStartAndEnd, findStartAndEnd2
import matplotlib.pyplot as plt
#from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working
from tensorflow.keras.applications.resnet50 import ResNet50

from ModelGen import ResNet18_2
from ResNet18exp import ResNet18_exp

from modified_densenet import DenseNet121

from keras.datasets import mnist
from sklearn.model_selection import train_test_split





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
input_shape = (None, 28, 28, 1)

#model parameters
batch_size = 16
learning_rate = 0.00005
epochs = 1

#SWAD parameters
NS = 3 #optimum patience
NE = 3 #overfit patience
r = 1.3 #tolerance ratio

train_size = 1000



seeds = [63528,30270,1186,47466,13938,27248,23050,32591,70485,44794,87752,67208,48357,41003,44268,55533,54862,59718,78523,69827,33651,12194,56602]


def setSeed(seed):
    newSeed = int(seed)

    from numpy.random import seed
    import random as ran
    
    #get_ipython().run_line_magic('env', 'PYTHONHASHSEED=1')
    ran.seed(newSeed)
    seed(newSeed)
    tf.random.set_seed(newSeed)

    session_conf = tf.compat.v1.ConfigProto()

    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'

    #from tensorflow.keras import backend as K
    #K.set_image_data_format('channels_first')

setSeed(seeds[4])

files = os.listdir("Weights")
for file in files:
    os.remove("Weights/"+file)




#download data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train[:train_size]
y_train = y_train[:train_size]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)


#normalize images to 0 - 1 range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


#create the velidation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)

print("x_train.shape = ", x_train.shape)

#returns the validation loss of the model
def validate():
    return model.evaluate(x_valid, y_valid, verbose=0)


def validate2():
    y_pred = model.predict(x_valid, verbose=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    val_loss = bce(y_valid, y_pred).numpy()
    return val_loss


#learning rate scheudle - constant
def constant_lr_schedule():
    return learning_rate

    


class checkpoint(tf.keras.callbacks.Callback):

    def __init__(self):
        self.min_loss = 1000000
        self.min_weight = None

    def on_train_batch_end(self, epoch, logs=None):
        val_loss = validate2()

        if val_loss < self.min_loss:
            print("\nValidation loss improved saving weights\n")
            self.min_loss = val_loss
            self.min_weight = model.get_weights()

    def on_train_end(self, logs=None):
        print("\nSetting new model weights.\n")
        model.set_weights(self.min_weight)



weights = []
new_weights = list()


#create callback for weight averaging
class swad_callback(tf.keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.loss_tracker = []
        self.iteration_tracker = 0
        self.weights_saved = 0

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


    #function called at the end of every batch
    def on_train_batch_end(self, batch, logs=None):

        #finds the validation loss after this batch
        #this is very slow and this is why this takes a while

        if self.iteration_tracker >= 0:
            val_loss = validate2()


            #save loss and weights for this batch
            self.loss_tracker.append(val_loss)


            #weights.append(model.get_weights())
            model.save_weights("Weights/weights_" + str(self.weights_saved) + ".h5")
            self.weights_saved += 1

        self.iteration_tracker += 1


    #function called at the end of training
    #NOTE WEIGHT AVERAGING HAPPENS HERE
    def on_train_end(self, logs=None):
        print("\nEnd of Training")

        #finds the start and end iteration to average weights
        ts, te = findStartAndEnd2(self.loss_tracker, NS, NE, r)
        print("ts is {} and te is {}".format(ts, te))

        #optional plot the loss
        plt.plot(self.loss_tracker)
        plt.axvline(x=ts, color='r')
        plt.axvline(x=te, color='b')
        plt.show()



        #optional save loss to csv
        df = pd.DataFrame(self.loss_tracker)
        df.to_csv('loss.csv') 

        print("\nAveraging Weights.")

        ts = int(input("TS:"))
        te = int(input("TE:"))

        new_weights = AverageWeights(model, ts, te, 200)



        '''
        #average up all saved weights and store them in new_weights
        #NOTE Weight averaging!
        for weights_list_tuple in zip(*saved_weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )8
        '''

        #set model weights to new average
        if len(new_weights) > 0:
            print("\nSetting new model weights.\n")
            model.set_weights(new_weights)





#build the model
#model definition in modelGen file
#model = Generate_Model_2(num_classes, input_shape)

model = ResNet18_exp(10)
model.build(input_shape=input_shape)
print(model.summary())






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
              callbacks=[checkpoint()])



#model evaluation
scores = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss:', scores[0])
print('Test accuracy:', scores[1])