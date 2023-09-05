from keras.datasets import cifar10
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import random
import gc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from ModelGen import Generate_Model_2
from ModelGen import ResNet18_2
from ResNet18exp import ResNet18_exp
from SwadUtility import findStartAndEnd2, AverageWeights

from ConvMixer import ConvMixer



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

#32 x 32 3-channel color images
input_shape = (32, 32, 3)

#model parameters
batch_size = 64
learning_rate = 0.00005
epochs = 90

runs = 1
train_size = 50000

#SWAD parameters
NS = 0 #optimum patience
NE = 0 #overfit patience
r = 1.2 #tolerance ratio




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

setSeed(seeds[1])

files = os.listdir("Weights")
for file in files:
    os.remove("Weights/"+file)


#download data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train[:train_size]
y_train = y_train[:train_size]

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)


#normalize images to 0 - 1 range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


#create the velidation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.01, random_state=0, stratify=y_train)


#returns the validation loss of the model
def validate():
    return model.evaluate(x_valid, y_valid, verbose=1)


def validate2():
    y_pred = model.predict(x_valid, verbose=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    val_loss = bce(y_valid, y_pred).numpy()
    return val_loss

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
        #print("\nEpoch {}: Learning rate is {}.".format(epoch, scheduled_lr))









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
        ts, te, l = findStartAndEnd2(self.loss_tracker, NS, NE, r)
        print("ts is {} and te is {}".format(ts, te))

        #optional plot the loss
        #plt.plot(self.loss_tracker)
        #plt.axvline(x=ts, color='r')
        #plt.axvline(x=te, color='b')
        #plt.show()



        #optional save loss to csv
        #df = pd.DataFrame(self.loss_tracker)
        #df.to_csv('loss.csv') 

        #print("\nAveraging Weights.")

        #ts = int(input("TS:"))
        #te = int(input("TE:"))

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










class checkpoint(tf.keras.callbacks.Callback):

    def __init__(self):
        self.min_loss = 1000000
        self.min_weight = None
        self.loss_tracker = []

    def on_epoch_end(self, epoch, logs=None):
        val_loss = validate2()
        self.loss_tracker.append(val_loss)

        if val_loss < self.min_loss:
            print("\nValidation loss improved saving weights\n")
            self.min_loss = val_loss
            self.min_weight = model.get_weights()

    def on_train_end(self, logs=None):
        plt.plot(self.loss_tracker)
        plt.show()
        print("\nSetting new model weights.\n")
        model.set_weights(self.min_weight)


results = []

for i in range(runs):

    print("******* Run Number: {} *******".format(i))

    weights_folder = os.listdir("Weights")
    for file in weights_folder:
        os.remove("Weights/"+file)

    setSeed(seeds[i])

    model = None
    opt = None

    gc.collect()

    
    #build the model

    #model = ConvMixer(num_classes)

    #model = Generate_Model_2(num_classes, image_shape)
    #model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)

    model = ResNet18_exp(10)
    model.build(input_shape = (None,32,32,3))
    #print(model.summary())


    #Adam optimizer with learning rate and 0.9 momentum
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

    #compile model with accuracy metric
    model.compile(loss="categorical_crossentropy",
                optimizer=opt,
                metrics=['accuracy'])



    #train the model
    model.fit(x=np.array(x_train, np.float32),
                y=np.array(y_train, np.float32),
                validation_data=(x_valid, y_valid),
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
                callbacks=[swad_callback()])

    #model evaluation
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss seen:', scores[0])
    print('Test accuracy seen:', scores[1])

    print("******* End Run Number: {} *******".format(i))

    results.append(scores[1])


df = pd.DataFrame(results)
df.to_csv('multirun_results.csv')

print("\n\n Final Results:\n")

for i, x in enumerate(results):
    print("\nRun: {}, Loss-Seen: {}".format(i, x[0]))
    print("\nRun: {}, Loss-unSeen: {}".format(i, x[1]))



