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
from WeightAverger import AverageWeights
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working


train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"

image_size = (244, 244)
image_shape = (244, 244, 3)
learning_rate = 0.0002

epochs = 10
batch_size = 16
num_classes = 2


#swad parameters
NS = 3
NE = 3
r = 1.2



train_x = []
train_y = []

val_x = []
val_y = []

test_seen_x = []
test_seen_y = []

test_unseen_x = []
test_unseen_y = []




# LOAD TRAIN DATA
for file in os.listdir(train_path + "/covid"):
    
    image = cv2.imread(train_path + "/covid/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    train_x.append(image)
    train_y.append([1, 0])

for file in os.listdir(train_path + "/pneumonia"):
    
    image = cv2.imread(train_path + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    train_x.append(image)
    train_y.append([0, 1])

train_y = np.asarray(train_y).reshape(-1, 2)
train_x = np.asarray(train_x)
# ----------------

# LOAD VAL DATA
for file in os.listdir(valid_path + "/covid"):
    
    image = cv2.imread(valid_path + "/covid/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    val_x.append(image)
    val_y.append([1, 0])

for file in os.listdir(valid_path + "/pneumonia"):
    
    image = cv2.imread(valid_path + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    val_x.append(image)
    val_y.append([0, 1])

val_y = np.asarray(val_y).reshape(-1, 2)
val_x = np.asarray(val_x)
# ----------------


# LOAD TEST-SEEN DATA
for file in os.listdir(test_path + "/covid"):
    
    image = cv2.imread(test_path + "/covid/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    test_seen_x.append(image)
    test_seen_y.append([1, 0])

for file in os.listdir(test_path + "/pneumonia"):
    
    image = cv2.imread(test_path + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    test_seen_x.append(image)
    test_seen_y.append([0, 1])

test_seen_y = np.asarray(test_seen_y).reshape(-1, 2)
test_seen_x = np.asarray(test_seen_x)
# ----------------

# LOAD TEST-UNSEEN DATA
for file in os.listdir(test_path_unseen + "/covid"):
    
    image = cv2.imread(test_path_unseen + "/covid/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    test_unseen_x.append(image)
    test_unseen_y.append([1, 0])

for file in os.listdir(test_path_unseen + "/pneumonia"):
    
    image = cv2.imread(test_path_unseen + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    test_unseen_x.append(image)
    test_unseen_y.append([0, 1])

test_unseen_y = np.asarray(test_unseen_y).reshape(-1, 2)
test_unseen_x = np.asarray(test_unseen_x)
# ----------------

image_shape = train_x[0].shape
print("Input Shape: {}".format(image_shape))
print("Label Shape: {}".format(train_y[0].shape))



model = Generate_Model_2(num_classes, image_shape)
#model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)
print(model.summary())


def sumPrev(array, n):
    new = array[-n:]
    sum_ = 0
    for x in new:
        sum_ = sum_ + x
    
    return sum_


def minValue(array):
    min_index = 0
    min_val = 1000000000
    for i, x in enumerate(array):
        if x < min_val:
            min_val = x
            min_index = i
    
    return min_val


def addWeights(w1, w2):
    tmp = [w1, w2]
    new = list()
    for wt in zip(*tmp):
        new.append(
            np.array([np.array(w).sum(axis=0) for w in zip(*wt)])
        )
    return new


def validate2():
    y_pred = model.predict(val_x, verbose=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    val_loss = bce(val_y, y_pred).numpy()
    return val_loss



class swad(tf.keras.callbacks.Callback):

    def __init__(self):
        self.weight = list()
        self.loss_tracker = []
        self.iteration_tracker = 0
        self.weights_saved = 0

        self.ts = 0
        self.te = epochs * 33
        self.l = None

    def on_train_batch_end(self, batch, logs=None):
        val_loss = validate2()
        self.loss_tracker.append(val_loss)


        '''
        #if its first iter then just get weights else add weights to accumulator
        if self.iteration_tracker == 0:
            self.weight = model.get_weights()
        else:
            self.weight = addWeights(self.weight, model.get_weights())
        self.weights_saved += 1
        '''

        if self.iteration_tracker > NS+1:
            if self.l == None:

                if self.loss_tracker[self.iteration_tracker-NS+1] == minValue(self.loss_tracker[-NS:]):
                    
                    self.ts = self.iteration_tracker - NS + 1
                    print("\n\nStarting Weight Save TS={}".format(self.ts))
                    print("\nCurrent iter is: {}".format(self.iteration_tracker))

                    self.l = float(r/NS) * sumPrev(self.loss_tracker, (NS-1))
                    print("\n\nL is: {}".format(self.l))



            elif self.l < minValue(self.loss_tracker[-NE:]):
                self.te = self.iteration_tracker - NE
                self.model.stop_training = True
                #STOP TRAINING 


        if self.ts != 0 and self.iteration_tracker <= self.te:
            if self.weights_saved == 0:
                self.weight = model.get_weights()
                self.weights_saved += 1
            else:
                self.weight = addWeights(self.weight, model.get_weights())
                self.weights_saved += 1


        self.iteration_tracker += 1

        



    def on_train_end(self, logs=None):
        print("\nSetting new model weights.\n")

        new_w = np.divide(self.weight, self.weights_saved)

        model.set_weights(new_w)







#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



#weights = model.get_weights()
        
#weights_n = np.divide(weights, 2)



#train the model
model.fit(x=np.array(train_x, np.float32),
            y=np.array(train_y, np.float32),
            validation_data=(val_x, val_y),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=[swad()])


#model evaluation
scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

#model evaluation
scores_unseen = model.evaluate(test_unseen_x, test_unseen_y, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])

