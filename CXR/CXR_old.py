import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from imutils import paths
import cv2
import glob
import shutil
import random

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from ModelGen import Generate_Model_2


image_size = (500, 500)

num_classes = 10

#model parameters
batch_size = 32
learning_rate = 0.0005
epochs = 100


covid_path = "data/covid"
pneumonia_path = "data/pneumonia"


covidPaths = sorted(list(paths.list_images(covid_path)))
pneumoniaPaths = sorted(list(paths.list_images(pneumonia_path)))



data = []
labels = []



for imagePath in covidPaths:
    image =  cv2.imread(imagePath)
    
    labels.append([1])
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image= image.astype('float32')
    image =  cv2.normalize(image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX)

    image = tf.keras.utils.img_to_array(image)
    data.append(image)

for imagePath in pneumoniaPaths:
    image =  cv2.imread(imagePath)
    
    labels.append(0)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image= image.astype('float32')
    image =  cv2.normalize(image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX)

    image = tf.keras.utils.img_to_array(image)
    data.append(image)

data = np.array(data)

print(data.shape)

#28 x 28 greyscale images
input_shape = data[0].shape


x_train = data[:400]
y_train = labels[:400]

x_val = data[400:450]
y_val = labels[400:450]

x_test = data[450:500]
y_test = labels[450:500]




#build the model
model = Generate_Model_2(num_classes, input_shape)

print(model.summary())





class checkpoint(keras.callbacks.Callback):

    def __init__(self):
        self.min_loss = 1000000
        self.min_weight = None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.min_loss:
            print("\nValidation loss improved saving weights\n")
            self.min_loss = logs["val_loss"]
            self.min_weight = model.get_weights()

    def on_train_end(self, logs=None):
        print("\nSetting new model weights.\n")
        model.set_weights(self.min_weight)



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
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=checkpoint())



#model evaluation
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])