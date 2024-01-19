import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from ModelGen import Generate_Model_2, ResNet18_2
from SwadUtility import findStartAndEnd2
from numpy.random import seed
from ModelGen import Generate_Model_2
import cv2
import numpy as np
import random as ran
import pandas as pd
import gc
import os

batch_size = 128
num_classes = 10
epochs = 10
runs = 10


NS = 3
NE = 6
r = 1.2

swad_start_iter = 1
rolling_window_size = 75
results = []


#download data
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


#normalize images to 0 - 1 range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


#x_valid = x_train[-5000:]
#y_valid = y_train[-5000:]

#x_train = x_train[:-5000]
#y_train = y_train[:-5000]

#x_test_ood = x_test
#y_test_ood = y_test

#create the velidation data
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)
#print("x_valid.shape = ", x_valid.shape)
#print("y_valid.shape = ", y_valid.shape)


for i in range(1):
    print("******* Run Number: {} *******".format(i))
    #setSeed(seeds[i])
    #model = None
    #gc.collect()
    
    weights_folder = os.listdir("Weights")
    for file in weights_folder:
        os.remove("Weights/"+file)
    
    model = Generate_Model_2(10, (28, 28, 1))

    #model = ResNet18_2(10)
    #model.build(input_shape = (None, 28, 28, 1))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 

    #compile model with accuracy metric
    model.compile(loss="categorical_crossentropy",
                optimizer=opt,
                metrics=['accuracy'])

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        #validation_data=(x_valid, y_valid),
        epochs=epochs,
        shuffle=True,
        verbose=2
    )
    
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss seen:', scores[0])
    print('Test accuracy seen:', scores[1])

    scores_unseen = model.evaluate(x_test_ood, y_test_ood, verbose=1)
    print('Test loss unseen:', scores_unseen[0])
    print('Test accuracy unseen:', scores_unseen[1])
    
    results.append([scores[1], scores_unseen[1]])
    