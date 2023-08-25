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
from SwadUtility import AverageWeights, findStartAndEnd2
import matplotlib.pyplot as plt
#from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working

from ModelGen import ResNet18_2
from ResNet18exp import ResNet18_exp
from modified_densenet import DenseNet121



num_classes = 10
image_size = (244, 244)
image_shape = (244, 244, 3)

learning_rate = 0.0004

test_seen_x = []
test_seen_y = []

test_unseen_x = []
test_unseen_y = []



from keras.datasets import mnist
from sklearn.model_selection import train_test_split


train_size = 700
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









loss = pd.read_csv("loss.csv")
loss_vals = list(loss.iloc[:,1])

min_index = 0
min_val = 1000000000
for i, x in enumerate(loss_vals):
    if x < min_val:
        min_val = x
        min_index = i

print("Lowest loss at iteration: {}".format(min_index))
ts, te, l = findStartAndEnd2(loss_vals)
print("ts: {}, te: {}".format(ts, te))



plt.plot(loss_vals)
plt.axvline(x=ts, color='r')
plt.axvline(x=te, color='b')
plt.show()



#model = Generate_Model_2(num_classes, image_shape)
#model = EfficientNetB1(input_shape=image_shape, classes=num_classes, weights=None)
#model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)
#model = ResNet18_2(2)
#model.build(input_shape = (None,244,244,3))

model = ResNet18_exp(10)
model.build(input_shape = (None,28,28,1))
#print(model.summary())


opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



ts = int(input("TS:"))
te = int(input("TE:"))

if ts == te:
    model.load_weights('Weights/weights_' + str(ts) + '.h5')
else:
    new_weights = AverageWeights(model, ts, te, 200)

    print("\nSetting new model weights.\n")
    model.set_weights(new_weights)


#model evaluation
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

