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
from SwadUtility import AverageWeights, findStartAndEnd2, findStartAndEnd3, findStartAndEnd
import matplotlib.pyplot as plt
#from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working



test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"



num_classes = 2
image_size = (244, 244)
image_shape = (244, 244, 3)

learning_rate = 0.0004

test_seen_x = []
test_seen_y = []

test_unseen_x = []
test_unseen_y = []





'''
# LOAD TEST-SEEN DATA
for file in os.listdir(test_path + "/covid"):
    
    image = cv2.imread(test_path + "/covid/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    #image /= 255 
    test_seen_x.append(image)
    test_seen_y.append([1, 0])

for file in os.listdir(test_path + "/pneumonia"):
    
    image = cv2.imread(test_path + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    #image /= 255 
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
    #image /= 255 
    test_unseen_x.append(image)
    test_unseen_y.append([1, 0])

for file in os.listdir(test_path_unseen + "/pneumonia"):
    
    image = cv2.imread(test_path_unseen + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    #image /= 255 
    test_unseen_x.append(image)
    test_unseen_y.append([0, 1])

test_unseen_y = np.asarray(test_unseen_y).reshape(-1, 2)
test_unseen_x = np.asarray(test_unseen_x)
# ----------------
'''


loss = pd.read_csv("loss.csv")
loss_vals = list(loss.iloc[:,1])

#accuracy = pd.read_csv("accuracy.csv")
#accuracy_vals = list(accuracy.iloc[:,1])

min_index = 0
min_val = 1000000000
for i, x in enumerate(loss_vals):
    if x < min_val:
        min_val = x
        min_index = i

print("Lowest loss at iteration: {}".format(min_index))


tsp, tep, l = findStartAndEnd2(loss_vals)
tso, teo, l = findStartAndEnd(loss_vals, 6, 6, 1.2)



fig, (ax1, ax2) = plt.subplots(1, 2)

'''
ax1.plot(self.accuracy_tracker, color='black')
ax1.axvline(x=tso, color='r')
ax1.axvline(x=teo, color='b')
ax1.set(xlabel="Iteration", ylabel="Validation Accuracy")

ax2.plot(self.accuracy_tracker, color='black')
ax2.axvline(x=tsp, color='r')
ax2.axvline(x=tep, color='b')
ax2.set(xlabel="Iteration", ylabel="Validation Accuracy")
'''

tep = 356

ax1.plot(loss_vals, color='black')
ax1.axvline(x=tso, color='r')
ax1.axvline(x=teo, color='b')
ax1.set(xlabel="Iteration", ylabel="Validation Loss")

ax2.plot(loss_vals, color='black')
ax2.axvline(x=tsp, color='r')
ax2.axvline(x=tep, color='b')
ax2.set(xlabel="Iteration", ylabel="Validation Loss")

for i in range(50, 1100, 50):
    ax1.axvline(x=i,linewidth=0.5, color='gray')
    ax2.axvline(x=i,linewidth=0.5, color='gray')

plt.show()

