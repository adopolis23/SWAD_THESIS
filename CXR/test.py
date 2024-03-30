import tensorflow as tf
import os
import cv2
import gc
import pandas as pd
import math
import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ModelGen import Generate_Model_2, LeNet
from SwadUtility import AverageWeights, findStartAndEnd, findStartAndEnd2, findStartAndEnd3
import matplotlib.pyplot as plt
#from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working
from tensorflow.keras.applications.resnet50 import ResNet50

from ModelGen import ResNet18_2
from ResNet18exp import ResNet18_exp

from modified_densenet import DenseNet121


train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"

image_size = (244, 244)
image_shape = (244, 244, 3)

#default = 0.00005
learning_rate = 0.0001

epochs = 40
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

setSeed(seeds[3])

files = os.listdir("Weights")
for file in files:
    os.remove("Weights/"+file)

# LOAD TRAIN DATA
for file in os.listdir(train_path + "/covid"):
    
    image = cv2.imread(train_path + "/covid/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    #image /= 255 
    train_x.append(image)
    train_y.append([1, 0])

for file in os.listdir(train_path + "/pneumonia"):
    
    image = cv2.imread(train_path + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    #image /= 255 
    train_x.append(image)
    train_y.append([0, 1])




print(train_x[50])