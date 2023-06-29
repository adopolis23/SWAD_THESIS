import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from imutils import paths
import cv2
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from ModelGen import Generate_Model_2


IMAGE_SIZE = (244, 244)


train_datadir = "data/train"
val_datadir = "data/valid"
test_seen_datadir = "data/test-seen"
test_unseen_datadir = "data/test-unseen"
data_dirs = [train_datadir, val_datadir, test_seen_datadir, test_unseen_datadir]

catagories = ["covid", "pneumonia"]


training_data = []
val_data = []
test_seen_data = []
test_unseen_data = []
data = [training_data, val_data, test_seen_data, test_unseen_data]

def create_training_data():

    for i, data_dir in enumerate(data_dirs):

        for catagory in catagories:
            path = os.path.join(data_dir, catagory)
            class_num = catagories.index(catagory)

            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, IMAGE_SIZE)
                    data[i].append([new_array, class_num])
                except Exception as e:
                    print("Error loading img: {}".format(img))
                    pass




create_training_data()










