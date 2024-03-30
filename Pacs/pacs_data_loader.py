import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random



def LoadPacsData(val_size_percent = 0.1, test_seen_size_percent = 0.1, train_size_percent = 1.0):

    data_folder = "data/"

    domains = ["art_painting/", "cartoon/", "photo/", "sketch/"]
    leave_out_domain = ["photo/"]

    classes = ["dog/", "elephant/", "giraffe/", "guitar/", "horse/", "house/", "person/"]

    train_x = []
    train_y = []

    val_x = []
    val_y = []

    test_seen_x = []
    test_seen_y = []

    test_unseen_x = []
    test_unseen_y = []


    #load in images
    for domain in domains:
        print("Loading domain: {}".format(domain))

        for i, _class in enumerate(classes):

            #load the unseed domain into test_unseen
            if domain in leave_out_domain:
                files = os.listdir(data_folder + domain + _class)
                
                for file in files:

                    image = cv2.imread(data_folder + domain + _class + file)
                    #image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
                    image=np.array(image)
                    image = image.astype('float32')
                    image /= 255 
                    test_unseen_x.append(image)

                    one_hot_y = [0, 0, 0, 0, 0, 0, 0]
                    one_hot_y[i] += 1

                    test_unseen_y.append(one_hot_y)

                continue


            #load the seen data inso training
            files = os.listdir(data_folder + domain + _class)

            #cut down training data based on train_size_percent; removed that percentage of files for each class
            files = files[:int(len(files)*train_size_percent)]

            for file in files:

                image = cv2.imread(data_folder + domain + _class + file)
                #image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
                image=np.array(image)
                image = image.astype('float32')
                image /= 255 
                train_x.append(image)

                one_hot_y = [0, 0, 0, 0, 0, 0, 0]
                one_hot_y[i] += 1

                train_y.append(one_hot_y)



    #randomly take examples from training and seperate them into val and test_seen
    val_size = int(val_size_percent * len(train_x))
    test_seen_size = int(test_seen_size_percent * len(train_x))


    for i in range(val_size):
        x = random.randrange(0, len(train_x), 1)

        val_x.append(train_x.pop(x))
        val_y.append(train_y.pop(x))

    for i in range(test_seen_size):
        x = random.randrange(0, len(train_x), 1)

        test_seen_x.append(train_x.pop(x))
        test_seen_y.append(train_y.pop(x))

    train_y = np.asarray(train_y).reshape(-1, 7)
    train_x = np.asarray(train_x)

    val_y = np.asarray(val_y).reshape(-1, 7)
    val_x = np.asarray(val_x)

    test_seen_y = np.asarray(test_seen_y).reshape(-1, 7)
    test_seen_x = np.asarray(test_seen_x)

    test_unseen_y = np.asarray(test_unseen_y).reshape(-1, 7)
    test_unseen_x = np.asarray(test_unseen_x)

    return train_x, train_y, val_x, val_y, test_seen_x, test_seen_y, test_unseen_x, test_unseen_y

      



