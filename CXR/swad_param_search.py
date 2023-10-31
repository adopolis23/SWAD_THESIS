import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from SwadUtility import AverageWeights, findStartAndEnd
from ModelGen import ResNet18_2


test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"

image_size = (244, 244)

loss = pd.read_csv("loss.csv")
loss = list(loss.iloc[:,1])

#plt.plot(loss)
#plt.show()


test_seen_x = []
test_seen_y = []

test_unseen_x = []
test_unseen_y = []






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


'''
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






model = ResNet18_2(2)
model.build(input_shape = (None,244,244,3))


#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])





min_loss = 100000000
optim_NS = None
optim_NE = None
optim_R = None

for NS in np.arange(1, 4, 1):
    for NE in np.arange(1, 4, 1):
        for R in np.arange(0.1, 1.6, 0.1):
            print("Testing parameters: {}, {}, {}".format(NS, NE, R))

            #find start and end iteration using current parameters
            ts, te, l = findStartAndEnd(loss, NS, NE, R)

            #if end is less than or equal to start then doesnt make sense
            if te <= ts:
                print("Skipping test te <= ts")
                continue

            #get new average weight
            weights = AverageWeights(model, ts, te, 200)

            #set model weights
            model.set_weights(weights)

            #model evaluation
            scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
            curr_loss = scores[0]

            #if loss is new min then log parameters
            if curr_loss < min_loss:
                print("New min loss found. NS={} NE={} R={}".format(NS, NE, R))
                min_loss = curr_loss
                optim_NS = NS
                optim_NE = NE
                optim_R = R


ts, te, l = findStartAndEnd(loss, optim_NS, optim_NE, optim_R)
plt.plot(loss)
plt.axvline(x=ts, color='r')
plt.axvline(x=te, color='b')
plt.show()
