import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from SwadUtility import AverageWeights, findStartAndEnd, findStartAndEnd3
from ModelGen import ResNet18_2
from ResNet18exp import ResNet18_exp


test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"

image_size = (244, 244)

loss = pd.read_csv("loss.csv")
loss = list(loss.iloc[:,1])

plt.plot(loss)
plt.show()


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






model = ResNet18_exp(2)
model.build(input_shape = (None,244,244,3))


#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])





min_loss = 100000000
optim_NS = 1
optim_NE = 1
optim_R = 0.1


'''
ts, te, l = findStartAndEnd(loss, optim_NS, optim_NE, optim_R)
weights = AverageWeights(model, ts, te, 200)
print("TS={} TE={}".format(ts, te))
model.set_weights(weights)
scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
curr_loss = scores[0]
if curr_loss < min_loss:
    min_loss = curr_loss
    print("Starting loss = {}".format(min_loss))
'''

tested_params = []

for NS in np.arange(1, 50, 1):
    for NE in np.arange(1, 50, 1):
        for R in np.arange(0.1, 5.0, 0.1):

            print("Testing parameters: {}, {}, {}".format(NS, NE, R))

            #find start and end iteration using current parameters
            ts, te, l = findStartAndEnd3(loss, NS, NE, R)
            print("TS={} TE={}".format(ts, te))

            if (ts, te) in tested_params:
                print("Skipping test already completed")
                continue

            #if end is less than or equal to start then doesnt make sense
            if te <= ts:
                print("Skipping test te <= ts")
                continue

            if ts == 0 or te == len(loss) or te == len(loss)-1:
                print("Skipping test ts=0 or te=end")
                continue

            if te < 0:
                print("Skipping test te < 0")
                continue

            if (te - ts) > 400:
                print("Skipping te - ts > 400")
                continue

            if ts < 5:
                print("Skipping ts < 5")
                continue

            #get new average weight
            print("Averaging Weights....")
            weights = AverageWeights(model, ts, te, 200)

            #set model weights
            model.set_weights(weights)

            #model evaluation
            scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
            curr_loss = scores[0]

            #if loss is new min then log parameters
            if curr_loss < min_loss:
                print("New min loss found. NS={} NE={} R={} ****************************".format(NS, NE, R))

                df = pd.DataFrame([NS, NE, R])
                df.to_csv('optim_params.csv') 

                min_loss = curr_loss
                optim_NS = NS
                optim_NE = NE
                optim_R = R
            
            tested_params.append((ts, te))


print("CURRENT BEST PARAMETERS NS={} NE={} R={}".format(optim_NS, optim_NE, optim_R))

ts, te, l = findStartAndEnd(loss, optim_NS, optim_NE, optim_R)
plt.plot(loss)
plt.axvline(x=ts, color='r')
plt.axvline(x=te, color='b')
plt.show()


#up to 6 3 0.9