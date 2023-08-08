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
from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working

from ModelGen import ResNet18_2
from ResNet18exp import ResNet18_exp

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



loss = pd.read_csv("loss.csv")
loss_vals = list(loss.iloc[:,1])

min_index = 0
min_val = 1000000000
for i, x in enumerate(loss_vals):
    if x < min_val:
        min_val = x
        min_index = i

print("Lowest loss at iteration: {}".format(min_index))
ts, te = findStartAndEnd2(loss_vals)
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

model = ResNet18_exp(2)
model.build(input_shape = (None,244,244,3))
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
scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

#model evaluation
scores_unseen = model.evaluate(test_unseen_x, test_unseen_y, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])