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
from SwadUtility import AverageWeights, findStartAndEnd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working
from tensorflow.keras.applications.resnet50 import ResNet50

from ModelGen import ResNet18_2



train_path = "data/train2"
valid_path = "data/valid2"
test_path = "data/test-seen2"

image_size = (244, 244)
image_shape = (244, 244, 3)
learning_rate = 0.00005

epochs = 100
batch_size = 16
num_classes = 2




train_x = []
train_y = []

val_x = []
val_y = []

test_seen_x = []
test_seen_y = []


weight_save_epoch = [50, 100, 150, 200]






# LOAD TRAIN DATA
for file in os.listdir(train_path + "/normal"):
    
    image = cv2.imread(train_path + "/normal/" + file)
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

train_y = np.asarray(train_y).reshape(-1, 2)
train_x = np.asarray(train_x)
# ----------------

# LOAD VAL DATA
for file in os.listdir(valid_path + "/normal"):
    
    image = cv2.imread(valid_path + "/normal/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    #image /= 255 
    val_x.append(image)
    val_y.append([1, 0])

for file in os.listdir(valid_path + "/pneumonia"):
    
    image = cv2.imread(valid_path + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    #image /= 255 
    val_x.append(image)
    val_y.append([0, 1])

val_y = np.asarray(val_y).reshape(-1, 2)
val_x = np.asarray(val_x)
# ----------------


# LOAD TEST-SEEN DATA
for file in os.listdir(test_path + "/normal"):
    
    image = cv2.imread(test_path + "/normal/" + file)
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



image_shape = train_x[0].shape
print("Input Shape: {}".format(image_shape))
print("Label Shape: {}".format(train_y[0].shape))



#model = Generate_Model_2(num_classes, image_shape)
#model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)
#model = ResNet18(input_shape=image_shape, classes=num_classes)

model = ResNet18_2(2)
model.build(input_shape = (None,244,244,3))

print(model.summary())



#returns the validation loss of the model
def validate():
    return model.evaluate(val_x, val_y, verbose=0)[0]


def validate2():
    y_pred = model.predict(val_x, verbose=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    val_loss = bce(val_y, y_pred).numpy()
    return val_loss





class checkpoint(tf.keras.callbacks.Callback):

    def __init__(self):
        self.min_loss = 1000000
        self.min_weight = None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.min_loss:
            print("\nValidation loss improved saving weights\n")
            self.min_loss = logs["val_loss"]
            self.min_weight = model.get_weights()

            model.save_weights("PretrainedWeights/ResNet18/ResNet18WeightsEpoch" + str(epoch) + ".h5")
        
        if epoch in weight_save_epoch:
            model.save_weights("PretrainedWeights/ResNet18/ResNet18WeightsEpoch" + str(epoch) + ".h5")

    def on_train_end(self, logs=None):
        print("\nSetting new model weights.\n")
        model.set_weights(self.min_weight)







#normal Keras stuff


#Adam optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



#train the model
model.fit(x=np.array(train_x, np.float32),
            y=np.array(train_y, np.float32),
            validation_data=(val_x, val_y),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=checkpoint())

#model evaluation
scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])
