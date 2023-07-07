import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ModelGen import Generate_Model_2




train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"

image_size = (244, 244)
image_shape = (244, 244, 3)
learning_rate = 0.0009
epochs = 10
batch_size = 16
num_classes = 2




train_x = []
train_y = []





for file in os.listdir(train_path + "/covid"):
    
    image = cv2.imread(train_path + "/covid/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    train_x.append(image)
    train_y.append([1, 0])

for file in os.listdir(train_path + "/pneumonia"):
    
    image = cv2.imread(train_path + "/pneumonia/" + file)
    image=cv2.resize(image, image_size,interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = image.astype('float32')
    image /= 255 
    train_x.append(image)
    train_y.append([0, 1])

train_y = np.asarray(train_y).reshape(-1, 2)
train_x = np.asarray(train_x)

image_shape = train_x[0].shape
print("Input Shape: {}".format(image_shape))
print("Label Shape: {}".format(train_y[0].shape))









model = Generate_Model_2(num_classes, image_shape)
print(model.summary())


#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

#compile model with accuracy metric
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



#train the model
model.fit(x=np.array(train_x, np.float32),
            y=np.array(train_y, np.float32),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)

