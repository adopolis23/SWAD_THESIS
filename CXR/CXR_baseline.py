import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from imutils import paths


from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ModelGen import Generate_Model_2


image_size = (244, 244)

input_shape = (244, 244, 3)

learning_rate = 0.0009
epochs = 60
batch_size = 32

num_classes = 2



train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"



train_batches = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=train_path, target_size=image_size, classes=['covid', 'pneumonia'], batch_size=batch_size)
valid_batches = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=valid_path, target_size=image_size, classes=['covid', 'pneumonia'], batch_size=batch_size)
test_batches = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=test_path, target_size=image_size, classes=['covid', 'pneumonia'], batch_size=batch_size)
test_batches_unseen = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=test_path_unseen, target_size=image_size, classes=['covid', 'pneumonia'], batch_size=batch_size)


#build the model
model = Generate_Model_2(num_classes, input_shape)

print(model.summary())




class checkpoint(keras.callbacks.Callback):

    def __init__(self):
        self.min_loss = 1000000
        self.min_weight = None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.min_loss:
            print("\nValidation loss improved saving weights\n")
            self.min_loss = logs["val_loss"]
            self.min_weight = model.get_weights()

    def on_train_end(self, logs=None):
        print("\nSetting new model weights.\n")
        model.set_weights(self.min_weight)



#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



#train the model
model.fit(x=train_batches,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=valid_batches,
              shuffle=True,
              callbacks=checkpoint())



#model evaluation
scores = model.evaluate(test_batches, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

#model evaluation
scores_unseen = model.evaluate(test_batches_unseen, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])