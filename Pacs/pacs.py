import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import random
import time
import gc

from pacs_data_loader import LoadPacsData
from ModelGen import Generate_Model_2
from tensorflow.keras.applications.resnet50 import ResNet50

learning_rate = 0.0001
epochs = 35
batch_size = 4
num_classes = 7


image_size = (227, 227)
image_shape = (227, 227, 3)


train_x, train_y, val_x, val_y, test_seen_x, test_seen_y, test_unseen_x, test_unseen_y = LoadPacsData()

print("Train length = {}".format(len(train_x)))
print("Val length = {}".format(len(val_x)))
print("Test_seen length = {}".format(len(test_seen_x)))
print("Test_unseen length = {}".format(len(test_unseen_x)))



model = ResNet50(input_shape=image_shape, classes=num_classes, weights=None)
#model = Generate_Model_2(num_classes, image_shape)
print(model.summary())





def validate():
  y_pred = model.predict(val_x, verbose=0)
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  val_loss = bce(val_y, y_pred).numpy()
  return val_loss




class checkpoint(tf.keras.callbacks.Callback):

  def __init__(self):
    self.min_loss = 1000000
    self.min_weight = None

  def on_train_batch_end(self, batch, logs=None):
    val_loss = validate()
 
  
    if val_loss < self.min_loss:
        print("\nValidation loss improved saving weights\n")
        self.min_loss = val_loss
        self.min_weight = model.get_weights()

  def on_train_end(self, logs=None):
    print("\nSetting new model weights.\n")
    model.set_weights(self.min_weight)





#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



start_time = time.time()

#train the model
model.fit(x=np.array(train_x, np.float32),
            y=np.array(train_y, np.float32),
            validation_data=(val_x, val_y),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=[checkpoint()])

elapsed_time = time.time() - start_time

print("Total training time: {}".format(elapsed_time))

gc.collect()

#model evaluation
scores = model.evaluate(test_seen_x, test_seen_y, verbose=1, batch_size=batch_size)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

gc.collect()

#model evaluation
scores_unseen = model.evaluate(test_unseen_x, test_unseen_y, verbose=1, batch_size=batch_size)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])