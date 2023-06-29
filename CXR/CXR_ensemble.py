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
from WeightAverger import AverageWeights


image_size = (244, 244)

input_shape = (244, 244, 3)

learning_rate = 0.0001
epochs = 10
batch_size = 32

num_classes = 2

n_models = 5

NS = 3
NE = 3
r = 1.2


train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"

num_test_seen = len(os.listdir(test_path+"/covid"))*2
num_test_unseen = len(os.listdir(test_path_unseen+"/covid"))*2


train_batches = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=train_path, target_size=image_size, classes=['covid', 'pneumonia'], batch_size=batch_size)
valid_batches = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=valid_path, target_size=image_size,classes=['covid', 'pneumonia'], batch_size=batch_size)
test_batches = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=test_path, target_size=image_size, classes=['covid', 'pneumonia'], batch_size=num_test_seen)
test_batches_unseen = ImageDataGenerator(preprocessing_function=None).flow_from_directory(directory=test_path_unseen, target_size=image_size, classes=['covid', 'pneumonia'], batch_size=num_test_unseen)


print(tf.config.list_physical_devices())


#build the model
model = Generate_Model_2(num_classes, input_shape)

print(model.summary())





#returns the validation loss of the model
def validate():
    return model.evaluate(valid_batches, verbose=0)





def findStartAndEnd(val_loss):
    ts = 0
    te = len(val_loss)
    l = None

    for i in range(NS-1, len(val_loss)):
        
        min1 = math.inf
        for j in range(NE):
            if val_loss[i-j] < min1:
                min1 = val_loss[i-j]
        
        if l == None:
            
            min = math.inf
            for j in range(NS):
                if val_loss[i-j] < min:
                    min = val_loss[i-j]

            if val_loss[i-NS+1] == min:

                ts = i-NS+1
                sums = 0
                for j in range(NS):
                    sums = sums + val_loss[i-j]
                l = (r/NS)*sums
        
        elif l < min1:
            te = i-NE
            break
    return ts, te, l



weights = []
loss = []
new_weights = list()


#create callback for weight averaging
class weight_saver_callback(keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.loss_tracker = []

        self.epoch_tracker = 0


    #function called at the end of every batch
    def on_epoch_end(self, epoch, logs=None):
        
        #save loss and weights for this batch
        self.loss_tracker.append(logs["val_loss"])
        loss.append(logs["val_loss"])

        if epoch >= 5:
            print("\nSaving weights from epoch {} with loss {}".format(epoch, logs["val_loss"]))
            weights.append(model.get_weights())

        self.epoch_tracker += 1
        




#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

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
              callbacks=weight_saver_callback())



#finds the index of an array that has the minimum value
def findMinIndex(arr):
    minVal = math.inf
    minIndex = 0

    for i in range(len(arr)):
        if arr[i] < minVal:
            minVal = arr[i]
            minIndex = i
    
    return minIndex



models = list()


print("Creating Ensemble...")
for i in range(n_models):


    #retrieve that weight and remove from lsit of weights
    curr_weight = weights[i]

    #remove loss from loss list

    #create model with those weights
    model = Generate_Model_2(num_classes, input_shape)

    model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

    model.set_weights(curr_weight)

    #add model to list of models
    models.append(model)





def evaluate_models(model_list, x_test, y_test):

    total_examples = len(y_test)
    total_correct = 0

    model_outputs = []

    for model in model_list:
        output = model.predict(x_test, verbose=1)
        model_outputs.append(output)
    
    for i in range(total_examples):
        votes = np.zeros(num_classes)

        for output_frame in model_outputs:
            vote = np.argmax(output_frame[i])
            votes[vote] += 1
        
        if np.argmax(y_test[i]) == np.argmax(votes):
            total_correct += 1
    
    return float(total_correct/total_examples)


x_test_seen, y_test_seen = test_batches.next()
x_test_unseen, y_test_unseen = test_batches_unseen.next()

#accuracy = evaluate_models(models, x_test_seen, y_test_seen)


print("\nEnsemble results")

#model evaluation
scores = evaluate_models(models, x_test_seen, y_test_seen)
#print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores)

#model evaluation
scores_unseen =evaluate_models(models, x_test_unseen, y_test_unseen)
print('Test accuracy unseen:', scores_unseen)
