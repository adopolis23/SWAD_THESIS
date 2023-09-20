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
from SwadUtility import AverageWeights, findStartAndEnd, findStartAndEnd2
import matplotlib.pyplot as plt
#from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working
from tensorflow.keras.applications.resnet50 import ResNet50

from ModelGen import ResNet18_2
from ResNet18exp import ResNet18_exp

from modified_densenet import DenseNet121

train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"


image_size = (244, 244)

input_shape = (244, 244, 3)

learning_rate = 0.00005
epochs = 63
batch_size = 16

num_classes = 2

n_models = 6
#save_at = [40, 43, 45, 47, 50, 53, 55, 57, 60, 63]
save_at = [50, 53, 55, 57, 60, 63]

NS = 3
NE = 3
r = 1.2

seeds = [63528,30270,1186,47466,13938,27248,23050,32591,70485,44794,87752,67208,48357,41003,44268,55533,54862,59718,78523,69827,33651,12194,56602]



train_x = []
train_y = []

val_x = []
val_y = []

test_seen_x = []
test_seen_y = []

test_unseen_x = []
test_unseen_y = []



def setSeed(seed):
    newSeed = int(seed)

    from numpy.random import seed
    import random as ran
    
    #get_ipython().run_line_magic('env', 'PYTHONHASHSEED=1')
    ran.seed(newSeed)
    seed(newSeed)
    tf.random.set_seed(newSeed)

    session_conf = tf.compat.v1.ConfigProto()

    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'

    #from tensorflow.keras import backend as K
    #K.set_image_data_format('channels_first')

setSeed(seeds[1])


files = os.listdir("Weights")
for file in files:
    os.remove("Weights/"+file)






# LOAD TRAIN DATA
for file in os.listdir(train_path + "/covid"):
    
    image = cv2.imread(train_path + "/covid/" + file)
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
for file in os.listdir(valid_path + "/covid"):
    
    image = cv2.imread(valid_path + "/covid/" + file)
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

image_shape = train_x[0].shape
print("Input Shape: {}".format(image_shape))
print("Label Shape: {}".format(train_y[0].shape))






#model = ResNet18_exp(2)
#model.build(input_shape = (None,244,244,3))

#model = ResNet18_2(2)
#model.build(input_shape = (None,244,244,3))

model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)

print(model.summary())



#returns the validation loss of the model
def validate():
    return model.evaluate(val_x, val_y, verbose=0)[0]


def validate2():
    y_pred = model.predict(val_x, verbose=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    val_loss = bce(val_y, y_pred).numpy()
    return val_loss




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

        if (epoch+1) in save_at:
            print("\nSaving weights from epoch {} with loss {}".format(epoch, logs["val_loss"]))
            weights.append(model.get_weights())

        self.epoch_tracker += 1
        

#finds the index of an array that has the minimum value
def findMinIndex(arr):
    minVal = math.inf
    minIndex = 0

    for i in range(len(arr)):
        if arr[i] < minVal:
            minVal = arr[i]
            minIndex = i
    
    return minIndex




#SGD optimizer with learning rate and 0.9 momentum
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
              callbacks=weight_saver_callback())






models = list()


print("Creating Ensemble...")
for i in range(n_models):


    #retrieve that weight and remove from lsit of weights
    curr_weight = weights[i]

    #remove loss from loss list

    #create model with those weights
    #model = ResNet18_2(2)
    #model.build(input_shape = (None,244,244,3))
    model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)

    model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

    model.set_weights(curr_weight)

    #add model to list of models
    models.append(model)





def evaluate_models(model_list, x_test, y_test):

    total_examples = len(y_test)
    total_correct = 0

    print("Number of total examples: {}".format(total_examples))

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


#x_test_seen, y_test_seen = test_seen_x.next()
#x_test_unseen, y_test_unseen = test_seen_y.next()

#accuracy = evaluate_models(models, x_test_seen, y_test_seen)


print("\nEnsemble results")


#model evaluation
scores = evaluate_models(models, test_seen_x, test_seen_y)
#print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores)

#model evaluation
scores_unseen =evaluate_models(models, test_unseen_x, test_unseen_y)
print('Test accuracy unseen:', scores_unseen)
