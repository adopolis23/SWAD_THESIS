import tensorflow as tf
import os
import cv2
import gc
import pandas as pd
import math
import numpy as np
import time
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
image_shape = (244, 244, 3)
learning_rate = 0.00005

epochs = 50
batch_size = 16
num_classes = 2

#swad parameters
NS = 0
NE = 0
r = 1.2

rolling_window_size = 50
swad_start_iter = 1


train_x = []
train_y = []

val_x = []
val_y = []

test_seen_x = []
test_seen_y = []

test_unseen_x = []
test_unseen_y = []

seeds = [63528,30270,1186,47466,13938,27248,23050,32591,70485,44794,87752,67208,48357,41003,44268,55533,54862,59718,78523,69827,33651,12194,56602]


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



#model = Generate_Model_2(num_classes, image_shape)
#model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)
#model = ResNet18(input_shape=image_shape, classes=num_classes)

#model = ResNet18_2(2)
#model.build(input_shape = (None,244,244,3))

model = ResNet18_exp(2)
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

    def on_train_batch_end(self, epoch, logs=None):
        val_loss = validate2()

        if val_loss < self.min_loss:
            print("\nValidation loss improved saving weights\n")
            self.min_loss = val_loss
            self.min_weight = model.get_weights()

    def on_train_end(self, logs=None):
        print("\nSetting new model weights.\n")
        model.set_weights(self.min_weight)






weights = []
new_weights = list()


#create callback for weight averaging
class swad_callback(tf.keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.iteration_tracker = 0
        self.weights_saved = 0
        self.min_loss = 100000000

        self.rolling_last_weights = []
        self.rolling_last_loss = []

        self.curr_best_weight_hist = []
        self.curr_best_loss_hist = []

        self.curr_best_weight_right = []
        self.curr_best_loss_right = []

        self.best_loss_iteration = 0

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


    #function called at the end of every batch
    def on_train_batch_end(self, batch, logs=None):

        #finds the validation loss after this batch
        #this is very slow and this is why this takes a while

        if self.iteration_tracker >= swad_start_iter:
            val_loss = validate2()



            #keeping track of the rolling window
            self.rolling_last_loss.append(val_loss)
            while len(self.rolling_last_loss) > rolling_window_size:
                self.rolling_last_loss.pop(0)

            self.rolling_last_weights.append(model.get_weights())
            while len(self.rolling_last_weights) > rolling_window_size:
                self.rolling_last_weights.pop(0)



            #new min loss found
            if val_loss < self.min_loss:
                self.min_loss = val_loss
                self.curr_best_weight_hist = self.rolling_last_weights
                self.curr_best_loss_hist = self.rolling_last_loss

                self.curr_best_loss_right.clear()
                self.curr_best_weight_right.clear()

                self.best_loss_iteration = self.iteration_tracker
            
            #save the weights after the minimum
            if len(self.curr_best_loss_right) < rolling_window_size:
                self.curr_best_loss_right.append(val_loss)
                self.curr_best_weight_right.append(model.get_weights())
            
            #debugging
            #print("{} : {}".format(len(self.curr_best_loss_hist), len(self.curr_best_loss_right)))




        self.iteration_tracker += 1


    #function called at the end of training
    #NOTE WEIGHT AVERAGING HAPPENS HERE
    def on_train_end(self, logs=None):
        print("\nEnd of Training")

        print("Absolute best loss at: {}".format(self.best_loss_iteration))


        full_loss = self.curr_best_loss_hist + self.curr_best_loss_right
        full_weights = self.curr_best_weight_hist + self.curr_best_weight_right

        #finds the start and end iteration to average weights
        ts, te, l = findStartAndEnd2(full_loss, NS, NE, r)
        print("ts is {} and te is {}".format(ts, te))

        #optional plot the loss
        #plt.plot(full_loss)
        #plt.axvline(x=ts, color='r')
        #plt.axvline(x=te, color='b')
        #plt.show()



        #optional save loss to csv
        df = pd.DataFrame(full_loss)
        df.to_csv('loss.csv') 

        print("\nAveraging Weights.")

        #ts = int(input("TS:"))
        #te = int(input("TE:"))


        for i, weight in enumerate(full_weights):
            model.save_weights("Weights/weights_" + str(i) + ".h5")


        new_weights = AverageWeights(model, 0, ((rolling_window_size*2) - 1), 200)


        '''
        #average up all saved weights and store them in new_weights
        #NOTE Weight averaging!
        for weights_list_tuple in zip(*saved_weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )8
        '''

        #set model weights to new average
        if len(new_weights) > 0:
            print("\nSetting new model weights.\n")
            model.set_weights(new_weights)






#normal Keras stuff


#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])


#model.load_weights("PretrainedWeights/ResNet18r/ResNet18rWeightsEpoch50.h5")

start_time = time.time()
#train the model
model.fit(x=np.array(train_x, np.float32),
            y=np.array(train_y, np.float32),
            validation_data=(val_x, val_y),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=swad_callback())

elapsed_time = time.time() - start_time
print("Total time is: {}".format(elapsed_time))


#model evaluation
scores = model.evaluate(test_seen_x, test_seen_y, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

#model evaluation
scores_unseen = model.evaluate(test_unseen_x, test_unseen_y, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])


