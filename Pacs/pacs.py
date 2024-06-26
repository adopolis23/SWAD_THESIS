import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import random
import time
import gc

from resnet9exp import ResNet9_exp
from pacs_data_loader import LoadPacsData
from ModelGen import Generate_Model_2
from SwadUtility import AverageWeights, findStartAndEnd, findStartAndEnd2, findStartAndEnd3
from ResNet18exp import ResNet18_exp
from ModelGen import ResNet18_2
from tensorflow.keras.applications.resnet50 import ResNet50

from modified_densenet import DenseNet121

learning_rate = 0.0001
epochs = 20
batch_size = 32
num_classes = 7
runs = 1

#swad parameters
NS = 8
NE = 6
r = 1.2


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



image_size = (224, 224)
image_shape = (224, 224, 3)


train_x, train_y, val_x, val_y, test_seen_x, test_seen_y, test_unseen_x, test_unseen_y = LoadPacsData(train_size_percent=0.10)

print("Train length = {}".format(len(train_x)))
print("Val length = {}".format(len(val_x)))
print("Test_seen length = {}".format(len(test_seen_x)))
print("Test_unseen length = {}".format(len(test_unseen_x)))










def validate():
  y_pred = model.predict(val_x, verbose=0)
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  val_loss = bce(val_y, y_pred).numpy()
  return val_loss

def acc():
    return model.evaluate(val_x, val_y, verbose=0)[1]


new_algo_widths = []
original_algo_widths = []

class checkpoint(tf.keras.callbacks.Callback):

  def __init__(self):
    self.min_loss = 1000000
    self.min_weight = None
    self.loss_tracker = []
    self.accuracy_tracker = []

  def on_train_batch_end(self, batch, logs=None):
    
    val_loss = validate()
    curr_acc = acc()
    self.loss_tracker.append(val_loss)
    self.accuracy_tracker.append(curr_acc)

    if val_loss < self.min_loss:
        #print("\nValidation loss improved saving weights\n")
        self.min_loss = val_loss
        self.min_weight = model.get_weights()

  def on_train_end(self, logs=None):
    print("\nSetting new model weights.\n")
    model.set_weights(self.min_weight)

    df = pd.DataFrame(self.loss_tracker)
    df.to_csv('loss.csv') 
    
    tsp, tep, l = findStartAndEnd2(self.loss_tracker)
    tso, teo, l = findStartAndEnd(self.loss_tracker, NS, NE, r)
    new_algo_widths.append(tep - tsp)
    original_algo_widths.append(teo - tso)

    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    '''
    ax1.plot(self.accuracy_tracker, color='black')
    ax1.axvline(x=tso, color='r')
    ax1.axvline(x=teo, color='b')
    ax1.set(xlabel="Iteration", ylabel="Validation Accuracy")

    ax2.plot(self.accuracy_tracker, color='black')
    ax2.axvline(x=tsp, color='r')
    ax2.axvline(x=tep, color='b')
    ax2.set(xlabel="Iteration", ylabel="Validation Accuracy")
    '''
  

    ax1.plot(self.loss_tracker, color='black')
    ax1.axvline(x=tso, color='r')
    ax1.axvline(x=teo, color='b')
    ax1.set(xlabel="Iteration", ylabel="Validation Loss")

    ax2.plot(self.loss_tracker, color='black')
    ax2.axvline(x=tsp, color='r')
    ax2.axvline(x=tep, color='b')
    ax2.set(xlabel="Iteration", ylabel="Validation Loss")

    for i in range(50, 1100, 50):
       ax1.axvline(x=i,linewidth=0.5, color='gray')
       ax2.axvline(x=i,linewidth=0.5, color='gray')

    plt.show()
    







weights = []
new_weights = list()


#create callback for weight averaging
class swad_callback(tf.keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.loss_tracker = []
        self.iteration_tracker = 0
        self.weights_saved = 0

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


    #function called at the end of every batch
    def on_train_batch_end(self, batch, logs=None):

        #finds the validation loss after this batch
        #this is very slow and this is why this takes a while

        if self.iteration_tracker >= 0:
            val_loss = validate()


            #save loss and weights for this batch
            self.loss_tracker.append(val_loss)


            #weights.append(model.get_weights())
            model.save_weights("Weights/weights_" + str(self.weights_saved) + ".h5")
            self.weights_saved += 1

        self.iteration_tracker += 1


    #function called at the end of training
    #NOTE WEIGHT AVERAGING HAPPENS HERE
    def on_train_end(self, logs=None):
        print("\nEnd of Training")

        #finds the start and end iteration to average weights
        #ts, te, l = findStartAndEnd(self.loss_tracker, NS, NE, r)
        ts, te, l = findStartAndEnd2(self.loss_tracker)
        print("ts is {} and te is {}".format(ts, te))

        #optional plot the loss
        #plt.plot(self.loss_tracker)
        #plt.axvline(x=ts, color='r')
        #plt.axvline(x=te, color='b')
        #plt.show()



        #optional save loss to csv
        df = pd.DataFrame(self.loss_tracker)
        df.to_csv('loss.csv') 

        print("\nAveraging Weights.")

        #ts = int(input("TS:"))
        #te = int(input("TE:"))

        new_weights = AverageWeights(model, ts, te, 200)



        '''
        #average up all saved weights and store them in new_weights
        #NOTE Weight averaging!
        for weights_list_tuple in zip(*saved_weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        '''

        #set model weights to new average
        if len(new_weights) > 0:
            print("\nSetting new model weights.\n")
            model.set_weights(new_weights)



results = []
for i in range(runs):

    print("******* Run Number: {} *******".format(i))

    weights_folder = os.listdir("Weights")
    for file in weights_folder:
        os.remove("Weights/"+file)

    setSeed(seeds[0])

    model = None
    opt = None

    gc.collect()

    #setSeed()

    model = Generate_Model_2(num_classes, image_shape)
    #model = DenseNet121(input_shape=image_shape, classes=num_classes, weights=None)

    #model = ResNet18_exp(7)
    #model.build(input_shape = (None,244,244,3))
    #print(model.summary())

    #model = ResNet9_exp(7)
    #model.build(input_shape = (None, 244, 244, 3))


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

    #model evaluation
    scores_unseen = model.evaluate(test_unseen_x, test_unseen_y, verbose=1)
    print('Test loss unseen:', scores_unseen[0])
    print('Test accuracy unseen:', scores_unseen[1])

    print("******* End Run Number: {} *******".format(i))

    results.append([scores[1], scores_unseen[1]])


df = pd.DataFrame(results)
df.to_csv('multirun_results.csv')

print("\n\n Final Results:\n")

for i, x in enumerate(results):
    print("\nRun: {}, Loss-Seen: {}".format(i, x[0]))
    print("\nRun: {}, Loss-unSeen: {}".format(i, x[1]))



num_iter_per_epoch = int(len(train_x)/batch_size)

#for width in new_algo_widths:
new_avg_epochs_list = [x/num_iter_per_epoch for x in new_algo_widths]
avg_epochs_new = sum(new_avg_epochs_list) / len(new_avg_epochs_list)

new_algo_widths = [x for x in new_algo_widths if x < num_iter_per_epoch]
n_new_sub_epoch = len(new_algo_widths)

original_avg_epochs_list = [x/num_iter_per_epoch for x in original_algo_widths]
avg_epochs_original = sum(original_avg_epochs_list) / len(original_avg_epochs_list)

original_algo_widths = [x for x in original_algo_widths if x < num_iter_per_epoch]
n_original_sub_epoch = len(original_algo_widths)

print("\nProposed algorithm was sub epoch {}%; Avg number of epochs chosen: {}".format((n_new_sub_epoch/runs * 100), avg_epochs_new))
print("Original algorithm was sub epoch {}%; Avg number of epochs chosen: {}".format((n_original_sub_epoch/runs * 100), avg_epochs_original))

    
