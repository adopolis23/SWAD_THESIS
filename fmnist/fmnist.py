import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from ModelGen import Generate_Model_2, ResNet18_2
from resnet9exp import ResNet9_exp
from SwadUtility import findStartAndEnd2, AverageWeights,findStartAndEnd3, findStartAndEnd
from numpy.random import seed
from ModelGen import Generate_Model_2
import cv2
from ResNet18exp import ResNet18_exp
from ModelGen import ResNet18_2
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import pandas as pd
import gc
import os
from numpy.random import seed
import random as ran

batch_size = 64
num_classes = 10
epochs = 50
runs = 1


NS = 6
NE = 6
r = 1.2

swad_start_iter = 1
rolling_window_size = 75
results = []


#download data
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))


#normalize images to 0 - 1 range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


x_valid = x_train[-5000:]
y_valid = y_train[-5000:]

#add - back in 
x_train = x_train[:2500]
y_train = y_train[:2500]

x_test_ood = x_test.copy()
y_test_ood = y_test.copy()

#create the velidation data
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)
print("x_valid.shape = ", x_valid.shape)
print("y_valid.shape = ", y_valid.shape)





seeds = [63528,30270,1186,47466,13938,27248,23050,32591,70485,44794,87752,67208,48357,41003,44268,55533,54862,59718,78523,69827,33651,12194,56602]


def setSeed(x):
    newSeed = int(x)
    
    ran.seed(newSeed)
    seed(newSeed)
    tf.random.set_seed(newSeed)

    session_conf = tf.compat.v1.ConfigProto()

    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['TF_DETERMINISTIC_OPS'] = 'true'





def add_gauss_noise(image, sigma, mean):
    row,col,ch= image.shape
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

for i, example in enumerate(x_test_ood):
    x_test_ood[i] = add_gauss_noise(example, 0.35, 0)


print("x_test_ood.shape = ", x_test_ood.shape)
print("y_test_ood.shape = ", y_test_ood.shape)



#returns the validation loss of the model
def validate():
    return model.evaluate(x_valid, y_valid, verbose=0)[0]


def validate2():
    y_pred = model.predict(x_valid, verbose=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    val_loss = bce(y_valid, y_pred).numpy()
    return val_loss


def minVal(loss, curr_i, ns):
    min_value = 999999999
    
    for i in range(ns):
        if loss[curr_i - i] < min_value:
            min_value = loss[curr_i - i]
    
    return min_value


def avgLastR(loss, curr_i, ns, r):
    curr_sum = 0
    for i in range(ns):
        curr_sum = curr_sum + loss[curr_i - i]
    
    curr_sum = float(curr_sum / ns) * r
    
    return curr_sum

def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)
    
def avg_fn_a(averaged_model_parameter, model_parameter, num_averaged):
                return np.add(averaged_model_parameter, np.divide(np.subtract(model_parameter, averaged_model_parameter), (num_averaged + 1)))


new_algo_widths = []
original_algo_widths = []
class checkpoint(tf.keras.callbacks.Callback):

    def __init__(self):
        self.min_loss = 1000000
        self.min_weight = None
        self.loss_tracker = []

    def on_train_batch_end(self, batch, logs=None):
        val_loss = validate2()
        self.loss_tracker.append(val_loss)

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
        original_algo_widths.append(teo - tso)
        new_algo_widths.append(tep-tsp)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
    

        tso = 337
        teo = 1864

        tsp = 599


        ax1.plot(self.loss_tracker, color='black')
        ax1.axvline(x=tso, color='r')
        ax1.axvline(x=teo, color='b')
        ax1.set(xlabel="Iteration", ylabel="Validation Loss")

        ax2.plot(self.loss_tracker, color='black')
        ax2.axvline(x=tsp, color='r')
        ax2.axvline(x=tep, color='b')
        ax2.set(xlabel="Iteration", ylabel="Validation Loss")

        for i in range(50, 2200, 50):
            ax1.axvline(x=i,linewidth=0.5, color='gray')
            ax2.axvline(x=i,linewidth=0.5, color='gray')

        plt.show()
        



NS = 6
NE = 6
r = 1.2
#create callback for weight averaging
class swad_callback_paper(tf.keras.callbacks.Callback):

    def __init__(self):
        self.ts = 0
        self.te = (int(len(x_train)/batch_size)-1) * epochs
        self.l = None
        
        self.averaged_model_param = None
        self.num_averaged = 0
        self.loss_tracker = []
        
        self.curr_iter = 0
        
    #function called at the end of every batch
    def on_train_batch_end(self, batch, logs=None):
        self.loss_tracker.append(validate2())
        
        if self.curr_iter >= 10:
            if self.l == None:
                if self.loss_tracker[self.curr_iter-NS+1] == minVal(self.loss_tracker, self.curr_iter, NS):
                    self.ts = batch - NS + 1
                    self.l = avgLastR(self.loss_tracker, self.curr_iter, NS, r)
                    print("L set to {} ***".format(self.l))
                    print("TS set to {} ***".format(self.ts))
            if self.l != None and self.l < minVal(self.loss_tracker, self.curr_iter, NE):
                te = self.curr_iter - NE
                print("TE set to {} ***".format(self.te))

            #if the start iteration has been encountered
            if self.l != None:
                if self.averaged_model_param == None:
                    self.averaged_model_param = model.get_weights()
                else:
                    self.averaged_model_param = avg_fn_a(self.averaged_model_param, model.get_weights(), self.num_averaged)
                    self.num_averaged += 1

            #early stopping condition
            if self.curr_iter > self.te:
                self.model.stop_training = True
    
        self.curr_iter += 1
        
    def on_train_end(self, logs=None):
        print("TS is {}".format(self.ts))
        print("TE is {}".format(self.te))
        
        df = pd.DataFrame(self.loss_tracker)
        df.to_csv('loss.csv') 
        
        print("Setting model weights...")
        model.set_weights(self.averaged_model_param)





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
        ts, te, l = findStartAndEnd2(full_loss)
        print("ts is {} and te is {}".format(ts, te))

        #optional plot the loss
        #plt.plot(full_loss)
        #lt.axvline(x=ts, color='r')
        #plt.axvline(x=te, color='b')
        #plt.show()

        #optional save loss to csv
        df = pd.DataFrame(full_loss)
        df.to_csv('loss.csv') 

        print("\nAveraging Weights.")

        for i, weight in enumerate(full_weights):
            model.save_weights("Weights/weights_" + str(i) + ".h5")

        new_weights = AverageWeights(model, ts, te, 200)

        #set model weights to new average
        if len(new_weights) > 0:
            print("\nSetting new model weights.\n")
            model.set_weights(new_weights)




for i in range(runs):
    print("******* Run Number: {} *******".format(i))
    setSeed(seeds[i])
    model = None
    gc.collect()
    
    weights_folder = os.listdir("Weights")
    for file in weights_folder:
        os.remove("Weights/"+file)
    
    model = Generate_Model_2(10, (28, 28, 1))

    #model = ResNet9_exp(10)
    #model.build(input_shape = (None, 28, 28, 1))
        
    #model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(28, 28, 1), classes=num_classes, weights=None)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 

    #compile model with accuracy metric
    model.compile(loss="categorical_crossentropy",
                optimizer=opt,
                metrics=['accuracy'])

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        validation_data=(x_valid, y_valid),
        epochs=epochs,
        shuffle=True,
        callbacks=[checkpoint()]
    )
    
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss seen:', scores[0])
    print('Test accuracy seen:', scores[1])

    scores_unseen = model.evaluate(x_test_ood, y_test_ood, verbose=1)
    print('Test loss unseen:', scores_unseen[0])
    print('Test accuracy unseen:', scores_unseen[1])
    
    results.append([scores[1], scores_unseen[1]])
    

df = pd.DataFrame(results)
df.to_csv('multirun_results.csv')

print("\n\n Final Results:\n")

for i, x in enumerate(results):
    print("\nRun: {}, Loss-Seen: {}".format(i, x[0]))
    print("\nRun: {}, Loss-unSeen: {}".format(i, x[1]))


num_iter_per_epoch = int(len(x_train)/batch_size)

print(original_algo_widths)
print(new_algo_widths)


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