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

learning_rate = 0.0009
epochs = 60
batch_size = 64

num_classes = 2


NS = 3
NE = 3
r = 1.2


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
new_weights = list()


#create callback for weight averaging
class swad_callback(keras.callbacks.Callback):

    def __init__(self):

        #list to track loss over training
        self.loss_tracker = []
        self.iteration_tracker = 0


    #function called at the end of every batch
    def on_train_batch_end(self, batch, logs=None):

        #finds the validation loss after this batch
        #this is very slow and this is why this takes a while

        val_loss = validate()[0]


        #save loss and weights for this batch
        if val_loss <= 2.0:
            self.loss_tracker.append(val_loss)


        #weights.append(model.get_weights())
        model.save_weights("Weights/weights_" + str(self.iteration_tracker) + ".h5")

        self.iteration_tracker += 1




    #function called at the end of training
    #NOTE WEIGHT AVERAGING HAPPENS HERE
    def on_train_end(self, logs=None):
        print("\nEnd of Training")

        #optional plot the loss
        plt.plot(self.loss_tracker)
        plt.show()

        #finds the start and end iteration to average weights
        ts, te, l = findStartAndEnd(self.loss_tracker)
        print("ts is {} and te is {} and l is {}".format(ts, te, l))


        #optional save loss to csv
        #df = pd.DataFrame(self.loss_tracker)
        #df.to_csv('loss.csv') 

        print("\nAveraging Weights.")

        ts = int(input("TS:"))
        te = int(input("TE:"))

        new_weights = AverageWeights(model, ts, te, 100)



        '''
        #average up all saved weights and store them in new_weights
        #NOTE Weight averaging!
        for weights_list_tuple in zip(*pruned_weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        '''

        #set model weights to new average
        if len(new_weights) > 0:
            print("\nSetting new model weights.\n")
            model.set_weights(new_weights)
        




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
              callbacks=swad_callback())



#model evaluation
scores = model.evaluate(test_batches, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

#model evaluation
scores_unseen = model.evaluate(test_batches_unseen, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])