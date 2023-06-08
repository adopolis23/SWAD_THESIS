from keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split

from ModelGen import Generate_Model_1, Generate_Model_2

#10 output classes nums 0-9
num_classes = 10

#28 x 28 greyscale images
input_shape = (28, 28, 1)

#model parameters
batch_size = 32
learning_rate = 0.0005
epochs = 100




#download data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("x_train.shape = ", x_train.shape)
print("y_train.shape = ", y_train.shape)
print("x_test.shape = ", x_test.shape)
print("y_test.shape = ", y_test.shape)


#normalize images to 0 - 1 range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


#create the velidation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)








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
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_valid, y_valid),
              shuffle=True,
              callbacks=checkpoint())



#model evaluation
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])