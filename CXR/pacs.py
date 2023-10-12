import deeplake
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


from ResNet18exp import ResNet18_exp
from ModelGen import Generate_Model_2



#get pacs data
train_ds = deeplake.load("hub://activeloop/pacs-train")
train_ds.summary()

val_ds = deeplake.load("hub://activeloop/pacs-val")
val_ds.summary()

test_ds = deeplake.load("hub://activeloop/pacs-test")
test_ds.summary()

train_dl = train_ds.tensorflow()
val_dl = val_ds.tensorflow()
test_dl = test_ds.tensorflow()



model = Generate_Model_2(7, (227, 227, 3))




learning_rate = 0.0001

epochs = 30
batch_size = 16
num_classes = 2



#SGD optimizer with learning rate and 0.9 momentum
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

#compile model with accuracy metric
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



start_time = time.time()

#train the model
model.fit(train_dl,
            validation_data=(val_dl),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=[])

elapsed_time = time.time() - start_time

print("Total training time: {}".format(elapsed_time))








