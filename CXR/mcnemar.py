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
from SwadUtility import AverageWeights
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import DenseNet201, DenseNet121 #dense 121 working
from tensorflow.keras.applications.efficientnet import EfficientNetB1 #working

from ModelGen import ResNet18_2

train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test-seen"
test_path_unseen = "data/test-unseen"



num_classes = 2
image_size = (244, 244)
image_shape = (244, 244, 3)
batch_size = 16

learning_rate = 0.0001


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

setSeed(1 * 406)


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



loss = pd.read_csv("loss.csv")
loss_vals = list(loss.iloc[:,1])

min_index = 0
min_val = 1000000000
for i, x in enumerate(loss_vals):
    if x < min_val:
        min_val = x
        min_index = i

print("Lowest loss at iteration: {}".format(min_index))

plt.plot(loss_vals)
plt.show()



#create model A


#modelA = ResNet18_2(2)
#modelA.build(input_shape = (None,244,244,3))
#model = EfficientNetB1(input_shape=image_shape, classes=num_classes, weights=None)
modelA = Generate_Model_2(num_classes, image_shape)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
modelA.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

modelA.load_weights('Weights/weights_' + str(min_index) + '.h5')




#create model B

#modelB = ResNet18_2(2)
#modelB.build(input_shape = (None,244,244,3))
#model = EfficientNetB1(input_shape=image_shape, classes=num_classes, weights=None)
modelB = Generate_Model_2(num_classes, image_shape)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
modelB.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])



ts = int(input("TS:"))
te = int(input("TE:"))

if ts == te:
    modelB.load_weights('Weights/weights_' + str(ts) + '.h5')
else:
    new_weights = AverageWeights(modelB, ts, te, 200)

    print("\nSetting new model weights.\n")
    modelB.set_weights(new_weights)







print("Score for model A:\n")

#model evaluation
scores = modelA.evaluate(test_seen_x, test_seen_y, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

#model evaluation
scores_unseen = modelA.evaluate(test_unseen_x, test_unseen_y, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])


print("Score for model B:\n")

#model evaluation
scores = modelB.evaluate(test_seen_x, test_seen_y, verbose=1)
print('Test loss seen:', scores[0])
print('Test accuracy seen:', scores[1])

#model evaluation
scores_unseen = modelB.evaluate(test_unseen_x, test_unseen_y, verbose=1)
print('Test loss unseen:', scores_unseen[0])
print('Test accuracy unseen:', scores_unseen[1])


#calculating mcnemars test

count_yes_no = 0
count_no_yes = 0
count_yes_yes = 0
count_no_no = 0

modelA_pred = modelA.predict(test_unseen_x, verbose=1)
modelB_pred = modelB.predict(test_unseen_x, verbose=1)

for i, y in enumerate(test_unseen_y):
    modelA_num = np.argmax(modelA_pred[i])
    modelB_num = np.argmax(modelB_pred[i])

    correct_answer = np.argmax(y)

    if modelA_num == correct_answer and modelB_num != correct_answer:
        count_yes_no += 1
    elif modelA_num != correct_answer and modelB_num == correct_answer:
        count_no_yes += 1
    elif modelA_num == correct_answer and modelB_num == correct_answer:
        count_yes_yes += 1
    elif modelA_num != correct_answer and modelB_num != correct_answer:
        count_no_no += 1

print("Yes No : {}".format(count_yes_no))
print("No Yes : {}".format(count_no_yes))

#mcnemars_stat = ((count_yes_no-count_no_yes) * (count_yes_no-count_no_yes)) / (count_yes_no + count_no_yes)
#print("\nMcNemars test statistic is: {}".format(mcnemars_stat))



from statsmodels.stats.contingency_tables import mcnemar
matrix = [[count_yes_yes, count_yes_no],[count_no_yes, count_no_no]]

result = mcnemar(matrix, exact=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
 print('Same proportions of errors (fail to reject H0)')
else:
 print('Different proportions of errors (reject H0)')




