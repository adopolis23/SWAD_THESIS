import numpy as np
from sklearn.model_selection import train_test_split
from ModelGen import Generate_Model_1, Generate_Model_2


#10 output classes nums 0-9
num_classes = 10
#28 x 28 greyscale images
input_shape = (28, 28, 1)

model = Generate_Model_2(num_classes, input_shape)


curr = 0
stop = 979

iterations = stop - curr
max_load = 200
curr_avg = 0



whole = int(iterations / max_load)
remainder = iterations % max_load

print("whole is {} and remainder is {}".format(whole, remainder))

weight_set = []
new_weights = list()

for i in range(whole):

    weight_set.clear()
    new_weights.clear()

    for j in range(curr, curr+max_load):
        model.load_weights("Weights/weights_" + str(j) + ".h5")
        weight_set.append(model.get_weights())
        curr += 1
    
    for weights_list_tuple in zip(*weight_set): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )

    model.set_weights(weight_set[0])
    model.save_weights("Weights/AvgWeight_" + str(curr_avg) + ".h5")
    curr_avg += 1








weight_set.clear()
new_weights.clear()
for i in range(curr, curr+remainder):
    model.load_weights("Weights/weights_" + str(j) + ".h5")
    weight_set.append(model.get_weights())
    curr += 1

for weights_list_tuple in zip(*weight_set): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )

model.set_weights(weight_set[0])
model.save_weights("Weights/AvgWeight_" + str(curr_avg) + ".h5")
curr_avg += 1




print("Total Averages is: {}".format(curr_avg))































