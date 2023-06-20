import numpy as np
from sklearn.model_selection import train_test_split
from ModelGen import Generate_Model_1, Generate_Model_2




def AverageWeights(model, ts, te, max_load):

    curr = ts
    stop = te

    iterations = stop - curr

    current_averaged = 0

    whole_averages = int(iterations / max_load)
    remainder = iterations % max_load


    weight_set = []
    new_weights = list()


    #average each whole chunk of weights
    for i in range(whole_averages):

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

        model.set_weights(new_weights)
        model.save_weights("Weights/AvgWeight_" + str(current_averaged) + ".h5")
        current_averaged += 1





    #average the remaining weights if there are any
    if remainder > 0:
        weight_set.clear()
        new_weights.clear()
        for i in range(curr, curr+remainder):
            model.load_weights("Weights/weights_" + str(i) + ".h5")
            weight_set.append(model.get_weights())
            curr += 1

        for weights_list_tuple in zip(*weight_set): 
                    new_weights.append(
                        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                    )

        model.set_weights(new_weights)
        model.save_weights("Weights/AvgWeight_" + str(current_averaged) + ".h5")
        current_averaged += 1

    print("Number of sub-averages is: {}".format(current_averaged))




    #average the sub_averages
    weight_set.clear()
    new_weights.clear()
    for i in range(current_averaged):
        model.load_weights("Weights/AvgWeight_" + str(i) + ".h5")
        weight_set.append(model.get_weights())   

    for weights_list_tuple in zip(*weight_set): 
                new_weights.append(
                    np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
                )


    return new_weights