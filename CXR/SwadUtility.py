import numpy as np




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



def findStartAndEnd(val_loss, NS, NE, r):
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



def findArrayMin(arr):
    minIndex = 0
    minVal = arr[minIndex]

    for i, x in enumerate(arr):
        if x < minVal:
            minIndex = i
            minVal = x
    
    return minIndex, minVal


#updated find start and end
def findStartAndEnd2(loss, NS, NE, r=1.0):
    
    minIndex, minVal = findArrayMin(loss)

    left_threshold = (r / NS) * sum(loss[minIndex-NS+1:minIndex+1])
    right_threshold = (r / NS) * sum(loss[minIndex-NS+1:minIndex+1])

    pass