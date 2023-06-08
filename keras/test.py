import pandas as pd
import matplotlib.pyplot as plt
import math


data = pd.read_csv("loss.csv")
data.columns = ['index','loss']

data = data["loss"]
data = data.values.tolist()



for i in range(60, 150):
    data[i] += (i-60)*0.0001


plt.plot(data)
plt.show()


weights = []
for i in range(1, 150):
    weights.append(i)


NS = 3 #optimum patience
NE = 2 #overfit patience
r = 1 #tolerance ratio


total_weights = []
def swad_train(iterations):
    ts = 0
    te = iterations
    l = None

    hist = []
    loss = []

    saving = False

    for i in range(0, iterations):

        #training loop get new loss for this iterations
        loss.append(data[i])
        ####

        


        if saving:
            
            while len(hist) > 0:
                total_weights.append(hist.pop(0))

            total_weights.append(weights[i])

        else:
            #store the last NS weights
            hist.append(weights[i])
            if len(hist) > NS:
                hist.pop(0)


        if i > NS:
            minNe = 1000000
            for j in range(NE):
                print("Index: {}".format(i-j))
                if minNe > loss[i - j]:
                    minNe = loss[i-j]
            

            if l == None:

                minNs = 1000000
                for j in range(NS):
                    if minNs > loss[i-j]:
                        minNs = loss[i-j]
                
                if loss[i-NS+1] == minNs:
                    saving = True

                    sum = 0
                    for j in range(NS-1):
                        sum += loss[i-j]
                    l = (r / NS) * sum
            
            elif l < minNe:
                saving = False

                for j in range(NE):
                    total_weights.pop(len(total_weights)-1)
                
                #stop training early
                break














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

                print("TS set")
                ts = i-NS+1
                sums = 0
                for j in range(NS):
                    sums = sums + val_loss[i-j]
                l = (r/NS)*sums
        
        elif l < min1:
            print("TE set")
            te = i-NE
            break
    return ts, te, l




ts, te, l = findStartAndEnd(data)
print("ts is {} and te is {} and l is {}".format(ts, te, l))



swad_train(150)

print(total_weights)

