import pandas as pd
import matplotlib.pyplot as plt
import math


data = pd.read_csv("Loss/Loss_CXR_Baseline.csv")
data.columns = ['index','loss']

data = data["loss"]
data = data.values.tolist()



plt.plot(data)
plt.show()


weights = []
for i in range(1, 150):
    weights.append(i)


NS = 3 #optimum patience
NE = 2 #overfit patience
r = 1 #tolerance ratio











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




