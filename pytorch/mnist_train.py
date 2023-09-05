import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

import numpy as np
import pandas as pd
import time




from pytorch_models import create_lenet
from pytorch_models import ConvMixer




batch_size = 32
epochs = 20
learning_rate = 0.0001




#check for gpu availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA Device Detected: {}".format(str(device)))
else:
    device = torch.device("cpu")
    print("No Cuda Available")
device



#create the transorm to_tensor
T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


#download the data
train_data = torchvision.datasets.CIFAR10('cifar10_data', train=True, download=True, transform=T)

print(len(train_data))
train_data, val_data = torch.utils.data.random_split(train_data, [45000, 5000])

test_data = torchvision.datasets.CIFAR10('cifar10_data', train=False, download=True, transform=T)


#create the data loader
train_dl = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = batch_size)
test_dl = torch.utils.data.DataLoader(test_data, batch_size = batch_size)




def validate(model, data):
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total


#91.26 expected accuracy
cnn = ConvMixer(dim=128, depth=4, patch_size=1, kernel_size=8, n_classes=10).to(device)


def train(numb_epoch=3, lr=1e-3, device="cpu"):
    accuracies = []

    #create model
    #cnn = create_lenet().to(device)

  

    #loss function
    cec = nn.CrossEntropyLoss()

    #adam optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    max_accuracy = 0

    #main training loop
    for epoch in range(numb_epoch):
        start_time = time.time()

        #for each batch in the training dataset
        for i, (images, labels) in enumerate(train_dl):
            #print("Training batch number {}".format(i))

            #send images and labels to device
            images = images.to(device)
            labels = labels.to(device)

            #init the optimizer
            optimizer.zero_grad()

            #predict on images
            pred = cnn(images)

            #calculate loss
            loss = cec(pred, labels)

            #backprob algo
            loss.backward()
            optimizer.step()

        #validation accuracy calculation
        accuracy = float(validate(cnn, val_dl))
        accuracies.append(accuracy)

        #if accuracy is better than current best save weights
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        
        elapsed_time = time.time() - start_time
        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%', "Time:", elapsed_time)

    #plot accuracies and save best weights
    plt.plot(accuracies)
    plt.show()
    return best_model

print("Starting Training")
best_model = train(epochs, learning_rate, device)
print("Training End")

final_accuracy = float(validate(cnn, test_dl))
print("Final Accuracy Is: {}".format(final_accuracy))


#print("Total examples in train_data is: {}".format(len(train_data)))
#print("Total examples in train_dl is: {}".format(len(train_dl)))