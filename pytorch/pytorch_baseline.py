import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy

batch_size = 1
epochs = 1

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

#train and val data
train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

#train and val data loaders
train_dl = torch.utils.data.DataLoader(train_data, batch_size = batch_size)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = batch_size)

def create_lenet():
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Dropout(0.5),

        nn.Conv2d(32, 32, 3, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Dropout(0.5),

        nn.Flatten(),

        nn.Linear(1152, 120),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(120, 10)
    )
    return model


def validate(model, data, device="cpu"):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total






def train(numb_epoch=3, lr=1e-3, device="cpu"):

    accuracies = []
    cnn = create_lenet().to(device)
    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0


    for epoch in range(numb_epoch):
        print("Start of Epoch: {}".format(epoch))
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = loss_func(pred, labels)
            loss.backward()
            optimizer.step()


        accuracy = float(validate(cnn, val_dl))
        accuracies.append(accuracy)

        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy: ", accuracy)
        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')

    plt.plot(accuracies)
    return best_model








if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")


print("Training Start.")
lenet = train(epochs, device=device)



