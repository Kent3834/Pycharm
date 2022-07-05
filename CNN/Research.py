import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, models, transforms, utils
from torchvision.datasets import ImageFolder
import torchvision
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
import pandas as pd
from time import perf_counter

torch.cuda.empty_cache()
torch.manual_seed(1)    # reproducible

img_folder = "D:/Dataset for train/archive/"



T = transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




dataset = ImageFolder(img_folder, transform=T)
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*.8), len(dataset)-int(len(dataset)*.8)])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=15,shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=15,shuffle=True)

cnn = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
cnn.fc= nn.Linear(1024, 10)


print(cnn)  # net architecture


model=cnn  #no cuda

LR = 0.0001  #Learning Rate
optimiser = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss = nn.CrossEntropyLoss()   # the target label is not one-hotted

nb_epochs = 3   # Num of training
acc_tot=np.zeros(nb_epochs)
start = perf_counter()    #Start counting of training
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    model.train()
    for batch in train_loader:

        x,y = batch
        if(torch.cuda.is_available()==True):
            x=x.cuda()
            y=y.cuda()


        # 1 forward
        l = model(x) # l: logits

        #2 compute the objective function
        J = loss(l,y)

        # 3 cleaning the gradients
        model.zero_grad()
        # optimiser.zero_grad()
        # params.grad.zero_()

        # 4 accumulate the partial derivatives of J wrt params
        J.backward()

        # 5 step in the opposite direction of the gradient
        optimiser.step()



        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}', end=', ')
    print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')


    losses = list()
    accuracies = list()
    model.eval()
    for batch in test_loader:
        x,y = batch
        if(torch.cuda.is_available()==True):
            x=x.cuda()
            y=y.cuda()

        with torch.no_grad():
            l = model(x)

        #2 compute the objective function
        J = loss(l,y)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}',end=', ')
    print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')
    acc_tot[epoch]=torch.tensor(accuracies).mean().numpy()
end = perf_counter()  #End counting of training
Time = int(end-start)/60 #Time of the training in minutes

def imformat(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return (inp)

class_names = dataset.classes
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca", "spider": "ragno", "squirrel": "scoiattolo"}
t_inv = {v: k for k, v in translate.items()}

train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=9)
plt.figure(figsize=(10, 12))


inputs, classes = next(iter(test_loader))
preds = model(inputs.cuda()).argmax(dim=1)

for i in range(0, 9):
    ax = plt.subplot(3, 3, i + 1)
    img = imformat(inputs[i])

    plt.imshow((img))


    try:
        plt.title('True:' + str(t_inv[class_names[classes[i]]]) + '    Pred:' + str(t_inv[class_names[preds[i]]]))
    except:
        plt.title(
            'True:' + str(translate[class_names[classes[i]]]) + '    Pred:' + str(translate[class_names[preds[i]]]))
    if (i == 9):
        plt.axis("off")

print('Training Time = '+ str(Time) +' (minutes)')
plt.show()

