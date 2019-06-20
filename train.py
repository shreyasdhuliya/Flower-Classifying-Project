#----------------------------------------------------------------------------
# Imports Starts here ----------------------------------------------------------
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import matplotlib.pyplot as plt
import torchS
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
#--------------------IMPORT ENDS HERE--------------------------------------
#------------------------------------------------------------------------------
#---------------------------------------------------------------------------
#-------------------check functions Here----------------------------------
def check_choose(in):
    if in == 1 or in == 2:
        return break
    else:
        print('Enter a corerct integer input!')

def validate(epochs,lr):
    if type(epochs) == int and epochs > 4 and type(lr) == float and lr > 0.05:
        return break
    else:
        print('Please type epochs(int) less than 4 and lr(float) less than 0.05!')


#Fuction creates a vgg19 or resnet34 and asks user for number of
def create_train_model(in):
    '''
    This function creates a model with pretrained vgg19 or Resnet34
    The classifier is changed for 102 outputs
    Itializes the hidden layers by the user

    Args:
    option: 1 for vgg19 and 2 for resnet 34

    Return:
    Pretrained model with classifier mentioned

    '''
    if in == 1:
        model = models.vgg19(pretrained=True)
        while True:
            hidden = imput('Choose number of hidden layer, Input layers 25088 and ouput layers 102:')
            if type(hidden) == int and hidden < 4096 and hidden > 80:
                break
            else:
                print('Please enter integer value between 80 and 4096!')
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('dropout1',nn.Dropout(0.4)),
                          ('fc1', nn.Linear(25088, hidden)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
        model = train_model(model,in)

    else:
        model = models.resnet34(pretrained=True)
        while True:
            hidden = imput('Choose number of hidden layer, Input layers 512 and ouput layers 102:')
            if type(hidden) == int and hidden < 512 and hidden > 80:
                break
            else:
                print('Please enter integer value between 80 and 512!')
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('dropout1',nn.Dropout(0.4)),
                          ('fc1', nn.Linear(512, hidden)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.fc = classifier
        model = train_model(model,in)

    return(model)

def train_model(model):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    val_test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder -----------------------------------------------------------------------------
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_test_transforms)


    train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloaders = torch.utils.data.DataLoader(val_data, batch_size=32)

    import json
    #importing json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    while True:
        epochs = input('Enter the number of epochs:')
        lr = input('Enter the learning rate:')
        validate(epochs,lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.NLLLoss()

    if in == 1:
        optimizer = optim.Adam(model.classifier.parameters(), lr)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr)

    model.to(device);

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in train_dataloaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_dataloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        val_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {val_loss/len(val_dataloaders):.3f}.. "
                      f"Validation accuracy: {(accuracy/len(val_dataloaders))*100:.3f}")
                running_loss = 0
                model.train()



#import json file --------------------------------------------------------------------------------------------------------
