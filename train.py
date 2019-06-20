#----------------------------------------------------------------------------
# Imports Starts here ----------------------------------------------------------
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
#--------------------IMPORT ENDS HERE--------------------------------------
#------------------------------------------------------------------------------


#Fuction creates a vgg19 or resnet34 and asks user for number of
def create_train_model(in_):
    '''
    This function creates a model with pretrained vgg19 or Resnet34
    The classifier is changed for 102 outputs
    Itializes the hidden layers by the user

    Args:
    option: 1 for vgg19 and 2 for resnet 34

    Return:
    Pretrained model with classifier mentioned

    '''
    if in_ == '1':
        model = models.vgg19(pretrained=True)
        while True:
            hidden = input('Choose number of hidden layer, Input layers 25088 and ouput layers 102:')
            try:
                hidden = int(hidden)
                if hidden <= 4096 and hidden >= 80:
                    break
                else:
                    print('\nPlease enter integer value between 80 and 4096!\n')
            except ValueError:
                print("\nNot an integer!\n")

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
        model = train_model(model,in_)

    else:
        model = models.resnet34(pretrained=True)
        while True:
            hidden = input('Choose number of hidden layer, Input layers 512 and ouput layers 102:')
            try:
                hidden = int(hidden)
                if hidden <= 512 and hidden >= 80:
                    break
                else:
                    print('\nPlease enter integer value between 80 and 512!\n')
            except ValueError:
                print("\nNot an integer!\n")
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
        model = train_model(model,in_)

    return(model)

def train_model(model,in_):

    train_dir = 'train'
    valid_dir = 'valid'
    test_dir = 'test'

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
        try:
                epochs = int(epochs)
                lr = float(lr)
                if epochs < 4 and lr < 0.05:
                    break
                else:
                    print('\nPlease enter epochs less than 4 and learning rate less than 0.05\n')
        except ValueError:
            print("\nNot an integer for epoch or float for learning rate!\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)
    criterion = nn.NLLLoss()

    if in_ == '1':
        optimizer = optim.Adam(model.classifier.parameters(), lr)
    else:
        optimizer = optim.Adam(model.fc.parameters(), lr)

    if device.type == 'cuda':
        print("\n**********************************************")
        print('Training model Please wait\n************************************************')
        print('Printing trainng and validation loss\n\n')

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
            print('model trained with accuracy' + str(accuracy/len(val_dataloaders) + 'epochs' + str(epochs))
        return model

    else:
        print('\nCuda not enabled. Please enable cuda and run again')
        print('***************Model not trained******************')
        print('*******Create and Train After enabling GPU********')
        return model



#import json file --------------------------------------------------------------------------------------------------------
