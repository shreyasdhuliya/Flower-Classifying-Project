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

def create_model(in):
    if in == 1:
        model = models.vgg19(pretrained=True)
        S

    else:
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                          ('dropout1',nn.Dropout(0.4)),
                          ('fc1', nn.Linear(512, 256)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.fc = classifier

    return(model)

#saving directeries ----------------------------------------------------------------------------------------------------
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Transforms ------------------------------------------------------------------------------------------------------------
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

#same transform for validation and test images
val_test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder -----------------------------------------------------------------------------
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=val_test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders---------------------------------------------
train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
#Don't shuffle test and validation images in batch
val_dataloaders = torch.utils.data.DataLoader(val_data, batch_size=32)
test_dataloaders = torch.utils.data.DataLoader(test_data, batch_size=32)


#import json file --------------------------------------------------------------------------------------------------------
import json
#importing json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
