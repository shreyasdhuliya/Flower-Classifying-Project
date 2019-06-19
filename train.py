# Imports files ----------------------------------------------------------------------------------------------------------------------------------------------------
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def create_model(in):
    if in == 1:
        model = models.vgg19(pretrained=True)
    else:
        model = models.resnet34(pretrained=True)
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
