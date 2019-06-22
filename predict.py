from train import *

from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def process_image():

    while True:

            try:
                image_path = input('please enter image path:')
                #print('in try')
                #int(image_path)
                pil_image = Image.open(image_path)
                break
             #if file note found try again
            except FileNotFoundError:
                print("\nWrong path given\n")
    #opening image
    #scaling to 256 on lower side
    width,height = pil_image.size
    if width >height:
        pil_image.thumbnail([400,256])
    else:
        pil_image.thumbnail([256,400])

    #Cropping image
    width,height  =pil_image.size
    crop_rect = left,top,right,bottom = int(width/2) - 112,int(height/2) - 112,int(width/2) + 112,int(height/2) + 112
    pil_image = pil_image.crop((left, top, right, bottom))

    #Converting to numpy and converting to float between [0,1]
    np_image = np.array(pil_image)
    np_image = np_image/256

    #normalizing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    #print(np_image.shape)
    np_image = np.transpose(np_image, (2, 0, 1))
    #print(type(np_image))

    return np_image

#
def predict(model,created,option):
    ''' Predict the class (or classes) of an image using a trained deep learning model.

    Args:
    model:
        - False: if model not trained and passed in the functional.
        - trained by function call in main.py

    created:if model created after calling create_train_model() in main.py
        - True:If model created using create_train_model() in main.py
        - False:If model not created and passed in the functional

    option: User option to predict using checkpoint or newly created model
        - '1': if predict using newly created using create_train_model() or in predict()


    prints:
    top 3 prbabiliy and Class names

    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #if model created and predict from newly created '1'
    if created == True and option == '1':
        model.to(device);
        image = process_image()
        model.eval()
        with torch.no_grad():
            image_t = torch.from_numpy(image)
            #print(image.shape)
            image_t = image_t.unsqueeze_(0)
            image_t = image_t.float()

            image_t = image_t.to(device)
            logps = model.forward(image_t)

            import json
            #importing json
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)

            #calculating
            while True:
                top_k = input('Enter the number of top probabilities to be printed: ')
                try:
                    top_k = int(top_k)
                    if top_k < 102 and top_k > 1:
                        break
                    else:
                        print("Please enter inter values more than 1 and less than 102!")

                except ValueError:
                    print("Please enter inter values more than 1 and less than 102")

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(top_k, dim = 1)
            top_p = top_p.cpu().numpy().tolist()[0]
            top_class = top_class.cpu().numpy().tolist()[0]
            #print(top_p, top_class)

            #Converting class number to Flower anme using dict  cat_to_name
            list_str = [cat_to_name.get(str(x)) for x in top_class]
            print('\nNAme \t\t\t  Probability')
            for i in range (0,top_k):
                print(list_str[i] + '\t\t' + str(top_p[i]))


    #if model not created and predict from newly created '1'
    #create a model and train the model first
    elif created == False and option == '1':
        print("Create and train a model first")
        while True:
            type_ = input('\n1.Create and train using VGG19\n2.Create and train using ResNet34 \n\nPlease type integer input: ')
                #Train.py if correct input break from the loop
            if type_ == '1' or type_ == '2':
                 break
            else:
                 print('Enter a corerct integer input!')
        #create the new model and train it to predict
        model = create_train_model(type_)
        created = True
        #Call predict again to predict from newly created model
        predict(model,created,'1')


    elif option == '2':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print("\n1Loading model from checkpoint.................")
            checkpoint = torch.load('checkpoint.pth')
            #print('load checkpoint')
            model_c = models.resnet34(pretrained=True)
            model_c.to(device);
            model_c.fc = checkpoint['fc']
            model_c.load_state_dict(checkpoint['state_dict'])
            model_c.epochs = checkpoint['epochs']
            optimizer = optim.Adam(model_c.fc.parameters(), lr=0.001)
            optimizer.load_state_dict(checkpoint['optimizer'])

            model_c.class_to_idx = checkpoint['class_to_idx']
            created = True
            print("\nModel loaded ...............")
            predict(model_c,created,'1')

        else:
            print('\t\t \n*************************************\n\t\t\tWARNING:Please enable GPU and re - run the program\n\t\t*********************************')
