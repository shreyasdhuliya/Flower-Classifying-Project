# function for creating and training models
from train import *
#function for predicting pictures
from predict import *
#asking the user if they want to train or predict
#---------------------------------------------------------------------------------------------------------------------------
#*********************************************MAIN FUNCTION-----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
#No model trained initially
model = False
created = False #no medel created

#User interface
while True:
    #input to create, predict or exit
    input_ = input('\n1.Create and train Model\n 2.Predict Photo from newly created model or from Pretrained RestNet Model\n3.Exit\n\nPlease Type Integer inputs:')
   #print(type(input_))   #

    #if 1 then create a model and train it
    if input_ == '1':
            #create and train  pretrained model
        #print('\nin create')
        while True:
            #Options to choose pretrained model
            type_ = input('\n1.Create and train using VGG19\n2.Create and train using ResNet34 \n\nPlease type integer input: ')
                #Train.py if correct input break from the loop
            if type_ == '1' or type_ == '2':
                 break
            else:
                 print('Enter a corerct integer input!')
        #create_train_model() creates pretrained vgg19 or restnet34 and trains based on user input
        #train.py
        model = create_train_model(type_)
        created = True #model created
            #Training

    #Predict from newly trained model or predict from checkpoint
    elif input_ == '2':
        while True:
            option = input('\n1.Predict from new created model \n2.Predict from checkpoint model - RestNet34 - 5 epochs\n\nPlease type integer input: ')
                #Train.py if correct input break from the loop
            if option == '1' or option == '2':
                 break
            else:
                 print('Enter a corerct integer input!')
        #predict image by taking image url from User
        #predict.py
        predict(model,created,option)

    #exit from the loop
    elif input_ == '3':
        print('------------------------------------------------')
        print('\nThank you for using Flower Image classifier\n')
        print('------------------------------------------------')
        break

    else:
        print('Wrong input provided!\n')
