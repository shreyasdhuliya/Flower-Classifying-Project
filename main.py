import train
import predict
#asking the user if they want to train or predict
#---------------------------------------------------------------------------------------------------------------------------
#*********************************************MAIN FUNCTION-----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
def main():
    create = 0
    while True:
        input = input('1.Create Model\n 2.Train An Exisiting Model\n 3.Predict Photo\n4.Exit\n\nPlease Type interger input: ')
        #

        if input == 1:
            #create a pretrained model
            while True:
                type = input('\n1.Create using VGG19\n2.Create using ResNet34\n\nPlease type integer input: ')
                check_choose(type)
                model = create_model(type)
            #Training
        if input == 2:
            While True:
                choose = imput('1.\nTrain the newly created model\n2. Train an Existing model:(Please enter Interger Input)')
                check_choose(choose)
                if choose == 1 or choose ==2:
                    create = 1
                    model = train_model(model,choose)
            #Predict
        if input == 3:
                predict()

        if input == 4:
            print('Thanks for using Flower Image classifier')
            break

        else:
            print('Please enter correct input:')

if __name__ == "__main__":
	main()
