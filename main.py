import train
import predict
#asking the user if they want to train or predict
#---------------------------------------------------------------------------------------------------------------------------
#*********************************************MAIN FUNCTION-----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
def main():
    created = False #no medel created
    while True:
        input = input('1.Create and train Model\n 2.Predict Photo\n3.Exit\n\nPlease Type interger input: ')
        #

        if input == 1:
            #create and train  pretrained model
            while True:
                type = input('\n1.Create and train using VGG19\n2.Create and train using ResNet34 \n\nPlease type integer input: ')
                #Train.py if correct input break from the loop
                check_choose(type)
                #train.py creating
            model = create_train_model(type)
            created = True #model created
            #Training
        #Predict
        if input == 2:
            while True:
                type = input('\n1.Predict from new created model\n2.Predict using using a Pre-trained ResNet34 \n\nPlease type integer input: ')
                #Train.py if correct input break from the loop
                check_choose(type)
                predict(model)

        if input == 3:
            print('Thanks for using Flower Image classifier')
            break

        else:
            print('Please enter correct input:')

if __name__ == "__main__":
	main()
