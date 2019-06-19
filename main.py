import train
import predict
#asking the user if they want to train or predict
#---------------------------------------------------------------------------------------------------------------------------
#*********************************************MAIN FUNCTION-----------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
def main():
    while True:
        input = input('1.Create Model\n 2.Train Exisiting Model\n 3.Predict Photo\n\nPlease Type interger input: ')
        #
        While True:
            if input == 1:
                #create a pretrained model
                while True:
                    choose = input('\n1.Create using VGG19\n2.Create using ResNet34\n\nPlease type integer input: ')
                    if choose == 1 or choose == 2:
                        #function returns a pretrained VGG19 or ResNet34 model
                        model = create_model(choose)
                        print('Pretrained Model successfully created')
                        #break from nearest while loop
                        break
                    else:
                        print('\nRe-enter -- WRONG INPUT PROVIDED:')
                while True:


if __name__ == "__main__":
	main()
