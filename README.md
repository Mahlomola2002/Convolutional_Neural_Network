# Convolutional_Neural_Network

PyTorch is used in this code to train a Convolutional Neural Network (CNN) on the CIFAR10 dataset. The network's purpose is to categorize photos from the CIFAR10 dataset into one of ten categories (for example, airplane, automobile, bird, and so on).

The code begins by importing the relevant PyTorch packages, such as torch, torchvision, and transformations. It then specifies two distinct transform sequences. The first transform sequence, "transform," turns the images to a PyTorch tensor and normalizes them to a mean and standard deviation of 0.5. The second transform sequence, "transform2," does the same as "transform," but adds random horizontal flipping and 20-pixel cropping.

The torchvision.datasets function is then used to load the CIFAR10 dataset.CIFAR10 is a function. There are two datasets created: one for training and one for testing. To normalize the pictures, the trainset and testset are run through the previously defined transform sequence "transform".

The torch.utils.data module is then used to develop data loaders.DataLoader is a function that loads data. During training and testing, the data loaders will feed data to the neural network in batches.

The nn.Module class is used to specify the neural network architecture. Two convolutional layers with batch normalization and max pooling precede two fully connected layers in the network.

Then the training and testing functions are defined. The train function takes a network, training data loader, loss function, optimizer, and device as inputs. It then iterates through the training data loader, computes the loss, sends it over the network, and updates the weights. This function returns the average loss of all batches. The test function takes the network, test data loader, and device as inputs. Evaluate the network on a test data set and return the accuracy.

The device is then identified using either a CUDA-enabled GPU ("cuda"), PyTorch's multi-process service ("mps"), or a CPU ("cpu").

 The learning rate and momentum are defined next. The CNN is then initialized and moved to the identified device.

The loss function, optimizer, and learning rate scheduler are defined next. The loss function used is CrossEntropyLoss, and the optimizer used is Stochastic Gradient Descent (SGD). The learning rate is set to 0.005 and the momentum to 0.9.

The CNN is trained for 15 epochs using a for loop. During each epoch, the train and test functions are called, and the train loss and test accuracy are printed to the console. The output shows the progress of the training process
 
 
for makefile:
makefile have venv || virtualenv venv in the test, the reason why included this part instead of python3 venv is because it was not working in the senior lab PC, so the purpose of this test case is to create the virtual environment , to run this make go to terminal and enter make, then it will create virtual environment and download all the packages 

