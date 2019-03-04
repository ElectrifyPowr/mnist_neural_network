# Classifier of Handwritten Number Images
## Using a Deep Neural Network & the MNIST Dataset

This is an implementation of a deep neural network for <b>classifying handwritten numbers between 0 and 9</b>.<br>
It can achieve an <b>accuracy of 97%</b>!<br><br>

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) has 60,000 images for training and 10,000 images for testing.<br>
Each image is a greyscale (values ranging from 0 - black to 255 - white) image of 28x28 pixels.
<br><br>

In order to improve the performance of the neural network each image was also rotated to the left & right by 10 degrees, resulting in <b>180,000 training images</b> & <b>30,000 testing images</b>.<br><br>

The network itself has the following characteristics:<br>
- Input layer with 784 neurons (28x28 pixels)
- 1 hidden layer
- 200 neurons in hidden layer
- Output layer with 10 neurons (prediction of each number, e.g. 0-9)
<br><br>

The following hyperparameters can be changed:<br>
- number of neurons in hidden layer
- number of epochs
- learning rate
<br><br>

## Steps
In order to execute this python script, please follow the following steps:
<br>
1. Create a folder called <b>mnist_dataset</b> in the same directory as the <b>neural_network.py</b> file
2. Download both training and test dataset as csv files from [here](https://pjreddie.com/projects/mnist-in-csv/)
3. Put both files into the <b>mnist_dataset directory</b> (from <i>step 1</i>) as <b>mnist_train.csv</b> and <b>mnist_test.csv</b>
4. Execute the <b>generate_rotated_images.py</b> script, which will generate two more files under the <i>mnist_dataset</i> directory called <b>rotated_mnist_train.csv</b> and <b>rotated_mnist_test.csv</b> (containing for each original image 2 more images rotated to the left & right by 10 degrees)
5. Change the hyperparameters as you like in the <b>neural_network.py</b> file
6. Train & test the neural network with `python neural_network.py`
<br>
Once executed, the network will:<br>
1. Train
2. Test
3. Print out accuracy
4. Show learned numbers (made by a back-query for each label, e.g. 0-9)

